"""Transformer-trunk student for distillation experiments.

Architecture change from BetaOneNetwork:
  - Replace LayerNorm + 2x Linear+ReLU trunk with entity-tokenized transformer
  - Entities: player(1) + enemies(5) + context(1) + hand_cards(10) = 17 tokens + CLS
  - Each entity projected to hidden_dim via entity-specific Linear
  - 3-layer transformer encoder with n_heads=4
  - CLS token pooled for state representation
  - Existing C51 distributional value head + dot-product policy head unchanged

Inductive bias: attention learns cross-entity interactions directly (e.g.
"this card affects that enemy", "discard this to draw that"). v3's flat
concatenation + linear trunk cannot express these interactions as naturally.

Target param count: ~700K-900K (vs v3's 144K). The capacity hypothesis: if
v3's 144K-param arch can't represent MCTS-1000's decision function, a bigger
+ attentional arch might — especially when given fixed distillation targets
(not needing self-play exploration).

Usage:
    python -m scripts.distill_transformer train \\
        --dataset experiments/distill-v1/dataset.pkl \\
        --output experiments/distill-transformer-v1 \\
        --epochs 200 --lr-schedule cosine_warmup
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from sts2_solver.betaone.network import (
    STATE_DIM,
    ACTION_DIM,
    MAX_ACTIONS,
    MAX_HAND,
    BASE_STATE_DIM,
    CARD_STATS_DIM,
    CARD_EMBED_DIM,
    ACTION_HIDDEN,
)
from distill_c51 import (
    DistStudentNetwork,
    _two_hot_project, N_ATOMS, V_MIN, V_MAX,
    _policy_ce_loss, _value_c51_loss, _top1_accuracy,
)


# Entity split layout (must match betaone encode.rs base layout):
#   player:   [0, 25)
#   enemies:  [25, 120)  = 5 × 19
#   context:  [120, 156) = context(6) + relics(27) + hand_agg(3)
#   hand:     [156, 436) = 10 × 28 (hand card stats)
#   hand_mask:[436, 446) = 10
PLAYER_DIMS = 25
ENEMY_DIMS = 19
N_ENEMIES = 5
CONTEXT_DIMS = 36  # 6 context + 27 relics + 3 hand_agg
HAND_STATS_DIMS = 28


class TransformerStudentNetwork(DistStudentNetwork):
    """Transformer trunk + C51 distributional value head (inherits from DistStudent)."""

    def __init__(
        self,
        num_cards: int,
        hidden_dim: int = 128,
        n_transformer_layers: int = 3,
        n_heads: int = 4,
        ffn_hidden: int = 256,
        dropout: float = 0.1,
        **betaone_kwargs,
    ):
        # v3-default heads; trunk replaced below. DistStudent already handles
        # the C51 distributional value head.
        betaone_kwargs.setdefault("value_head_layers", 3)
        betaone_kwargs.setdefault("trunk_hidden", hidden_dim)
        betaone_kwargs.setdefault("policy_head_type", "dot_product")
        super().__init__(num_cards=num_cards, **betaone_kwargs)

        self.hidden_dim = hidden_dim

        # Remove parent's trunk (we replace it with transformer)
        del self.trunk

        # Entity projections → hidden_dim
        self.player_proj = nn.Linear(PLAYER_DIMS, hidden_dim)
        self.enemy_proj = nn.Linear(ENEMY_DIMS, hidden_dim)
        self.context_proj = nn.Linear(CONTEXT_DIMS, hidden_dim)
        # Hand token = card_embed + card_stats, projected to hidden_dim
        self.hand_proj_tf = nn.Linear(CARD_EMBED_DIM + HAND_STATS_DIMS, hidden_dim)

        # Type embeddings (learnable, one per entity type)
        # 4 types: player, enemy, context, hand
        self.type_embedding = nn.Embedding(4, hidden_dim)
        # CLS token for pooling
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Transformer encoder (pre-norm recommended for stability)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=ffn_hidden,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_transformer_layers)
        self.final_norm = nn.LayerNorm(hidden_dim)

    def _tokenize_state(self, state: torch.Tensor, hand_card_ids: torch.Tensor):
        """Split state into entity tokens, project, add type embeddings.

        Returns (tokens, padding_mask):
            tokens: (B, 17, hidden_dim)  — [player, enemies×5, context, hand×10]
            padding_mask: (B, 17)  — True where token should be masked (invalid hand slots)
        """
        B = state.shape[0]
        hidden = self.hidden_dim

        # Split state into entity chunks
        player = state[:, 0:PLAYER_DIMS]                             # (B, 25)
        enemies = state[:, PLAYER_DIMS:PLAYER_DIMS + N_ENEMIES * ENEMY_DIMS]
        enemies = enemies.view(B, N_ENEMIES, ENEMY_DIMS)              # (B, 5, 19)
        context = state[:, PLAYER_DIMS + N_ENEMIES * ENEMY_DIMS:BASE_STATE_DIM]  # (B, 36)
        hand_stats = state[:, BASE_STATE_DIM:BASE_STATE_DIM + MAX_HAND * HAND_STATS_DIMS]
        hand_stats = hand_stats.view(B, MAX_HAND, HAND_STATS_DIMS)    # (B, 10, 28)
        hand_mask_float = state[:, BASE_STATE_DIM + MAX_HAND * HAND_STATS_DIMS:]  # (B, 10)

        # Project to hidden
        player_tok = self.player_proj(player).unsqueeze(1)            # (B, 1, H)
        enemy_toks = self.enemy_proj(enemies)                         # (B, 5, H)
        context_tok = self.context_proj(context).unsqueeze(1)         # (B, 1, H)
        hand_embeds = self.card_embed(hand_card_ids.long())           # (B, 10, CARD_EMBED_DIM)
        hand_input = torch.cat([hand_embeds, hand_stats], dim=-1)     # (B, 10, CARD_EMBED_DIM + 28)
        hand_toks = self.hand_proj_tf(hand_input)                     # (B, 10, H)

        # Type embeddings: player(0), enemy(1), context(2), hand(3)
        device = state.device
        player_type = self.type_embedding(torch.tensor([0], device=device)).unsqueeze(0)  # (1,1,H)
        enemy_type = self.type_embedding(torch.tensor([1], device=device)).view(1, 1, -1).expand(B, N_ENEMIES, -1)
        context_type = self.type_embedding(torch.tensor([2], device=device)).unsqueeze(0)
        hand_type = self.type_embedding(torch.tensor([3], device=device)).view(1, 1, -1).expand(B, MAX_HAND, -1)

        player_tok = player_tok + player_type
        enemy_toks = enemy_toks + enemy_type
        context_tok = context_tok + context_type
        hand_toks = hand_toks + hand_type

        # Combine tokens: [player, enemies×5, context, hand×10] = 17 tokens
        entity_tokens = torch.cat([player_tok, enemy_toks, context_tok, hand_toks], dim=1)  # (B, 17, H)

        # Prepend CLS token → 18 tokens total
        cls_tok = self.cls_token.expand(B, -1, -1)                     # (B, 1, H)
        tokens = torch.cat([cls_tok, entity_tokens], dim=1)            # (B, 18, H)

        # Padding mask (True where invalid): CLS + player + 5 enemies + context always valid;
        # hand mask from state.
        valid_head = torch.ones(B, 1 + 1 + N_ENEMIES + 1, dtype=torch.bool, device=device)  # 8 valid
        hand_valid = hand_mask_float > 0.5                              # (B, 10)
        valid = torch.cat([valid_head, hand_valid], dim=1)              # (B, 18)
        padding_mask = ~valid                                           # True = mask out
        return tokens, padding_mask

    def _policy_and_value_from_h(self, h: torch.Tensor, action_features, action_mask, action_card_ids):
        """Apply policy head and distributional value head given trunk-like h."""
        action_embeds = self.card_embed(action_card_ids.long())
        action_input = torch.cat([action_embeds, action_features], dim=-1)
        if self.policy_head_type == "dot_product":
            query = self.policy_query(h)
            keys = self.action_encoder(action_input)
            logits = torch.bmm(keys, query.unsqueeze(-1)).squeeze(-1) / (ACTION_HIDDEN ** 0.5)
        else:  # mlp: per-action MLP over concat(h, action_input)
            n_actions = action_input.shape[1]
            h_b = h.unsqueeze(1).expand(-1, n_actions, -1)
            x = torch.cat([h_b, action_input], dim=-1)
            x = F.relu(self.policy_mlp_fc1(x))
            logits = self.policy_mlp_fc2(x).squeeze(-1)
        logits = logits.masked_fill(action_mask, -1e9)

        # Distributional value
        dist_logits = self.value_head(h)
        probs = F.softmax(dist_logits, dim=-1)
        value = (probs * self._get_atoms(h.device).unsqueeze(0)).sum(dim=-1, keepdim=True)
        return logits, value, dist_logits

    def forward(self, state, action_features, action_mask, hand_card_ids, action_card_ids):
        tokens, padding_mask = self._tokenize_state(state, hand_card_ids)
        attended = self.transformer(tokens, src_key_padding_mask=padding_mask)
        h = self.final_norm(attended[:, 0])  # CLS token
        logits, value, _ = self._policy_and_value_from_h(h, action_features, action_mask, action_card_ids)
        return logits, value

    def forward_with_value_dist(self, state, action_features, action_mask, hand_card_ids, action_card_ids):
        tokens, padding_mask = self._tokenize_state(state, hand_card_ids)
        attended = self.transformer(tokens, src_key_padding_mask=padding_mask)
        h = self.final_norm(attended[:, 0])
        logits, value, dist_logits = self._policy_and_value_from_h(h, action_features, action_mask, action_card_ids)
        return logits, value, dist_logits


def train_cmd(args):
    os.makedirs(args.output, exist_ok=True)
    with open(args.dataset, "rb") as f:
        data = pickle.load(f)
    meta = data.get("meta", {})
    N = len(data["states"])
    print(f"Dataset: {N} states")

    num_cards = 578
    if meta.get("teacher_checkpoint") and os.path.exists(meta["teacher_checkpoint"]):
        tc = torch.load(meta["teacher_checkpoint"], map_location="cpu", weights_only=False)
        num_cards = tc.get("num_cards", num_cards)

    network = TransformerStudentNetwork(
        num_cards=num_cards,
        hidden_dim=args.hidden_dim,
        n_transformer_layers=args.n_layers,
        n_heads=args.n_heads,
        ffn_hidden=args.ffn_hidden,
        dropout=args.dropout,
        policy_head_type=args.policy_head_type,
        policy_mlp_hidden=args.policy_mlp_hidden,
    ).to(args.device)
    print(
        f"Transformer student (hidden={args.hidden_dim}, layers={args.n_layers}, "
        f"heads={args.n_heads}, ffn={args.ffn_hidden}, policy={args.policy_head_type}"
        f"{'/'+str(args.policy_mlp_hidden) if args.policy_head_type=='mlp' else ''}): "
        f"{network.param_count():,} params"
    )

    states = torch.from_numpy(data["states"]).float()
    act_feat = torch.from_numpy(data["action_features"]).float()
    act_masks = torch.from_numpy(data["action_masks"]).bool()
    hand_ids = torch.from_numpy(data["hand_card_ids"]).long()
    action_ids = torch.from_numpy(data["action_card_ids"]).long()
    target_policies = torch.from_numpy(data["target_policies"]).float()
    target_values = torch.from_numpy(data["target_values"]).float()

    rng = np.random.RandomState(42)
    idx = rng.permutation(N)
    n_val = max(1, int(N * args.val_frac))
    val_idx = torch.from_numpy(idx[:n_val])
    train_idx = torch.from_numpy(idx[n_val:])
    print(f"Train: {len(train_idx)} | Val: {len(val_idx)}")

    optimizer = torch.optim.Adam(network.parameters(), lr=args.lr)
    history_path = os.path.join(args.output, "distill_history.jsonl")
    open(history_path, "a").close()

    def _lr_at_epoch(ep: int) -> float:
        if args.lr_schedule == "constant":
            return args.lr
        warmup = max(1, int(args.epochs * args.lr_warmup_frac))
        if ep <= warmup:
            return args.lr * (ep / warmup)
        progress = (ep - warmup) / max(1, args.epochs - warmup)
        import math as _m
        lr_min = args.lr * args.lr_min_frac
        return lr_min + 0.5 * (args.lr - lr_min) * (1 + _m.cos(_m.pi * progress))

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        cur_lr = _lr_at_epoch(epoch)
        for g in optimizer.param_groups:
            g["lr"] = cur_lr
        network.train()
        shuf = train_idx[torch.randperm(len(train_idx))]
        n_batches = (len(shuf) + args.batch_size - 1) // args.batch_size
        train_pol = train_val = train_top1 = 0.0
        steps = 0
        for bi in range(n_batches):
            bs = shuf[bi * args.batch_size : (bi + 1) * args.batch_size]
            s = states[bs].to(args.device)
            af = act_feat[bs].to(args.device)
            am = act_masks[bs].to(args.device)
            hi = hand_ids[bs].to(args.device)
            ai = action_ids[bs].to(args.device)
            tp = target_policies[bs].to(args.device)
            tv = target_values[bs].to(args.device)

            logits, _ev, dist_logits = network.forward_with_value_dist(s, af, am, hi, ai)
            pol_loss = _policy_ce_loss(logits, tp, am)
            val_loss = _value_c51_loss(dist_logits, tv)
            loss = pol_loss + args.value_coef * val_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(network.parameters(), 1.0)
            optimizer.step()

            train_pol += pol_loss.item()
            train_val += val_loss.item()
            train_top1 += _top1_accuracy(logits.detach(), tp)
            steps += 1
        train_pol /= steps; train_val /= steps; train_top1 /= steps

        network.eval()
        with torch.no_grad():
            val_pol = val_val = val_top1 = 0.0
            v_steps = 0
            for bi in range((len(val_idx) + args.batch_size - 1) // args.batch_size):
                bs = val_idx[bi * args.batch_size : (bi + 1) * args.batch_size]
                s = states[bs].to(args.device); af = act_feat[bs].to(args.device)
                am = act_masks[bs].to(args.device); hi = hand_ids[bs].to(args.device)
                ai = action_ids[bs].to(args.device); tp = target_policies[bs].to(args.device)
                tv = target_values[bs].to(args.device)
                logits, _ev, dist_logits = network.forward_with_value_dist(s, af, am, hi, ai)
                val_pol += _policy_ce_loss(logits, tp, am).item()
                val_val += _value_c51_loss(dist_logits, tv).item()
                val_top1 += _top1_accuracy(logits, tp)
                v_steps += 1
            val_pol /= max(v_steps, 1); val_val /= max(v_steps, 1); val_top1 /= max(v_steps, 1)

        dt = time.time() - t0
        print(
            f"Epoch {epoch:3d}: lr={cur_lr:.5f} | train pol={train_pol:.4f} c51={train_val:.4f} top1={train_top1:.3f} | "
            f"val pol={val_pol:.4f} c51={val_val:.4f} top1={val_top1:.3f} | {dt:.1f}s",
            flush=True,
        )
        with open(history_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "epoch": epoch, "lr": cur_lr,
                "train_pol_loss": train_pol, "train_val_loss": train_val, "train_top1": train_top1,
                "val_pol_loss": val_pol, "val_val_loss": val_val, "val_top1": val_top1,
                "time_s": dt, "timestamp": time.time(),
            }) + "\n")
        if epoch % args.save_every == 0 or epoch == args.epochs:
            ckpt_path = os.path.join(args.output, f"student_epoch{epoch}.pt")
            arch = network.arch_meta()
            arch["distributional_value"] = True
            arch["c51_atoms"] = N_ATOMS
            arch["c51_v_min"] = V_MIN
            arch["c51_v_max"] = V_MAX
            arch["transformer_trunk"] = True
            arch["transformer_hidden"] = args.hidden_dim
            arch["transformer_layers"] = args.n_layers
            arch["transformer_heads"] = args.n_heads
            arch["transformer_ffn"] = args.ffn_hidden
            torch.save({
                "gen": epoch, "epoch": epoch, "arch_meta": arch,
                "model_state_dict": network.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "num_cards": num_cards, "distill_meta": meta,
            }, ckpt_path)
            latest = os.path.join(args.output, "betaone_latest.pt")
            torch.save(torch.load(ckpt_path, map_location="cpu", weights_only=False), latest)
            print(f"  saved {ckpt_path}")

        # Optional per-epoch eval harness so we get a dense P/V/MCTS trajectory
        # (not just end-of-training snapshot). Nyquist: without dense sampling we
        # can't see if eval peaks mid-training and decays by the end.
        if args.eval_every > 0 and epoch % args.eval_every == 0:
            # Need latest.pt to exist for eval harness to load from.
            latest_path = os.path.join(args.output, "betaone_latest.pt")
            if not os.path.exists(latest_path):
                # Save a temporary latest.pt if save didn't trigger this epoch.
                arch = network.arch_meta()
                arch["distributional_value"] = True
                arch["c51_atoms"] = N_ATOMS
                arch["c51_v_min"] = V_MIN
                arch["c51_v_max"] = V_MAX
                arch["transformer_trunk"] = True
                arch["transformer_hidden"] = args.hidden_dim
                arch["transformer_layers"] = args.n_layers
                arch["transformer_heads"] = args.n_heads
                arch["transformer_ffn"] = args.ffn_hidden
                torch.save({
                    "gen": epoch, "epoch": epoch, "arch_meta": arch,
                    "model_state_dict": network.state_dict(),
                    "num_cards": num_cards, "distill_meta": meta,
                }, latest_path)
            # Monkey-patch BetaOneNetwork → TransformerStudentNetwork for eval.
            import sts2_solver.betaone.eval as _ev
            import sts2_solver.betaone.network as _nw
            _orig_bn = _ev.BetaOneNetwork
            _ev.BetaOneNetwork = TransformerStudentNetwork
            _nw.BetaOneNetwork = TransformerStudentNetwork
            try:
                from sts2_solver.betaone.eval import run_eval as _re, run_value_eval as _rve, run_mcts_eval as _rme
                from sts2_solver.betaone.suite import compute_eval_suite, suite_id as _sid
                bench_dir = os.path.join(args.output, "benchmarks")
                os.makedirs(bench_dir, exist_ok=True)
                sid = _sid(compute_eval_suite())
                ts = time.time()
                pol = _re(latest_path)
                with open(os.path.join(bench_dir, "eval.jsonl"), "a", encoding="utf-8") as f:
                    f.write(json.dumps({
                        "suite": sid, "timestamp": ts, "gen": epoch,
                        "passed": pol["passed"], "total": pol["total"],
                        "score": round(pol["passed"] / max(pol["total"], 1), 4),
                        "end_turn_avg": pol.get("end_turn_avg"),
                        "end_turn_high": pol.get("end_turn_high", 0),
                        "bad_count": pol.get("bad_count"),
                        "conf_bad": pol.get("conf_bad"),
                        "close_bad": pol.get("close_bad"),
                        "conf_clean": pol.get("conf_clean"),
                        "by_category": pol.get("by_category", {}),
                    }) + "\n")
                val = _rve(latest_path)
                with open(os.path.join(bench_dir, "value_eval.jsonl"), "a", encoding="utf-8") as f:
                    f.write(json.dumps({
                        "suite": sid, "timestamp": ts, "gen": epoch,
                        "passed": val["passed"], "total": val["total"],
                        "score": round(val["passed"] / max(val["total"], 1), 4),
                        "by_category": val.get("by_category", {}),
                    }) + "\n")
                mev = _rme(latest_path)
                with open(os.path.join(bench_dir, "mcts_eval.jsonl"), "a", encoding="utf-8") as f:
                    f.write(json.dumps({
                        "suite": sid, "timestamp": ts, "gen": epoch,
                        "total": mev["total"],
                        "clean": mev["clean"], "echo": mev["echo"],
                        "fixed": mev["fixed"], "broke": mev["broke"],
                        "rescue_rate": mev["rescue_rate"],
                    }) + "\n")
                print(
                    f"  eval@{epoch}: P={pol['passed']}/{pol['total']} "
                    f"V={val['passed']}/{val['total']} "
                    f"rescue={mev['rescue_rate']*100:+.0f}%",
                    flush=True,
                )
            finally:
                _ev.BetaOneNetwork = _orig_bn
                _nw.BetaOneNetwork = _orig_bn
            network.train()  # restore training mode after eval


def main():
    p = argparse.ArgumentParser()
    sp = p.add_subparsers(dest="cmd", required=True)

    t = sp.add_parser("train")
    t.add_argument("--dataset", required=True)
    t.add_argument("--output", required=True)
    t.add_argument("--epochs", type=int, default=200)
    t.add_argument("--batch-size", type=int, default=256)  # smaller batch for bigger net
    t.add_argument("--lr", type=float, default=3e-4)
    t.add_argument("--value-coef", type=float, default=1.0)
    t.add_argument("--val-frac", type=float, default=0.05)
    t.add_argument("--save-every", type=int, default=10)
    t.add_argument("--eval-every", type=int, default=0,
                   help="Run P-Eval+V-Eval+MCTS-eval every N epochs (0 = off)")
    t.add_argument("--device", default="cpu")
    t.add_argument("--lr-schedule", default="cosine_warmup", choices=["constant", "cosine_warmup"])
    t.add_argument("--lr-warmup-frac", type=float, default=0.05)
    t.add_argument("--lr-min-frac", type=float, default=0.1)
    # Transformer-specific
    t.add_argument("--hidden-dim", type=int, default=128)
    t.add_argument("--n-layers", type=int, default=3)
    t.add_argument("--n-heads", type=int, default=4)
    t.add_argument("--ffn-hidden", type=int, default=256)
    t.add_argument("--dropout", type=float, default=0.1)
    t.add_argument("--policy-head-type", default="dot_product", choices=["dot_product", "mlp"])
    t.add_argument("--policy-mlp-hidden", type=int, default=64)
    t.set_defaults(func=train_cmd)

    args = p.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    args.func(args)


if __name__ == "__main__":
    main()
