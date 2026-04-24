"""Distill student with C51-style distributional value head.

Uses the same supervised dataset as distill_train.py (MCTS-2000 targets
from v3 g88). Only change: value head outputs a 51-atom distribution
over return values instead of a scalar. At inference, returns E[V] so
the eval harness sees a scalar drop-in.

Hypothesis: distributional value is a richer function approximator
(per C51 / Muesli) and may help even when targets are scalar.

Two-hot projection of scalar MCTS value onto atoms is standard.

Usage:
    python -m scripts.distill_c51 train \\
        --dataset experiments/distill-v1/dataset.pkl \\
        --output experiments/distill-c51-v1 \\
        --epochs 100 --device cpu

    python -m scripts.distill_c51 eval \\
        --checkpoint experiments/distill-c51-v1/betaone_latest.pt

    python -m scripts.distill_c51 benchmark \\
        --checkpoint experiments/distill-c51-v1/betaone_latest.pt \\
        --compare-with <other.pt> --repeats 50
"""
from __future__ import annotations

import argparse
import json
import math
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

from sts2_solver.betaone.network import (
    BetaOneNetwork,
    STATE_DIM,
    ACTION_DIM,
    MAX_ACTIONS,
    MAX_HAND,
    BASE_STATE_DIM,
    CARD_STATS_DIM,
    HAND_PROJ_DIM,
    ACTION_HIDDEN,
    network_kwargs_from_meta,
)


# C51 hyperparameters
N_ATOMS = 51
V_MIN = -1.2
V_MAX = 1.5


def _atoms() -> torch.Tensor:
    return torch.linspace(V_MIN, V_MAX, N_ATOMS)


def _two_hot_project(scalar_targets: torch.Tensor) -> torch.Tensor:
    """Project scalar targets onto the atom support as a two-hot distribution.

    scalar_targets: (B,) in any range — clamped to [V_MIN, V_MAX].
    Returns: (B, N_ATOMS) soft distribution that sums to 1 per row and whose
             expectation equals the (clamped) scalar target.
    """
    B = scalar_targets.shape[0]
    atoms = _atoms().to(scalar_targets.device)
    delta_z = (V_MAX - V_MIN) / (N_ATOMS - 1)
    v = scalar_targets.clamp(V_MIN, V_MAX)
    # Continuous atom index in [0, N_ATOMS - 1]
    b = (v - V_MIN) / delta_z
    b_low = b.floor().long().clamp(0, N_ATOMS - 1)
    b_high = (b_low + 1).clamp(0, N_ATOMS - 1)
    w_high = (b - b_low.float())
    w_low = 1.0 - w_high
    # When b_low == b_high (v exactly at top atom), put all mass on that atom
    at_top = (b_low == b_high)
    w_low = torch.where(at_top, torch.ones_like(w_low), w_low)
    w_high = torch.where(at_top, torch.zeros_like(w_high), w_high)
    target = torch.zeros(B, N_ATOMS, device=scalar_targets.device)
    target.scatter_add_(1, b_low.unsqueeze(1), w_low.unsqueeze(1))
    target.scatter_add_(1, b_high.unsqueeze(1), w_high.unsqueeze(1))
    return target


class DistStudentNetwork(BetaOneNetwork):
    """BetaOne with a C51-style distributional value head.

    Keeps the parent's state_dict shape-compatible EXCEPT the value_head
    submodule which is overwritten to a hidden→N_ATOMS Linear.
    """

    def __init__(self, num_cards: int, **kwargs):
        super().__init__(num_cards=num_cards, **kwargs)
        # Replace value head with distributional
        hidden = self.trunk_hidden
        # Keep same internal depth as scalar head for fair capacity comparison,
        # but final output is N_ATOMS not 1.
        vhl = self.value_head_layers
        if vhl == 0:
            self.value_head = nn.Sequential(
                nn.Linear(hidden, 32), nn.ReLU(),
                nn.Linear(32, N_ATOMS),
            )
        elif vhl == 1:
            self.value_head = nn.Sequential(
                nn.Linear(hidden, 64), nn.ReLU(),
                nn.Linear(64, N_ATOMS),
            )
        elif vhl == 3:
            self.value_head = nn.Sequential(
                nn.Linear(hidden, 256), nn.ReLU(),
                nn.Linear(256, 128), nn.ReLU(),
                nn.Linear(128, 64), nn.ReLU(),
                nn.Linear(64, N_ATOMS),
            )
        else:
            raise ValueError(f"unsupported vhl={vhl}")
        self._atoms_buf = None  # lazy: move to device on first forward

    def _get_atoms(self, device):
        if self._atoms_buf is None or self._atoms_buf.device != device:
            self._atoms_buf = _atoms().to(device)
        return self._atoms_buf

    def forward(self, state, action_features, action_mask, hand_card_ids, action_card_ids):
        """Returns (logits, E[V]) — scalar value for drop-in compat."""
        # Replicate parent's forward up to `value = self.value_head(h)` but
        # compute E[V] from the distributional output instead of returning
        # the raw head scalar.
        B = state.shape[0]
        base = state[:, :BASE_STATE_DIM]
        hand_raw = state[:, BASE_STATE_DIM:BASE_STATE_DIM + MAX_HAND * CARD_STATS_DIM]
        hand_raw = hand_raw.view(B, MAX_HAND, CARD_STATS_DIM)
        hand_mask_float = state[:, BASE_STATE_DIM + MAX_HAND * CARD_STATS_DIM:]

        hand_embeds = self.card_embed(hand_card_ids.long())
        hand_input = torch.cat([hand_embeds, hand_raw], dim=-1)
        h_cards = self.hand_proj(hand_input)
        Q = self.attn_q(h_cards); K = self.attn_k(h_cards); V = self.attn_v(h_cards)
        scores = torch.bmm(Q, K.transpose(1, 2)) / (HAND_PROJ_DIM ** 0.5)
        mask_2d = hand_mask_float.unsqueeze(1) * hand_mask_float.unsqueeze(2)
        scores = scores.masked_fill(mask_2d == 0, -1e9)
        attn_weights = F.softmax(scores, dim=-1)
        attended = torch.bmm(attn_weights, V)
        mask_expanded = hand_mask_float.unsqueeze(-1)
        hand_pooled = (attended * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)

        combined = torch.cat([base, hand_pooled], dim=1)
        h = self.trunk(combined)

        action_embeds = self.card_embed(action_card_ids.long())
        action_input = torch.cat([action_embeds, action_features], dim=-1)
        if self.policy_head_type == "dot_product":
            query = self.policy_query(h)
            keys = self.action_encoder(action_input)
            logits = torch.bmm(keys, query.unsqueeze(-1)).squeeze(-1) / (ACTION_HIDDEN ** 0.5)
        else:
            n_actions = action_input.shape[1]
            h_b = h.unsqueeze(1).expand(-1, n_actions, -1)
            x = torch.cat([h_b, action_input], dim=-1)
            x = F.relu(self.policy_mlp_fc1(x))
            logits = self.policy_mlp_fc2(x).squeeze(-1)
        logits = logits.masked_fill(action_mask, -1e9)

        # Distributional value: hidden → N_ATOMS
        dist_logits = self.value_head(h)           # (B, N_ATOMS)
        probs = F.softmax(dist_logits, dim=-1)     # (B, N_ATOMS)
        atoms = self._get_atoms(h.device)          # (N_ATOMS,)
        expected_v = (probs * atoms.unsqueeze(0)).sum(dim=-1, keepdim=True)  # (B, 1)
        return logits, expected_v

    def forward_with_dist(self, state, action_features, action_mask, hand_card_ids, action_card_ids):
        """Same as forward but also returns the raw N_ATOMS logits for CE loss."""
        logits, expected_v = self.forward(state, action_features, action_mask, hand_card_ids, action_card_ids)
        # Re-run value head trunk path to get dist_logits
        # For efficiency, inline it: we already computed h but don't have it here.
        # Simplest: recompute via forward hooks is overkill; just duplicate trunk path.
        B = state.shape[0]
        base = state[:, :BASE_STATE_DIM]
        hand_raw = state[:, BASE_STATE_DIM:BASE_STATE_DIM + MAX_HAND * CARD_STATS_DIM].view(B, MAX_HAND, CARD_STATS_DIM)
        hand_mask_float = state[:, BASE_STATE_DIM + MAX_HAND * CARD_STATS_DIM:]
        hand_embeds = self.card_embed(hand_card_ids.long())
        hand_input = torch.cat([hand_embeds, hand_raw], dim=-1)
        h_cards = self.hand_proj(hand_input)
        Q = self.attn_q(h_cards); K = self.attn_k(h_cards); V = self.attn_v(h_cards)
        scores = torch.bmm(Q, K.transpose(1, 2)) / (HAND_PROJ_DIM ** 0.5)
        mask_2d = hand_mask_float.unsqueeze(1) * hand_mask_float.unsqueeze(2)
        scores = scores.masked_fill(mask_2d == 0, -1e9)
        attn_weights = F.softmax(scores, dim=-1)
        attended = torch.bmm(attn_weights, V)
        mask_expanded = hand_mask_float.unsqueeze(-1)
        hand_pooled = (attended * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
        combined = torch.cat([base, hand_pooled], dim=1)
        h = self.trunk(combined)
        dist_logits = self.value_head(h)  # (B, N_ATOMS)
        return logits, expected_v, dist_logits


def _policy_ce_loss(logits, targets, mask):
    log_probs = F.log_softmax(logits, dim=-1)
    return -(targets * log_probs).sum(dim=-1).mean()


def _value_c51_loss(dist_logits, scalar_targets):
    """CE(predicted dist, two-hot projection of scalar target)."""
    target_dist = _two_hot_project(scalar_targets)
    log_probs = F.log_softmax(dist_logits, dim=-1)
    return -(target_dist * log_probs).sum(dim=-1).mean()


def _top1_accuracy(logits, targets):
    pred = logits.argmax(dim=-1)
    true = targets.argmax(dim=-1)
    return (pred == true).float().mean().item()


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

    kw = dict(
        value_head_layers=args.value_head_layers,
        trunk_layers=2, trunk_hidden=128,
        policy_head_type=args.policy_head_type,
        policy_mlp_hidden=args.policy_mlp_hidden,
    )
    network = DistStudentNetwork(num_cards=num_cards, **kw).to(args.device)
    print(f"Student (C51, {N_ATOMS} atoms, [{V_MIN}, {V_MAX}], policy={args.policy_head_type}"
          f"{'/'+str(args.policy_mlp_hidden) if args.policy_head_type=='mlp' else ''}): "
          f"{network.param_count():,} params")

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

    # Optional cosine LR schedule with warmup. lr_schedule='constant' (default)
    # keeps args.lr flat. 'cosine_warmup' warms up for lr_warmup_frac of epochs
    # then decays to args.lr * lr_min_frac via half-cosine.
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

            logits, _expected_v, dist_logits = network.forward_with_dist(s, af, am, hi, ai)
            pol_loss = _policy_ce_loss(logits, tp, am)
            val_loss = _value_c51_loss(dist_logits, tv)
            loss = pol_loss + args.value_coef * val_loss

            optimizer.zero_grad()
            loss.backward()
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
                logits, _ev, dist_logits = network.forward_with_dist(s, af, am, hi, ai)
                val_pol += _policy_ce_loss(logits, tp, am).item()
                val_val += _value_c51_loss(dist_logits, tv).item()
                val_top1 += _top1_accuracy(logits, tp)
                v_steps += 1
            val_pol /= max(v_steps, 1); val_val /= max(v_steps, 1); val_top1 /= max(v_steps, 1)

        dt = time.time() - t0
        print(
            f"Epoch {epoch:3d}: train pol={train_pol:.4f} c51={train_val:.4f} top1={train_top1:.3f} | "
            f"val pol={val_pol:.4f} c51={val_val:.4f} top1={val_top1:.3f} | {dt:.1f}s",
            flush=True,
        )
        with open(history_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "epoch": epoch, "train_pol_loss": train_pol, "train_val_loss": train_val,
                "train_top1": train_top1, "val_pol_loss": val_pol, "val_val_loss": val_val,
                "val_top1": val_top1, "time_s": dt, "timestamp": time.time(),
            }) + "\n")
        if epoch % args.save_every == 0 or epoch == args.epochs:
            ckpt_path = os.path.join(args.output, f"student_epoch{epoch}.pt")
            arch = network.arch_meta()
            arch["distributional_value"] = True
            arch["c51_atoms"] = N_ATOMS
            arch["c51_v_min"] = V_MIN
            arch["c51_v_max"] = V_MAX
            torch.save({
                "gen": epoch, "epoch": epoch, "arch_meta": arch,
                "model_state_dict": network.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "num_cards": num_cards, "distill_meta": meta,
            }, ckpt_path)
            latest = os.path.join(args.output, "betaone_latest.pt")
            torch.save(torch.load(ckpt_path, map_location="cpu", weights_only=False), latest)
            print(f"  saved {ckpt_path}")


def main():
    p = argparse.ArgumentParser()
    sp = p.add_subparsers(dest="cmd", required=True)

    t = sp.add_parser("train")
    t.add_argument("--dataset", required=True)
    t.add_argument("--output", required=True)
    t.add_argument("--epochs", type=int, default=100)
    t.add_argument("--batch-size", type=int, default=512)
    t.add_argument("--lr", type=float, default=3e-4)
    t.add_argument("--value-coef", type=float, default=1.0)
    t.add_argument("--val-frac", type=float, default=0.05)
    t.add_argument("--save-every", type=int, default=5)
    t.add_argument("--device", default="cpu")
    t.add_argument("--lr-schedule", default="constant", choices=["constant", "cosine_warmup"])
    t.add_argument("--lr-warmup-frac", type=float, default=0.05)
    t.add_argument("--lr-min-frac", type=float, default=0.1)
    t.add_argument("--policy-head-type", default="dot_product", choices=["dot_product", "mlp"])
    t.add_argument("--policy-mlp-hidden", type=int, default=64)
    t.add_argument("--value-head-layers", type=int, default=3)
    t.set_defaults(func=train_cmd)

    args = p.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    args.func(args)


if __name__ == "__main__":
    main()
