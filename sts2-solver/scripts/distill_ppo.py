"""PPO fine-tuning of a distilled BetaOne student.

Loads a transformer+MLP distillation checkpoint (near v3-parity imitation
ceiling) and fine-tunes via PPO with combat WR as reward. Policy gradient
is NOT bounded by teacher performance, so this is the only mechanism that
can exceed v3's WR.

Uses existing Rust `collect_betaone_rollouts` for fast parallel rollouts
(already returns log_probs, values, rewards, dones). Applies PPO clipped
surrogate + GAE + entropy bonus + KL regularization toward frozen reference
(the initial distilled student) to prevent catastrophic forgetting.

Usage:
    python -m scripts.distill_ppo \\
        --init-checkpoint experiments/distill-transformer-mlp-v1/student_epoch25.pt \\
        --card-vocab C:/coding-projects/sts2-reanalyse-v3/.../card_vocab.json \\
        --encounter-set uber-decks-v1 \\
        --output experiments/ppo-v1 \\
        --num-gens 100 --combats-per-gen 256
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import sts2_engine

from sts2_solver.betaone.network import export_onnx, STATE_DIM, ACTION_DIM, MAX_ACTIONS, MAX_HAND, network_kwargs_from_meta
from sts2_solver.betaone.data_utils import load_solver_json, build_monster_data_json, sample_combat_batches
from sts2_solver.betaone.encounter_set import load_encounter_set
from sts2_solver.betaone.ppo import compute_gae


def _load_network(ckpt_path: str, device: str):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    arch_meta = ckpt.get("arch_meta") or {}
    kwargs = network_kwargs_from_meta(arch_meta)
    num_cards = ckpt.get("num_cards") or arch_meta.get("num_cards") or 578

    is_transformer = bool(arch_meta.get("transformer_trunk"))
    is_distributional = bool(arch_meta.get("distributional_value"))

    if is_transformer:
        from distill_transformer import TransformerStudentNetwork
        net = TransformerStudentNetwork(num_cards=num_cards, **kwargs).to(device)
        tag = "[transformer]"
    elif is_distributional:
        from distill_c51 import DistStudentNetwork
        net = DistStudentNetwork(num_cards=num_cards, **kwargs).to(device)
        tag = "[distributional]"
    else:
        from sts2_solver.betaone.network import BetaOneNetwork
        net = BetaOneNetwork(num_cards=num_cards, **kwargs).to(device)
        tag = "[scalar]"
    net.load_state_dict(ckpt["model_state_dict"])
    net.train()
    print(f"Loaded {tag} from {Path(ckpt_path).name}: {net.param_count():,} params")
    return net, arch_meta, num_cards


def _compute_log_probs_and_entropy(logits, chosen, mask):
    # Mask invalid actions (mask True = invalid)
    logits = logits.masked_fill(mask, -1e9)
    log_probs = F.log_softmax(logits, dim=-1)
    probs = F.softmax(logits, dim=-1)
    chosen_log_probs = log_probs.gather(1, chosen.unsqueeze(1)).squeeze(1)
    # Entropy only over valid actions
    entropy = -(probs * log_probs).nan_to_num(0.0).sum(dim=-1)
    return chosen_log_probs, log_probs, entropy


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--init-checkpoint", required=True)
    p.add_argument("--card-vocab", required=True)
    p.add_argument("--encounter-set", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--num-gens", type=int, default=100)
    p.add_argument("--combats-per-gen", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--clip-ratio", type=float, default=0.2)
    p.add_argument("--value-coef", type=float, default=0.5)
    p.add_argument("--entropy-coef", type=float, default=0.01)
    p.add_argument("--kl-coef", type=float, default=0.05,
                   help="KL regularization toward frozen reference policy (prevents forgetting)")
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--ppo-epochs", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--save-every", type=int, default=5)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    os.makedirs(args.output, exist_ok=True)

    network, arch_meta, num_cards = _load_network(args.init_checkpoint, args.device)
    reference = copy.deepcopy(network).eval()
    for p_ in reference.parameters():
        p_.requires_grad = False
    print(f"Reference (frozen) copy made: {reference.param_count():,} params")

    optimizer = torch.optim.Adam(network.parameters(), lr=args.lr)

    with open(args.card_vocab, encoding="utf-8") as f:
        card_vocab_json = f.read()
    monster_json = build_monster_data_json()
    profiles_json = load_solver_json("enemy_profiles.json")
    encounter_set = load_encounter_set(args.encounter_set)
    print(f"Encounter set: {args.encounter_set} ({len(encounter_set)} encounters)")
    print(f"Starting PPO: {args.num_gens} gens × {args.combats_per_gen} combats, lr={args.lr}, kl={args.kl_coef}")

    history_path = os.path.join(args.output, "ppo_history.jsonl")
    progress_path = os.path.join(args.output, "betaone_progress.json")
    onnx_dir = os.path.join(args.output, "onnx")
    os.makedirs(onnx_dir, exist_ok=True)

    for gen in range(1, args.num_gens + 1):
        t0 = time.time()
        # Export current network for rollouts
        onnx_path = export_onnx(network, onnx_dir)

        # Collect rollouts via Rust engine
        batches = sample_combat_batches(encounter_set, args.combats_per_gen, gen)
        seeds = [gen * 100_000 + i for i in range(args.combats_per_gen)]

        all_states, all_af, all_am, all_hi, all_ai = [], [], [], [], []
        all_chosen, all_old_lp, all_values = [], [], []
        all_rewards, all_dones = [], []
        all_outcomes = []
        seed_idx = 0
        for b_enc, b_dks, b_rels, b_hp, b_count in batches:
            if not b_enc:
                continue
            b_seeds = [seeds[seed_idx + i] if seed_idx + i < len(seeds)
                       else gen * 100_000 + seed_idx + i for i in range(b_count)]
            seed_idx += b_count
            rollout = sts2_engine.collect_betaone_rollouts(
                encounters_json=json.dumps(b_enc),
                decks_json=json.dumps(b_dks),
                player_hp=b_hp,
                player_max_hp=70,
                player_max_energy=3,
                relics_json=json.dumps(b_rels),
                potions_json="[]",
                monster_data_json=monster_json,
                enemy_profiles_json=profiles_json,
                onnx_path=onnx_path,
                temperature=args.temperature,
                seeds=b_seeds,
                gen_id=gen,
                card_vocab_json=card_vocab_json,
            )
            n = rollout["total_steps"]
            if n == 0:
                continue
            all_states.append(np.array(rollout["states"], dtype=np.float32).reshape(-1, STATE_DIM))
            all_af.append(np.array(rollout["action_features"], dtype=np.float32).reshape(-1, MAX_ACTIONS, ACTION_DIM))
            all_am.append(np.array(rollout["action_masks"]).reshape(-1, MAX_ACTIONS))
            all_hi.append(np.array(rollout["hand_card_ids"], dtype=np.int64).reshape(-1, MAX_HAND))
            all_ai.append(np.array(rollout["action_card_ids"], dtype=np.int64).reshape(-1, MAX_ACTIONS))
            all_chosen.append(np.array(rollout["chosen_indices"], dtype=np.int64))
            all_old_lp.append(np.array(rollout["log_probs"], dtype=np.float32))
            all_values.append(np.array(rollout["values"], dtype=np.float32))
            all_rewards.append(np.array(rollout["rewards"], dtype=np.float32))
            all_dones.append(np.array(rollout["dones"], dtype=bool))
            all_outcomes.extend(rollout["outcomes"])

        if not all_states:
            print(f"Gen {gen}: no rollouts, skipping"); continue

        states = np.concatenate(all_states, axis=0)
        af = np.concatenate(all_af, axis=0)
        am = np.concatenate(all_am, axis=0)
        hi = np.concatenate(all_hi, axis=0)
        ai = np.concatenate(all_ai, axis=0)
        chosen = np.concatenate(all_chosen, axis=0)
        old_lp = np.concatenate(all_old_lp, axis=0)
        values = np.concatenate(all_values, axis=0)
        rewards = np.concatenate(all_rewards, axis=0)
        dones = np.concatenate(all_dones, axis=0)

        # GAE
        advantages, returns_ = compute_gae(rewards, values, dones, gamma=args.gamma, lam=args.lam)
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # To tensors
        states_t = torch.from_numpy(states).to(args.device)
        af_t = torch.from_numpy(af).to(args.device)
        am_t = torch.from_numpy(am).bool().to(args.device)
        hi_t = torch.from_numpy(hi).long().to(args.device)
        ai_t = torch.from_numpy(ai).long().to(args.device)
        chosen_t = torch.from_numpy(chosen).long().to(args.device)
        old_lp_t = torch.from_numpy(old_lp).float().to(args.device)
        adv_t = torch.from_numpy(advantages).float().to(args.device)
        ret_t = torch.from_numpy(returns_).float().to(args.device)

        # PPO update with KL regularization
        T = len(states_t)
        indices = np.arange(T)
        total_pol, total_val, total_ent, total_kl = 0.0, 0.0, 0.0, 0.0
        n_updates = 0
        network.train()
        for _ in range(args.ppo_epochs):
            np.random.shuffle(indices)
            for start in range(0, T, args.batch_size):
                end = min(start + args.batch_size, T)
                b = torch.from_numpy(indices[start:end]).long().to(args.device)

                logits, vals_pred = network(states_t[b], af_t[b], am_t[b], hi_t[b], ai_t[b])
                new_lp, new_log_probs_full, entropy = _compute_log_probs_and_entropy(logits, chosen_t[b], am_t[b])

                # Clipped surrogate
                ratio = torch.exp(new_lp - old_lp_t[b])
                surr1 = ratio * adv_t[b]
                surr2 = torch.clamp(ratio, 1 - args.clip_ratio, 1 + args.clip_ratio) * adv_t[b]
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(vals_pred.squeeze(-1), ret_t[b])
                entropy_loss = -entropy.mean()

                # KL regularization toward reference (frozen distilled policy)
                # Prevents catastrophic forgetting: policy updates toward higher
                # reward but stays close to the distilled policy that already
                # has decent combat skill.
                with torch.no_grad():
                    ref_logits, _ = reference(states_t[b], af_t[b], am_t[b], hi_t[b], ai_t[b])
                    ref_logits = ref_logits.masked_fill(am_t[b], -1e9)
                    ref_log_probs = F.log_softmax(ref_logits, dim=-1)
                # Forward KL: KL(ref || current) — keeps current near ref
                # Uses full per-state distribution, not just chosen action
                kl = (torch.exp(ref_log_probs) * (ref_log_probs - new_log_probs_full)).nan_to_num(0.0).sum(dim=-1).mean()

                loss = (policy_loss
                        + args.value_coef * value_loss
                        + args.entropy_coef * entropy_loss
                        + args.kl_coef * kl)

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(network.parameters(), args.max_grad_norm)
                optimizer.step()

                total_pol += policy_loss.item()
                total_val += value_loss.item()
                total_ent += entropy.mean().item()
                total_kl += kl.item()
                n_updates += 1

        total_pol /= n_updates; total_val /= n_updates; total_ent /= n_updates; total_kl /= n_updates

        # Stats
        n_wins = sum(1 for o in all_outcomes if o == "win")
        win_rate = n_wins / max(len(all_outcomes), 1)
        dt = time.time() - t0

        row = {
            "gen": gen,
            "win_rate": win_rate,
            "policy_loss": total_pol,
            "value_loss": total_val,
            "entropy": total_ent,
            "kl_from_ref": total_kl,
            "n_steps": int(T),
            "n_combats": len(all_outcomes),
            "gen_time": dt,
            "timestamp": time.time(),
        }
        with open(history_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        with open(progress_path, "w", encoding="utf-8") as f:
            json.dump({**row, "phase": "PPO", "num_generations": args.num_gens}, f, indent=2)
        print(
            f"Gen {gen}/{args.num_gens}: WR={win_rate:.3f} pol={total_pol:+.4f} "
            f"val={total_val:.4f} ent={total_ent:.3f} kl={total_kl:.4f} "
            f"T={T} {dt:.0f}s",
            flush=True,
        )

        # Save checkpoint
        if gen % args.save_every == 0 or gen == args.num_gens:
            ckpt_path = os.path.join(args.output, f"betaone_gen{gen}.pt")
            # Reuse the init checkpoint's arch_meta rather than network.arch_meta():
            # the TransformerStudentNetwork/DistStudentNetwork's arch_meta() omits
            # the transformer_trunk/distributional_value discriminator flags that
            # downstream eval/benchmark code needs to pick the right class.
            torch.save({
                "gen": gen, "arch_meta": arch_meta,
                "model_state_dict": network.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "num_cards": num_cards, "win_rate": win_rate,
            }, ckpt_path)
            latest = os.path.join(args.output, "betaone_latest.pt")
            torch.save(torch.load(ckpt_path, map_location="cpu", weights_only=False), latest)


if __name__ == "__main__":
    main()
