"""BetaOne training orchestrator.

Usage:
    python -m sts2_solver.betaone.train [--generations N] [--combats N]

Loop:
    1. Export network → ONNX
    2. Collect rollouts in Rust (parallel combats, GIL-free)
    3. Compute GAE advantages
    4. PPO update (multiple epochs, minibatches)
    5. Log metrics, checkpoint
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import sts2_engine

from .paths import SOLVER_PKG
from .data_utils import (
    load_solver_json,
    build_monster_data_json,
    build_card_vocab,
    find_latest_checkpoint,
    setup_training_data,
    sample_combat_batches,
)
from .network import (
    ACTION_DIM,
    MAX_ACTIONS,
    MAX_HAND,
    STATE_DIM,
    BetaOneNetwork,
    export_onnx,
    save_checkpoint,
    load_checkpoint,
    ArchitectureMismatchError,
)
from .ppo import compute_gae, ppo_update


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(
    encounter_set_id: str,
    num_generations: int = 500,
    combats_per_gen: int = 256,
    lr: float = 3e-4,
    gamma: float = 0.99,
    lam: float = 0.95,
    temperature_start: float = 1.0,
    temperature_end: float = 0.5,
    entropy_coef: float = 0.03,
    clip_ratio: float = 0.2,
    value_coef: float = 0.5,
    max_grad_norm: float = 0.5,
    ppo_epochs: int = 4,
    ppo_batch_size: int = 256,
    output_dir: str = "betaone_checkpoints",
    value_head_layers: int = 1,
):
    os.makedirs(output_dir, exist_ok=True)
    onnx_dir = os.path.join(output_dir, "onnx")
    os.makedirs(onnx_dir, exist_ok=True)

    # Load game data (once)
    monster_json = build_monster_data_json()
    profiles_json = load_solver_json("enemy_profiles.json")

    # Card vocabulary for learned embeddings
    card_vocab, card_vocab_json = build_card_vocab(output_dir)
    num_cards = len(card_vocab)

    # Network + optimizer
    network = BetaOneNetwork(num_cards=num_cards, value_head_layers=value_head_layers)
    optimizer = torch.optim.Adam(network.parameters(), lr=lr)
    print(f"BetaOne network: {network.param_count():,} parameters ({num_cards} card vocab)")

    # Load the frozen encounter set
    td = setup_training_data(encounter_set_id=encounter_set_id)
    encounter_set = td["encounter_set"]

    best_win_rate = 0.0
    start_gen = 1

    history_path = os.path.join(output_dir, "betaone_history.jsonl")
    progress_path = os.path.join(output_dir, "betaone_progress.json")

    latest_ckpt = find_latest_checkpoint(output_dir)
    if latest_ckpt:
        ckpt = torch.load(latest_ckpt, weights_only=False)
        old_state = ckpt["model_state_dict"]
        # Dimension-aware warm-start: copy overlapping submatrix for shape
        # mismatches, identity-init new trunk layers, preserve everything else.
        try:
            network.load_state_dict(old_state)
            print("Loaded all weights (exact match)")
            arch_changed = False
        except RuntimeError:
            new_state = network.state_dict()
            exact, expanded, identity_init, skipped_keys = 0, 0, 0, []
            for key in new_state:
                if key not in old_state:
                    # New layer — check if it's a trunk Linear (identity-init)
                    if "trunk" in key and "weight" in key and new_state[key].dim() == 2:
                        n = min(new_state[key].shape)
                        nn.init.eye_(new_state[key])
                        # Add small noise so gradients aren't perfectly symmetric
                        new_state[key] += torch.randn_like(new_state[key]) * 0.01
                        identity_init += 1
                        print(f"  {key}: identity-init {list(new_state[key].shape)}")
                    else:
                        skipped_keys.append(key)
                        print(f"  {key}: zero-init {list(new_state[key].shape)}")
                    continue

                old_t = old_state[key]
                new_t = new_state[key]
                if old_t.shape == new_t.shape:
                    new_state[key] = old_t
                    exact += 1
                elif old_t.dim() == new_t.dim():
                    # Dimension expansion: copy overlapping submatrix
                    # Works for Linear weights (2D) and LayerNorm params (1D)
                    slices = tuple(slice(0, min(o, n)) for o, n in zip(old_t.shape, new_t.shape))
                    if all(o <= n for o, n in zip(old_t.shape, new_t.shape)):
                        # LayerNorm: init new dims to mean=0/var=1
                        if key.endswith(".weight") and new_t.dim() == 1:
                            if "weight" in key:
                                new_state[key] = torch.ones_like(new_t)
                            else:
                                new_state[key] = torch.zeros_like(new_t)
                        new_state[key][slices] = old_t[slices]
                        expanded += 1
                        print(f"  {key}: expanded {list(old_t.shape)} -> {list(new_t.shape)}")
                    else:
                        skipped_keys.append(key)
                        print(f"  {key}: shape mismatch {list(old_t.shape)} vs {list(new_t.shape)}, skipped")
                else:
                    skipped_keys.append(key)
                    print(f"  {key}: rank mismatch, skipped")
            network.load_state_dict(new_state)
            print(f"Warm-start: {exact} exact, {expanded} expanded, {identity_init} identity-init, {len(skipped_keys)} zero-init")
            arch_changed = True
        # Only restore optimizer if architecture didn't change
        if not arch_changed:
            try:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            except (ValueError, KeyError):
                print("Optimizer reset (state mismatch)")
        else:
            print("Optimizer reset (architecture changed)")
        start_gen = 1 if arch_changed else ckpt["gen"] + 1
        best_win_rate = ckpt.get("win_rate", 0.0)
        for f in [history_path, progress_path]:
            if os.path.exists(f):
                os.remove(f)
        print(f"Warm restart from gen {start_gen - 1}")
    else:
        print("Cold start — no checkpoint found")

    for gen in range(start_gen, num_generations + 1):
        t0 = time.time()

        # Temperature schedule (cosine decay)
        progress = gen / num_generations
        temperature = temperature_end + (temperature_start - temperature_end) * (
            1 + math.cos(math.pi * progress)
        ) / 2

        # Export current network
        onnx_path = export_onnx(network, onnx_dir)

        # Sample encounters grouped by HP (shared with selfplay_train.py)
        batches = sample_combat_batches(encounter_set, combats_per_gen, gen)

        # Collect rollouts — one call per HP level, merge results
        all_states, all_act_feat, all_act_masks = [], [], []
        all_hand_card_ids, all_action_card_ids = [], []
        all_draw_pile_ids, all_discard_pile_ids, all_exhaust_pile_ids = [], [], []
        all_chosen, all_lp, all_values, all_rewards, all_dones = [], [], [], [], []
        all_outcomes, all_hps = [], []
        seed_offset = gen * 100_000
        seed_idx = 0

        for b_enc, b_dks, b_rels, b_pots, b_hp, b_count in batches:
            if not b_enc:
                continue
            b_seeds = [seed_offset + seed_idx + i for i in range(b_count)]
            seed_idx += b_count
            rollout = sts2_engine.collect_betaone_rollouts(
                encounters_json=json.dumps(b_enc),
                decks_json=json.dumps(b_dks),
                player_hp=b_hp,
                player_max_hp=70,
                player_max_energy=3,
                relics_json=json.dumps(b_rels),
                potions_json=json.dumps(b_pots),
                monster_data_json=monster_json,
                enemy_profiles_json=profiles_json,
                onnx_path=onnx_path,
                temperature=temperature,
                seeds=b_seeds,
                gen_id=gen,
                card_vocab_json=card_vocab_json,
            )
            if rollout["total_steps"] == 0:
                continue
            all_states.extend(rollout["states"])
            all_act_feat.extend(rollout["action_features"])
            all_act_masks.extend(rollout["action_masks"])
            all_hand_card_ids.extend(rollout["hand_card_ids"])
            all_action_card_ids.extend(rollout["action_card_ids"])
            all_draw_pile_ids.extend(rollout["draw_pile_ids"])
            all_discard_pile_ids.extend(rollout["discard_pile_ids"])
            all_exhaust_pile_ids.extend(rollout["exhaust_pile_ids"])
            all_chosen.extend(rollout["chosen_indices"])
            all_lp.extend(rollout["log_probs"])
            all_values.extend(rollout["values"])
            all_rewards.extend(rollout["rewards"])
            all_dones.extend(rollout["dones"])
            all_outcomes.extend(rollout["outcomes"])
            all_hps.extend(rollout["final_hps"])

        T = len(all_chosen)
        if T == 0:
            print(f"Gen {gen}: no steps collected, skipping")
            continue

        rollout = {"outcomes": all_outcomes, "final_hps": all_hps, "total_steps": T}

        # Reshape flat arrays → tensors
        states = torch.tensor(all_states, dtype=torch.float32).reshape(T, STATE_DIM)
        act_feat = torch.tensor(
            all_act_feat, dtype=torch.float32
        ).reshape(T, MAX_ACTIONS, ACTION_DIM)
        act_masks = torch.tensor(all_act_masks).reshape(T, MAX_ACTIONS)
        hand_card_ids = torch.tensor(all_hand_card_ids, dtype=torch.long).reshape(T, MAX_HAND)
        action_card_ids = torch.tensor(all_action_card_ids, dtype=torch.long).reshape(T, MAX_ACTIONS)
        from .network import MAX_DRAW_PILE, MAX_DISCARD_PILE, MAX_EXHAUST_PILE
        draw_pile_ids = torch.tensor(all_draw_pile_ids, dtype=torch.long).reshape(T, MAX_DRAW_PILE)
        discard_pile_ids = torch.tensor(all_discard_pile_ids, dtype=torch.long).reshape(T, MAX_DISCARD_PILE)
        exhaust_pile_ids = torch.tensor(all_exhaust_pile_ids, dtype=torch.long).reshape(T, MAX_EXHAUST_PILE)
        chosen = torch.tensor(all_chosen, dtype=torch.long)
        old_lp = torch.tensor(all_lp, dtype=torch.float32)
        values = np.array(all_values, dtype=np.float32)
        rewards = np.array(all_rewards, dtype=np.float32)
        dones = np.array(all_dones, dtype=bool)

        # GAE
        advantages, returns = compute_gae(rewards, values, dones, gamma, lam)
        # Normalize advantages
        adv_std = advantages.std()
        if adv_std > 1e-8:
            advantages = (advantages - advantages.mean()) / adv_std

        # PPO update
        metrics = ppo_update(
            network,
            optimizer,
            states,
            act_feat,
            act_masks,
            hand_card_ids,
            action_card_ids,
            draw_pile_ids,
            discard_pile_ids,
            exhaust_pile_ids,
            chosen,
            old_lp,
            torch.tensor(advantages, dtype=torch.float32),
            torch.tensor(returns, dtype=torch.float32),
            clip_ratio=clip_ratio,
            value_coef=value_coef,
            entropy_coef=entropy_coef,
            max_grad_norm=max_grad_norm,
            epochs=ppo_epochs,
            batch_size=ppo_batch_size,
        )

        # Stats
        outcomes = rollout["outcomes"]
        n_episodes = len(outcomes)
        wins = sum(1 for o in outcomes if o == "win")
        win_rate = wins / max(n_episodes, 1)
        win_hps = [hp for hp, o in zip(rollout["final_hps"], outcomes) if o == "win"]
        avg_hp = np.mean(win_hps) if win_hps else 0.0
        avg_reward = float(rewards.mean())

        elapsed = time.time() - t0

        print(
            f"Gen {gen:4d} | "
            f"win {win_rate:5.1%} | "
            f"hp {avg_hp:4.1f} | "
            f"steps {T:5d} | "
            f"r {avg_reward:+.3f} | "
            f"pi {metrics['policy_loss']:.3f} | "
            f"v {metrics['value_loss']:.3f} | "
            f"H {metrics['entropy']:.2f} | "
            f"t={temperature:.2f} | "
            f"{elapsed:.1f}s"
        )

        # Log to JSONL + progress snapshot
        record = {
            "gen": gen,
            "win_rate": round(win_rate, 4),
            "avg_hp": round(float(avg_hp), 1),
            "avg_reward": round(avg_reward, 4),
            "steps": T,
            "episodes": n_episodes,
            "encounter_set": encounter_set_id,
            "policy_loss": round(metrics["policy_loss"], 5),
            "value_loss": round(metrics["value_loss"], 5),
            "entropy": round(metrics["entropy"], 4),
            "temperature": round(temperature, 3),
            "gen_time": round(elapsed, 2),
            "timestamp": time.time(),
        }
        with open(history_path, "a") as f:
            f.write(json.dumps(record) + "\n")
        with open(progress_path, "w") as f:
            record["num_generations"] = num_generations
            record["best_win_rate"] = round(best_win_rate, 4)
            json.dump(record, f, indent=2)

        # Save checkpoints: always save "latest", keep milestones
        best_win_rate = max(best_win_rate, win_rate)
        ckpt_data = {
            "gen": gen,
            "arch_meta": network.arch_meta(),
            "model_state_dict": network.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "win_rate": win_rate,
        }
        # Always overwrite latest (resume point)
        torch.save(ckpt_data, os.path.join(output_dir, "betaone_latest.pt"))
        # Keep milestone checkpoints
        if gen % 50 == 0 or win_rate >= best_win_rate:
            torch.save(ckpt_data, os.path.join(output_dir, f"betaone_gen{gen}.pt"))

    print(f"\nTraining complete. Best win rate: {best_win_rate:.1%}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="BetaOne PPO training")
    parser.add_argument("--encounter-set", required=True,
                        help="Encounter set id (e.g. lean-decks-v1)")
    parser.add_argument("--generations", type=int, default=500)
    parser.add_argument("--combats", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--entropy-coef", type=float, default=0.03)
    parser.add_argument("--output-dir", default="betaone_checkpoints")
    args = parser.parse_args()

    train(
        encounter_set_id=args.encounter_set,
        num_generations=args.generations,
        combats_per_gen=args.combats,
        lr=args.lr,
        entropy_coef=args.entropy_coef,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
