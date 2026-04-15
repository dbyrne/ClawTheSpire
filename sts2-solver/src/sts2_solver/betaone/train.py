"""BetaOne training orchestrator.

Usage:
    python -m sts2_solver.betaone.train [--generations N] [--combats N]

Loop:
    1. Export network → ONNX
    2. Collect rollouts in Rust (parallel combats, GIL-free)
    3. Compute GAE advantages
    4. PPO update (multiple epochs, minibatches)
    5. Log metrics, adjust curriculum
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
    encounter_pool_path: str | None = None,
    skip_to_final: bool = False,
    lock_tier: int | None = None,
    recorded_encounters: bool = False,
    mixed: bool = False,
    recorded_frac: float = 0.5,
    training_set_id: str | None = None,
):
    os.makedirs(output_dir, exist_ok=True)
    onnx_dir = os.path.join(output_dir, "onnx")
    os.makedirs(onnx_dir, exist_ok=True)

    # Load game data (once)
    monster_json = build_monster_data_json()
    profiles_json = load_solver_json("enemy_profiles.json")

    enc_pool_path = encounter_pool_path or str(SOLVER_PKG / "encounter_pool.json")

    # Card vocabulary for learned embeddings
    card_vocab, card_vocab_json = build_card_vocab(output_dir)
    num_cards = len(card_vocab)

    # Network + optimizer
    network = BetaOneNetwork(num_cards=num_cards)
    optimizer = torch.optim.Adam(network.parameters(), lr=lr)
    print(f"BetaOne network: {network.param_count():,} parameters ({num_cards} card vocab)")

    # Set up training data (shared with selfplay_train.py)
    td = setup_training_data(
        output_dir=output_dir,
        training_set_id=training_set_id,
        mixed=mixed,
        recorded_encounters=recorded_encounters,
        recorded_frac=recorded_frac,
        skip_to_final=skip_to_final,
    )
    curriculum = td["curriculum"]
    recorded_encounters = td["recorded_encounters"]
    mixed = td["mixed"]
    recorded_frac = td["recorded_frac"]

    best_win_rate = 0.0
    cumulative_wins = 0
    cumulative_games = 0
    start_gen = 1

    # Per-tier cumulative stats — persisted in a separate file that survives restarts
    cumulative_path = os.path.join(output_dir, "betaone_cumulative.json")
    tier_cumulative: dict[str, list[int]] = {}  # "tier_idx" -> [wins, games]
    if os.path.exists(cumulative_path):
        try:
            with open(cumulative_path) as f:
                tier_cumulative = json.load(f)
        except (json.JSONDecodeError, ValueError):
            pass

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
        if lock_tier is not None:
            curriculum.tier = lock_tier
            curriculum.consecutive_good = 0
            curriculum.gens_at_tier = 0
            print(f"Warm restart from gen {start_gen - 1} — locked to T{lock_tier}: {curriculum.config.name}")
        elif skip_to_final:
            curriculum.tier = curriculum.max_tier
            curriculum.consecutive_good = 0
            curriculum.gens_at_tier = 0
            print(f"Warm restart from gen {start_gen - 1} — skipping to T{curriculum.tier}: {curriculum.config.name}")
        else:
            print(f"Warm restart from gen {start_gen - 1} — validating tiers")

        # Pre-flight: greedy eval through tiers without training
        onnx_path = export_onnx(network, onnx_dir)
        while lock_tier is None and not skip_to_final and curriculum.tier < curriculum.max_tier:
            cfg = curriculum.config
            n_eval = 128
            eval_enc = curriculum.sample_encounters(n_eval)
            eval_decks = [json.loads(curriculum.sample_deck_json()) for _ in range(n_eval)]
            eval_rollout = sts2_engine.collect_betaone_rollouts(
                encounters_json=json.dumps(eval_enc),
                decks_json=json.dumps(eval_decks),
                player_hp=cfg.player_hp, player_max_hp=70, player_max_energy=3,
                relics_json="[]", potions_json="[]",
                monster_data_json=monster_json,
                enemy_profiles_json=profiles_json,
                onnx_path=onnx_path,
                temperature=0.01,
                seeds=list(range(n_eval)),
                gen_id=start_gen,
                card_vocab_json=card_vocab_json,
            )
            eval_outcomes = eval_rollout["outcomes"]
            eval_wr = sum(1 for o in eval_outcomes if o == "win") / max(len(eval_outcomes), 1)
            passed = eval_wr >= cfg.promote_threshold

            print(f"  T{curriculum.tier} {cfg.name:25s} need {cfg.promote_threshold:.0%}  wr={eval_wr:.1%}  {'PASS' if passed else 'FAIL'}")

            # Log the eval-only result
            record = {
                "gen": start_gen - 1,
                "win_rate": round(eval_wr, 4),
                "tier_wr": round(eval_wr, 4),
                "tier": curriculum.tier,
                "tier_name": cfg.name,
                "tier_change": "promoted" if passed else "hold",
                "eval_only": True,
                "policy_loss": 0, "value_loss": 0, "entropy": 0,
                "avg_reward": 0, "avg_hp": 0, "steps": 0, "episodes": len(eval_outcomes),
                "temperature": 0, "gen_time": 0, "timestamp": time.time(),
                "gens_at_tier": 0,
            }
            with open(history_path, "a") as f:
                f.write(json.dumps(record) + "\n")

            if passed:
                curriculum.tier += 1
                curriculum.consecutive_good = 0
                curriculum.gens_at_tier = 0
            else:
                break

        print(f"  Starting training at T{curriculum.tier}: {curriculum.config.name}")
    else:
        print("Cold start — no checkpoint found")

    # No runtime calibration when using training sets.
    # Legacy calibration preserved for backward compatibility without training sets.
    if not td["ts_data"]:
        if recorded_encounters:
            from .calibrate import calibrate_all
            recorded_path = str(Path(output_dir) / "recorded_encounters.jsonl")
            onnx_path = export_onnx(network, onnx_dir)
            print("Running initial HP calibration...")
            curriculum.recorded_encounters, _ = calibrate_all(
                curriculum.recorded_encounters, monster_json, profiles_json,
                card_vocab_json, onnx_path,
                encounters_path=recorded_path,
                num_sims=50, combats=32,
            )
        if mixed:
            from .packages import calibrate_packages
            if not recorded_encounters:
                onnx_path = export_onnx(network, onnx_dir)
            print("Calibrating archetype packages...")
            calibrate_packages(monster_json, profiles_json, card_vocab_json, onnx_path)

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
        batches = sample_combat_batches(
            curriculum, combats_per_gen, mixed, recorded_encounters,
            recorded_frac, gen,
        )

        # Collect rollouts — one call per HP level, merge results
        all_states, all_act_feat, all_act_masks = [], [], []
        all_hand_card_ids, all_action_card_ids = [], []
        all_chosen, all_lp, all_values, all_rewards, all_dones = [], [], [], [], []
        all_outcomes, all_hps = [], []
        seed_offset = gen * 100_000
        seed_idx = 0

        for b_enc, b_dks, b_rels, b_hp, b_count in batches:
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
                potions_json="[]",
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
        tier_wins = sum(1 for o in outcomes if o == "win")
        tier_wr = tier_wins / max(n_episodes, 1)

        all_wins = sum(1 for o in outcomes if o == "win")
        win_rate = all_wins / max(n_episodes, 1)  # blended (for logging)
        win_hps = [hp for hp, o in zip(rollout["final_hps"], outcomes) if o == "win"]
        avg_hp = np.mean(win_hps) if win_hps else 0.0
        avg_reward = float(rewards.mean())
        cumulative_wins += all_wins
        cumulative_games += n_episodes

        # Curriculum update — use tier-only win rate (no review inflation)
        tier_before = curriculum.tier
        # Per-tier cumulative (persisted across runs via checkpoint)
        tk = str(tier_before)
        prev = tier_cumulative.get(tk, [0, 0])
        tier_cumulative[tk] = [prev[0] + tier_wins, prev[1] + len(tier_outcomes)]
        if lock_tier is not None:
            tier_change = "hold"
            curriculum.gens_at_tier += 1
        else:
            tier_change = curriculum.update(tier_wr)

        elapsed = time.time() - t0

        print(
            f"Gen {gen:4d} | "
            f"win {tier_wr:5.1%} | "
            f"hp {avg_hp:4.1f} | "
            f"steps {T:5d} | "
            f"r {avg_reward:+.3f} | "
            f"pi {metrics['policy_loss']:.3f} | "
            f"v {metrics['value_loss']:.3f} | "
            f"H {metrics['entropy']:.2f} | "
            f"T{tier_before}{'UP' if tier_change == 'promoted' else 'DN' if tier_change == 'demoted' else '  '} | "
            f"t={temperature:.2f} | "
            f"{elapsed:.1f}s"
        )

        # Log to JSONL + progress snapshot
        record = {
            "gen": gen,
            "win_rate": round(tier_wr, 4),
            "tier_wr": round(tier_wr, 4),
            "regressed": None,
            "regressed_detail": None,
            "cumulative_win_rate": round(cumulative_wins / max(cumulative_games, 1), 4),
            "avg_hp": round(float(avg_hp), 1),
            "avg_reward": round(avg_reward, 4),
            "steps": T,
            "episodes": n_episodes,
            "tier": tier_before,
            "tier_name": TIER_CONFIGS[tier_before].name,
            "gens_at_tier": curriculum.gens_at_tier,
            "tier_change": tier_change,
            "policy_loss": round(metrics["policy_loss"], 5),
            "value_loss": round(metrics["value_loss"], 5),
            "entropy": round(metrics["entropy"], 4),
            "temperature": round(temperature, 3),
            "gen_time": round(elapsed, 2),
            "timestamp": time.time(),
            "recorded_encounters": len(curriculum.recorded_encounters) if recorded_encounters else None,
            "training_set": training_set_id,
        }
        with open(history_path, "a") as f:
            f.write(json.dumps(record) + "\n")
        with open(progress_path, "w") as f:
            record["num_generations"] = num_generations
            record["cumulative_wins"] = cumulative_wins
            record["cumulative_games"] = cumulative_games
            record["best_win_rate"] = round(best_win_rate, 4)
            record["tier_cumulative"] = tier_cumulative
            json.dump(record, f, indent=2)
        with open(cumulative_path, "w") as f:
            json.dump(tier_cumulative, f)

        # Save checkpoints: always save "latest", keep milestones
        best_win_rate = max(best_win_rate, win_rate)
        ckpt_data = {
            "gen": gen,
            "model_state_dict": network.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "win_rate": win_rate,
            "tier": curriculum.tier,
            "tier_name": curriculum.config.name,
            "gens_at_tier": curriculum.gens_at_tier,
            "tier_cumulative": tier_cumulative,
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
    parser.add_argument("--generations", type=int, default=500)
    parser.add_argument("--combats", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--entropy-coef", type=float, default=0.03)
    parser.add_argument("--output-dir", default="betaone_checkpoints")
    parser.add_argument("--final-exam", action="store_true",
                        help="Skip tier progression, start at final exam")
    parser.add_argument("--tier", type=int, default=None,
                        help="Lock to a specific tier (no promotion)")
    parser.add_argument("--recorded-encounters", action="store_true",
                        help="Train exclusively on recorded death encounters from live games")
    parser.add_argument("--mixed", action="store_true",
                        help="Train on mix of recorded encounters and final exam")
    parser.add_argument("--recorded-frac", type=float, default=0.5,
                        help="Fraction of combats from recorded encounters in mixed mode")
    args = parser.parse_args()

    train(
        num_generations=args.generations,
        combats_per_gen=args.combats,
        lr=args.lr,
        entropy_coef=args.entropy_coef,
        output_dir=args.output_dir,
        skip_to_final=args.final_exam,
        lock_tier=args.tier,
        recorded_encounters=args.recorded_encounters,
        mixed=args.mixed,
        recorded_frac=args.recorded_frac,
    )


if __name__ == "__main__":
    main()
