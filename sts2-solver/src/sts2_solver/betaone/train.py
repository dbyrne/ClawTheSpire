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

from .curriculum import CombatCurriculum, TIER_CONFIGS
from .deck_gen import build_random_deck_json, _make_starter
from .network import (
    ACTION_DIM,
    MAX_ACTIONS,
    STATE_DIM,
    BetaOneNetwork,
    export_onnx,
)
from .ppo import compute_gae, ppo_update

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

_DATA_DIR = Path(__file__).resolve().parents[4] / "STS2-Agent" / "mcp_server" / "data" / "eng"
_SOLVER_DIR = Path(__file__).resolve().parents[1]  # sts2_solver/


def _load_json(filename: str) -> str:
    """Load a JSON file and return as string for Rust."""
    path = _DATA_DIR / filename
    if not path.exists():
        return "[]"
    return path.read_text(encoding="utf-8")


def _load_solver_json(filename: str) -> str:
    """Load a JSON from the solver data directory."""
    path = _SOLVER_DIR / filename
    if not path.exists():
        return "{}"
    return path.read_text(encoding="utf-8")


def _build_monster_data_json() -> str:
    """Build monster data dict keyed by ID."""
    monsters_raw = json.loads(_load_json("monsters.json"))
    monsters = {}
    for m in monsters_raw:
        mid = m.get("id", "")
        if mid:
            monsters[mid] = {
                "name": m.get("name", mid),
                "min_hp": m.get("min_hp") or 20,
                "max_hp": m.get("max_hp") or m.get("min_hp") or 20,
            }
    return json.dumps(monsters)



def _find_latest_checkpoint(output_dir: str) -> str | None:
    """Find the best resume checkpoint: latest.pt first, else highest gen."""
    latest = os.path.join(output_dir, "betaone_latest.pt")
    if os.path.exists(latest):
        return latest
    import glob
    pattern = os.path.join(output_dir, "betaone_gen*.pt")
    ckpts = glob.glob(pattern)
    if not ckpts:
        return None
    def gen_num(p: str) -> int:
        base = os.path.basename(p)
        return int(base.replace("betaone_gen", "").replace(".pt", ""))
    return max(ckpts, key=gen_num)


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
    temperature_end: float = 0.3,
    entropy_coef: float = 0.01,
    ppo_epochs: int = 4,
    ppo_batch_size: int = 256,
    output_dir: str = "betaone_checkpoints",
    encounter_pool_path: str | None = None,
):
    os.makedirs(output_dir, exist_ok=True)
    onnx_dir = os.path.join(output_dir, "onnx")
    os.makedirs(onnx_dir, exist_ok=True)

    # Load game data (once)
    monster_json = _build_monster_data_json()
    profiles_json = _load_solver_json("enemy_profiles.json")

    enc_pool_path = encounter_pool_path or str(_SOLVER_DIR / "encounter_pool.json")

    # Network + optimizer
    network = BetaOneNetwork()
    optimizer = torch.optim.Adam(network.parameters(), lr=lr)
    print(f"BetaOne network: {network.param_count():,} parameters")

    # Curriculum
    curriculum = CombatCurriculum(encounter_pool_path=enc_pool_path)

    best_win_rate = 0.0
    cumulative_wins = 0
    cumulative_games = 0
    start_gen = 1

    history_path = os.path.join(output_dir, "betaone_history.jsonl")
    progress_path = os.path.join(output_dir, "betaone_progress.json")

    latest_ckpt = _find_latest_checkpoint(output_dir)
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
                        if "LayerNorm" in key or key.endswith(".weight") and new_t.dim() == 1:
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
        start_gen = ckpt["gen"] + 1
        best_win_rate = ckpt.get("win_rate", 0.0)
        for f in [history_path, progress_path]:
            if os.path.exists(f):
                os.remove(f)
        print(f"Warm restart from gen {start_gen - 1} — validating tiers")

        # Pre-flight: greedy eval through tiers without training
        onnx_path = export_onnx(network, onnx_dir)
        while curriculum.tier < curriculum.max_tier:
            cfg = curriculum.config
            n_eval = 128
            eval_enc = curriculum.sample_encounters(n_eval)
            eval_decks = [json.loads(curriculum.sample_deck_json()) for _ in range(n_eval)]
            eval_rollout = sts2_engine.collect_betaone_rollouts(
                encounters_json=json.dumps(eval_enc),
                decks_json=json.dumps(eval_decks),
                player_hp=cfg.player_hp, player_max_hp=70, player_max_energy=3,
                relics=[], potions_json="[]",
                monster_data_json=monster_json,
                enemy_profiles_json=profiles_json,
                onnx_path=onnx_path,
                temperature=0.01,
                seeds=list(range(n_eval)),
                gen_id=start_gen,
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

    regressed_tiers: set[int] = set()
    regressed_detail: dict[int, float] = {}

    for gen in range(start_gen, num_generations + 1):
        t0 = time.time()

        # Temperature schedule (cosine decay)
        progress = gen / num_generations
        temperature = temperature_end + (temperature_start - temperature_end) * (
            1 + math.cos(math.pi * progress)
        ) / 2

        # Export current network
        onnx_path = export_onnx(network, onnx_dir)

        # Regression check every 50 gens: eval all previous tiers
        if gen % 50 == 0 and curriculum.tier > 0:
            regressed_tiers.clear()
            regressed_detail: dict[int, float] = {}
            for check_tier in range(curriculum.tier):
                check_cfg = TIER_CONFIGS[check_tier]
                n_check = 50
                check_enc = []
                check_decks_list = []
                for _ in range(n_check):
                    if check_cfg.custom_encounters:
                        check_enc.append(random.choice(check_cfg.custom_encounters))
                    elif check_cfg.encounter_level >= 0:
                        check_enc.append(random.choice(
                            curriculum.encounter_pools[check_cfg.encounter_level]))
                    else:
                        check_enc.append(random.choice(curriculum.encounter_pools[0]))
                    if check_cfg.custom_deck is not None:
                        check_decks_list.append(check_cfg.custom_deck)
                    elif check_cfg.deck_mode == "starter":
                        check_decks_list.append(json.loads(json.dumps(_make_starter())))
                    elif check_cfg.deck_mode == "review_all":
                        check_decks_list.append(json.loads(json.dumps(_make_starter())))
                    else:
                        check_decks_list.append(json.loads(build_random_deck_json(
                            rng=curriculum.deck_rng,
                            min_size=check_cfg.deck_min_size,
                            max_size=check_cfg.deck_max_size,
                            min_removals=check_cfg.deck_min_removals,
                            max_removals=check_cfg.deck_max_removals,
                            archetypes=check_cfg.deck_archetypes,
                        )))
                check_rollout = sts2_engine.collect_betaone_rollouts(
                    encounters_json=json.dumps(check_enc),
                    decks_json=json.dumps(check_decks_list),
                    player_hp=check_cfg.player_hp, player_max_hp=70, player_max_energy=3,
                    relics=[], potions_json="[]",
                    monster_data_json=monster_json,
                    enemy_profiles_json=profiles_json,
                    onnx_path=onnx_path,
                    temperature=0.01,
                    seeds=[gen * 100_000 + 90_000 + i for i in range(n_check)],
                    gen_id=gen,
                )
                check_wr = sum(1 for o in check_rollout["outcomes"] if o == "win") / max(len(check_rollout["outcomes"]), 1)
                if check_wr < check_cfg.promote_threshold * 0.9:  # 90% of original threshold
                    regressed_tiers.add(check_tier)
                    regressed_detail[check_tier] = check_wr
            if regressed_tiers:
                parts = [f"T{t} {regressed_detail[t]:.0%}/{TIER_CONFIGS[t].promote_threshold:.0%}" for t in sorted(regressed_tiers)]
                print(f"  Regression detected: {', '.join(parts)} — adding review combats")

        # Build combat batches: current tier + regressed tier review (each with correct HP)
        cfg = curriculum.config
        batches: list[tuple[list, list, int, int]] = []  # (encounters, decks, hp, count)

        # Regressed tier review batches
        n_review = 0
        for review_tier_idx in regressed_tiers:
            review_cfg = TIER_CONFIGS[review_tier_idx]
            review_per_tier = combats_per_gen // 8
            enc = []
            dks = []
            for _ in range(review_per_tier):
                if review_cfg.custom_encounters:
                    enc.append(random.choice(review_cfg.custom_encounters))
                else:
                    pool = curriculum.encounter_pools[review_cfg.encounter_level]
                    enc.append(random.choice(pool))
                if review_cfg.custom_deck is not None:
                    dks.append(review_cfg.custom_deck)
                elif review_cfg.deck_mode == "starter":
                    dks.append(json.loads(json.dumps(_make_starter())))
                else:
                    dks.append(json.loads(build_random_deck_json(
                        rng=curriculum.deck_rng,
                        min_size=review_cfg.deck_min_size,
                        max_size=review_cfg.deck_max_size,
                        min_removals=review_cfg.deck_min_removals,
                        max_removals=review_cfg.deck_max_removals,
                        archetypes=review_cfg.deck_archetypes,
                    )))
            batches.append((enc, dks, review_cfg.player_hp, review_per_tier))
            n_review += review_per_tier

        # Current tier batch
        n_current = combats_per_gen - n_review
        cur_enc = curriculum.sample_encounters(n_current)
        cur_dks = [json.loads(curriculum.sample_deck_json()) for _ in range(n_current)]
        batches.append((cur_enc, cur_dks, cfg.player_hp, n_current))

        # Collect rollouts — one call per HP level, merge results
        all_states, all_act_feat, all_act_masks = [], [], []
        all_chosen, all_lp, all_values, all_rewards, all_dones = [], [], [], [], []
        all_outcomes, all_hps = [], []
        seed_offset = gen * 100_000
        seed_idx = 0

        for b_enc, b_dks, b_hp, b_count in batches:
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
                relics=[],
                potions_json="[]",
                monster_data_json=monster_json,
                enemy_profiles_json=profiles_json,
                onnx_path=onnx_path,
                temperature=temperature,
                seeds=b_seeds,
                gen_id=gen,
            )
            if rollout["total_steps"] == 0:
                continue
            all_states.extend(rollout["states"])
            all_act_feat.extend(rollout["action_features"])
            all_act_masks.extend(rollout["action_masks"])
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
            chosen,
            old_lp,
            torch.tensor(advantages, dtype=torch.float32),
            torch.tensor(returns, dtype=torch.float32),
            entropy_coef=entropy_coef,
            epochs=ppo_epochs,
            batch_size=ppo_batch_size,
        )

        # Stats — use current-tier combats only (exclude review) for promotion
        outcomes = rollout["outcomes"]
        n_episodes = len(outcomes)
        n_review = n_episodes - n_current  # review combats are first in the list
        tier_outcomes = outcomes[n_review:]  # current-tier combats only
        tier_wins = sum(1 for o in tier_outcomes if o == "win")
        tier_wr = tier_wins / max(len(tier_outcomes), 1)

        all_wins = sum(1 for o in outcomes if o == "win")
        win_rate = all_wins / max(n_episodes, 1)  # blended (for logging)
        win_hps = [hp for hp, o in zip(rollout["final_hps"], outcomes) if o == "win"]
        avg_hp = np.mean(win_hps) if win_hps else 0.0
        avg_reward = float(rewards.mean())
        cumulative_wins += all_wins
        cumulative_games += n_episodes

        # Curriculum update — use tier-only win rate (no review inflation)
        tier_before = curriculum.tier
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
            "regressed": sorted(regressed_tiers) if regressed_tiers else None,
            "regressed_detail": {str(k): round(v, 3) for k, v in regressed_detail.items()} if regressed_tiers else None,
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
        }
        with open(history_path, "a") as f:
            f.write(json.dumps(record) + "\n")
        with open(progress_path, "w") as f:
            record["num_generations"] = num_generations
            record["cumulative_wins"] = cumulative_wins
            record["cumulative_games"] = cumulative_games
            record["best_win_rate"] = round(best_win_rate, 4)
            json.dump(record, f, indent=2)

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
    parser.add_argument("--entropy-coef", type=float, default=0.005)
    parser.add_argument("--output-dir", default="betaone_checkpoints")
    args = parser.parse_args()

    train(
        num_generations=args.generations,
        combats_per_gen=args.combats,
        lr=args.lr,
        entropy_coef=args.entropy_coef,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
