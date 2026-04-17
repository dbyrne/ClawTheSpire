"""BetaOne AlphaZero-style self-play training with replay buffer.

Replaces PPO with MCTS self-play: network learns from MCTS visit
distributions (policy) and game outcomes (value).  A replay buffer
accumulates samples across generations so the value head sees the
same states with different outcomes, enabling convergence to true
win probabilities.

Usage:
    python -m sts2_solver.betaone.selfplay_train [--generations N] [--combats N]
    python -m sts2_solver.betaone.selfplay_train --recorded-encounters --sims 150
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random as stdlib_random
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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


# ---------------------------------------------------------------------------
# Replay buffer
# ---------------------------------------------------------------------------

class ReplayBuffer:
    """Fixed-capacity ring buffer of self-play samples.

    Each entry stores tensors for one decision point: state encoding,
    action features/mask/ids, MCTS policy target, and game outcome value.
    Oldest generations are evicted when the buffer is full.
    """

    def __init__(self, max_steps: int = 200_000):
        self.max_steps = max_steps
        self.states: list[np.ndarray] = []
        self.act_feat: list[np.ndarray] = []
        self.act_masks: list[np.ndarray] = []
        self.hand_ids: list[np.ndarray] = []
        self.action_ids: list[np.ndarray] = []
        self.policies: list[np.ndarray] = []
        self.values: list[np.ndarray] = []
        # Track generation boundaries for FIFO eviction
        self._gen_sizes: deque[int] = deque()

    def __len__(self) -> int:
        return len(self.states)

    def add_generation(
        self,
        states: np.ndarray,
        act_feat: np.ndarray,
        act_masks: np.ndarray,
        hand_ids: np.ndarray,
        action_ids: np.ndarray,
        policies: np.ndarray,
        values: np.ndarray,
    ) -> None:
        """Add one generation's samples. Evicts oldest gens if over capacity."""
        n = len(states)
        self._gen_sizes.append(n)
        self.states.extend(states)
        self.act_feat.extend(act_feat)
        self.act_masks.extend(act_masks)
        self.hand_ids.extend(hand_ids)
        self.action_ids.extend(action_ids)
        self.policies.extend(policies)
        self.values.extend(values)

        # Evict oldest generations until under capacity
        while len(self.states) > self.max_steps and len(self._gen_sizes) > 1:
            drop = self._gen_sizes.popleft()
            del self.states[:drop]
            del self.act_feat[:drop]
            del self.act_masks[:drop]
            del self.hand_ids[:drop]
            del self.action_ids[:drop]
            del self.policies[:drop]
            del self.values[:drop]

    def sample_tensors(self, batch_size: int) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor,
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
    ]:
        """Sample a random batch from the buffer, returns tensors."""
        n = len(self.states)
        indices = np.random.choice(n, size=min(batch_size, n), replace=False)
        return (
            torch.tensor(np.array([self.states[i] for i in indices]), dtype=torch.float32),
            torch.tensor(np.array([self.act_feat[i] for i in indices]), dtype=torch.float32),
            torch.tensor(np.array([self.act_masks[i] for i in indices])),
            torch.tensor(np.array([self.hand_ids[i] for i in indices]), dtype=torch.long),
            torch.tensor(np.array([self.action_ids[i] for i in indices]), dtype=torch.long),
            torch.tensor(np.array([self.policies[i] for i in indices]), dtype=torch.float32),
            torch.tensor(np.array([self.values[i] for i in indices]), dtype=torch.float32),
        )


# ---------------------------------------------------------------------------
# Q-target mixing
# ---------------------------------------------------------------------------

def compute_mixed_policy_target(
    visits: np.ndarray,
    q_values: np.ndarray,
    mask: np.ndarray,
    mix: float,
    temp: float,
) -> np.ndarray:
    """Mix visit distribution with softmax(Q/temp) to make the policy target
    reflect Q values more, not just visit counts.

    target = (1 - mix) * visits_norm + mix * softmax(Q/temp over visited only)

    Args:
        visits: (T, MAX_ACTIONS) raw visit counts (zero-padded).
        q_values: (T, MAX_ACTIONS) Q values per action (zero-padded).
        mask: (T, MAX_ACTIONS) bool, True for valid actions.
        mix: weight on Q-softmax part (0 = pure visits, 1 = pure Q).
        temp: softmax temperature for Q.

    Returns:
        (T, MAX_ACTIONS) float32 policy target.

    Important: the softmax only includes *visited* children. Unvisited actions
    have Q=0 (init) which would dominate softmax if included; we mask them out
    so unvisited actions get only the (zero) visit-norm contribution.
    """
    visits_f = visits.astype(np.float32)
    visit_sums = visits_f.sum(axis=1, keepdims=True)
    visit_sums = np.where(visit_sums > 0, visit_sums, 1.0)
    visits_norm = visits_f / visit_sums

    if mix <= 0:
        return visits_norm.astype(np.float32)

    visited = (visits > 0) & mask
    # Build logits: -inf for un-included actions (excludes from softmax)
    safe_logits = np.where(visited, q_values / max(temp, 1e-6), -np.inf)
    # Per-row max for numerical stability; rows with zero visited fall back to 0
    max_logits = np.max(safe_logits, axis=1, keepdims=True)
    max_logits = np.where(np.isfinite(max_logits), max_logits, 0.0)
    exp_logits = np.exp(safe_logits - max_logits)
    exp_sums = exp_logits.sum(axis=1, keepdims=True)
    safe_sums = np.where(exp_sums > 0, exp_sums, 1.0)
    softmax_q = np.where(exp_sums > 0, exp_logits / safe_sums, 0.0)

    return ((1.0 - mix) * visits_norm + mix * softmax_q).astype(np.float32)


# ---------------------------------------------------------------------------
# Training step
# ---------------------------------------------------------------------------

def train_batch(
    network: BetaOneNetwork,
    optimizer: torch.optim.Optimizer,
    states: torch.Tensor,
    action_features: torch.Tensor,
    action_masks: torch.Tensor,
    hand_card_ids: torch.Tensor,
    action_card_ids: torch.Tensor,
    target_policies: torch.Tensor,
    target_values: torch.Tensor,
    value_coef: float = 1.0,
) -> dict[str, float]:
    """Single training step: cross-entropy policy + MSE value."""
    logits, values = network(states, action_features, action_masks,
                             hand_card_ids, action_card_ids)

    # Policy loss: cross-entropy against MCTS visit distribution
    log_probs = F.log_softmax(logits, dim=1)
    policy_loss = -(target_policies * log_probs).nan_to_num(0.0).sum(dim=1).mean()

    # Value loss: MSE against game outcome
    value_loss = F.mse_loss(values.squeeze(-1), target_values)

    loss = policy_loss + value_coef * value_loss

    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(network.parameters(), 1.0)
    optimizer.step()

    return {
        "policy_loss": policy_loss.item(),
        "value_loss": value_loss.item(),
    }


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(
    encounter_set_id: str,
    num_generations: int = 2000,
    combats_per_gen: int = 256,
    num_sims: int = 150,
    lr: float = 3e-4,
    value_coef: float = 1.0,
    train_epochs: int = 4,
    batch_size: int = 512,
    temperature: float = 1.0,
    output_dir: str = "betaone_checkpoints",
    replay_capacity: int = 200_000,
    cold_start: bool = False,
    turn_boundary_eval: bool = False,
    dense_value_targets: bool = False,
    gamma: float = 0.99,
    c_puct: float = 2.5,
    pomcp: bool = False,
    mcts_bootstrap: bool = False,
    noise_frac: float = 0.25,
    pw_k: float = 1.0,
    q_target_mix: float = 0.0,
    q_target_temp: float = 0.5,
    eval_every: int = 0,
    value_head_layers: int = 1,
):
    os.makedirs(output_dir, exist_ok=True)
    onnx_dir = os.path.join(output_dir, "onnx")
    os.makedirs(onnx_dir, exist_ok=True)

    # mcts_bootstrap supersedes dense_value_targets — the Rust-side reward
    # computation is dead work when bootstrap is on. Strip the flag here so
    # the Rust self-play loop skips the per-turn reward plumbing entirely.
    # Configs that set both don't fail; they just get a one-line notice.
    if mcts_bootstrap and dense_value_targets:
        print("Note: mcts_bootstrap=True overrides dense_value_targets — "
              "disabling dense reward compute.")
        dense_value_targets = False

    # Load game data
    monster_json = build_monster_data_json()
    profiles_json = load_solver_json("enemy_profiles.json")

    # Card vocabulary
    card_vocab, card_vocab_json = build_card_vocab(output_dir)
    num_cards = len(card_vocab)

    # Network + optimizer
    network = BetaOneNetwork(num_cards=num_cards, value_head_layers=value_head_layers)
    optimizer = torch.optim.Adam(network.parameters(), lr=lr)
    print(f"BetaOne self-play: {network.param_count():,} params, {num_cards} card vocab")

    # Load the frozen encounter set
    td = setup_training_data(encounter_set_id=encounter_set_id)
    encounter_set = td["encounter_set"]
    encounter_set_name = td["encounter_set_name"]

    best_win_rate = 0.0
    start_gen = 1

    history_path = os.path.join(output_dir, "betaone_history.jsonl")
    progress_path = os.path.join(output_dir, "betaone_progress.json")

    # Resume from checkpoint (unless cold start)
    if not cold_start:
        latest_ckpt = find_latest_checkpoint(output_dir)
        if latest_ckpt:
            ckpt = torch.load(latest_ckpt, weights_only=False)
            try:
                network.load_state_dict(ckpt["model_state_dict"])
                print(f"Loaded checkpoint: gen {ckpt.get('gen', '?')}")
                try:
                    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                except (ValueError, KeyError):
                    print("Optimizer reset")
                start_gen = ckpt.get("gen", 0) + 1
                best_win_rate = ckpt.get("win_rate", 0.0)
            except RuntimeError:
                # Dimension-aware warm-start
                old_state = ckpt["model_state_dict"]
                new_state = network.state_dict()

                # Reset value_head entirely when layer count changes.
                # The slice-copy below would otherwise wedge the old readout
                # matrix [1, H] into a new hidden-layer position [H', H] at
                # slice [:1, :H], turning a read-out into a semantically
                # broken first row of a hidden→hidden transform. Random init
                # is strictly better than that — layers that don't exist in
                # the old head stay at init, and the readout layer starts
                # fresh rather than from a displaced weight.
                old_vh_keys = {k for k in old_state if k.startswith("value_head.")}
                new_vh_keys = {k for k in new_state if k.startswith("value_head.")}
                reset_value_head = old_vh_keys != new_vh_keys

                loaded, skipped, reset = 0, 0, 0
                for key in new_state:
                    if reset_value_head and key.startswith("value_head."):
                        reset += 1
                        continue
                    if key not in old_state:
                        skipped += 1
                        continue
                    old_t = old_state[key]
                    new_t = new_state[key]
                    if old_t.shape == new_t.shape:
                        new_state[key] = old_t
                        loaded += 1
                    elif old_t.dim() == new_t.dim() and all(
                        o <= n for o, n in zip(old_t.shape, new_t.shape)
                    ):
                        slices = tuple(slice(0, o) for o in old_t.shape)
                        new_state[key][slices] = old_t
                        loaded += 1
                    else:
                        skipped += 1
                network.load_state_dict(new_state)
                msg = f"Warm-start: {loaded} loaded, {skipped} new/skipped"
                if reset_value_head:
                    msg += f", {reset} value_head reset (layer count changed)"
                print(msg)
                for f in [history_path, progress_path]:
                    if os.path.exists(f):
                        os.remove(f)
        else:
            print("Cold start — no checkpoint found")
    else:
        print("Cold start (forced)")
        for f in [history_path, progress_path]:
            if os.path.exists(f):
                os.remove(f)

    # No runtime calibration — use pre-calibrated training set HPs.
    player_max_hp = 70

    # Replay buffer
    replay = ReplayBuffer(max_steps=replay_capacity)
    print(f"Replay buffer: capacity {replay_capacity:,} steps")

    for gen in range(start_gen, num_generations + 1):
        t0 = time.time()

        # Export ONNX
        onnx_path = export_onnx(network, onnx_dir)

        # Sample encounters grouped by HP (shared with train.py)
        batches = sample_combat_batches(encounter_set, combats_per_gen, gen)
        seeds = [gen * 100_000 + i for i in range(combats_per_gen)]

        # Self-play: MCTS combats (one call per HP level)
        all_outcomes = []
        all_final_hps = []
        gen_states, gen_act_feat, gen_act_masks = [], [], []
        gen_hand_ids, gen_action_ids, gen_policies = [], [], []
        gen_visits, gen_q_values = [], []
        gen_rewards = []
        gen_mcts_values = []
        gen_combat_indices = []
        combat_offset = 0
        seed_idx = 0

        for b_enc, b_dks, b_rels, b_hp, b_count in batches:
            if not b_enc:
                continue
            b_seeds = [seeds[seed_idx + i] if seed_idx + i < len(seeds)
                       else gen * 100_000 + seed_idx + i
                       for i in range(b_count)]
            seed_idx += b_count
            rollout = sts2_engine.betaone_mcts_selfplay(
                encounters_json=json.dumps(b_enc),
                decks_json=json.dumps(b_dks),
                player_hp=b_hp,
                player_max_hp=player_max_hp,
                player_max_energy=3,
                relics_json=json.dumps(b_rels),
                potions_json="[]",
                monster_data_json=monster_json,
                enemy_profiles_json=profiles_json,
                onnx_path=onnx_path,
                card_vocab_json=card_vocab_json,
                num_sims=num_sims,
                temperature=temperature,
                seeds=b_seeds,
                gen_id=gen,
                add_noise=True,
                turn_boundary_eval=turn_boundary_eval,
                dense_value_targets=dense_value_targets,
                c_puct=c_puct,
                pomcp=pomcp,
                noise_frac=noise_frac,
                pw_k=pw_k,
            )

            n_steps = rollout["total_steps"]
            if n_steps == 0:
                continue

            # Offset combat indices
            ci = np.array(rollout["combat_indices"], dtype=np.int64) + combat_offset
            combat_offset += len(rollout["outcomes"])

            gen_states.extend(np.array(rollout["states"], dtype=np.float32).reshape(-1, STATE_DIM))
            gen_act_feat.extend(np.array(rollout["action_features"], dtype=np.float32).reshape(-1, MAX_ACTIONS * ACTION_DIM))
            gen_act_masks.extend(np.array(rollout["action_masks"]).reshape(-1, MAX_ACTIONS))
            gen_hand_ids.extend(np.array(rollout["hand_card_ids"], dtype=np.int64).reshape(-1, MAX_HAND))
            gen_action_ids.extend(np.array(rollout["action_card_ids"], dtype=np.int64).reshape(-1, MAX_ACTIONS))
            gen_policies.extend(np.array(rollout["policies"], dtype=np.float32).reshape(-1, MAX_ACTIONS))
            if q_target_mix > 0:
                gen_visits.extend(np.array(rollout["child_visits"], dtype=np.int64).reshape(-1, MAX_ACTIONS))
                gen_q_values.extend(np.array(rollout["child_q_values"], dtype=np.float32).reshape(-1, MAX_ACTIONS))
            gen_combat_indices.extend(ci)
            gen_mcts_values.extend(np.array(rollout["mcts_values"], dtype=np.float32))
            if dense_value_targets:
                gen_rewards.extend(np.array(rollout["rewards"], dtype=np.float32))
            all_outcomes.extend(rollout["outcomes"])
            all_final_hps.extend(rollout["final_hps"])

        T = len(gen_states)
        if T == 0:
            print(f"Gen {gen}: no steps, skipping")
            continue

        # Build value targets
        combat_indices = np.array(gen_combat_indices, dtype=np.int64)
        gen_values = np.zeros(T, dtype=np.float32)
        if mcts_bootstrap:
            # MCTS-bootstrapped: use search root values directly as targets.
            # The search already assigns credit through tree backup — terminal
            # HP-scaled win/loss is the only reward signal.
            gen_values = np.array(gen_mcts_values, dtype=np.float32)
        elif dense_value_targets:
            # Monte Carlo returns: G_t = sum_{k=t}^{T} gamma^{k-t} * r_k
            rewards_arr = np.array(gen_rewards, dtype=np.float32)
            for ci in range(len(all_outcomes)):
                mask = combat_indices == ci
                ep_r = rewards_arr[mask].copy()
                G = 0.0
                for t in reversed(range(len(ep_r))):
                    G = ep_r[t] + gamma * G
                    ep_r[t] = G
                gen_values[mask] = ep_r
        else:
            # Broadcast game outcome (HP-scaled: win → 1.0 + 0.3*hp_frac, lose → -1.0)
            for ci, outcome in enumerate(all_outcomes):
                mask = combat_indices == ci
                if outcome == "win":
                    hp_frac = max(all_final_hps[ci], 0) / max(player_max_hp, 1)
                    gen_values[mask] = 1.0 + 0.3 * hp_frac
                else:
                    gen_values[mask] = -1.0

        # Mix Q-based softmax into the policy target. Default mix=0 keeps the
        # standard AlphaZero visit-distribution target. mix>0 attacks the echo
        # chamber where visits inherit a sharp prior even when Q values say
        # actions are competitive.
        if q_target_mix > 0:
            visits_arr = np.array(gen_visits, dtype=np.int64)
            q_arr = np.array(gen_q_values, dtype=np.float32)
            mask_arr = np.array(gen_act_masks, dtype=bool)
            mixed = compute_mixed_policy_target(
                visits_arr, q_arr, mask_arr, q_target_mix, q_target_temp
            )
            gen_policies = list(mixed)

        # Add to replay buffer
        replay.add_generation(
            states=gen_states,
            act_feat=gen_act_feat,
            act_masks=gen_act_masks,
            hand_ids=gen_hand_ids,
            action_ids=gen_action_ids,
            policies=gen_policies,
            values=gen_values,
        )

        # Stats
        n_combats = len(all_outcomes)
        n_wins = sum(1 for o in all_outcomes if o == "win")
        win_rate = n_wins / max(n_combats, 1)
        win_hps = [hp for hp, o in zip(all_final_hps, all_outcomes) if o == "win"]
        avg_hp = np.mean(win_hps) if win_hps else 0.0

        # Train from replay buffer
        network.train()
        total_ploss = 0.0
        total_vloss = 0.0
        n_updates = 0

        buf_size = len(replay)
        updates_per_epoch = max(1, buf_size // batch_size)

        for _epoch in range(train_epochs):
            for _ in range(updates_per_epoch):
                (b_states, b_act_feat, b_act_masks,
                 b_hand_ids, b_action_ids,
                 b_policies, b_values) = replay.sample_tensors(batch_size)

                # Reshape action features from flat to (B, MAX_ACTIONS, ACTION_DIM)
                b_act_feat = b_act_feat.reshape(-1, MAX_ACTIONS, ACTION_DIM)

                metrics = train_batch(
                    network, optimizer,
                    b_states, b_act_feat, b_act_masks,
                    b_hand_ids, b_action_ids,
                    b_policies, b_values,
                    value_coef=value_coef,
                )
                total_ploss += metrics["policy_loss"]
                total_vloss += metrics["value_loss"]
                n_updates += 1

        n = max(n_updates, 1)
        avg_ploss = total_ploss / n
        avg_vloss = total_vloss / n

        elapsed = time.time() - t0

        print(
            f"Gen {gen:4d} | "
            f"win {win_rate:5.1%} | "
            f"hp {avg_hp:4.1f} | "
            f"steps {T:5d} | "
            f"buf {buf_size:6d} | "
            f"pi {avg_ploss:.3f} | "
            f"v {avg_vloss:.3f} | "
            f"sims {num_sims} | "
            f"{elapsed:.1f}s"
        )

        # Log
        record = {
            "gen": gen,
            "win_rate": round(win_rate, 4),
            "avg_hp": round(float(avg_hp), 1),
            "steps": T,
            "buffer_size": buf_size,
            "episodes": n_combats,
            "encounter_set": encounter_set_id,
            "policy_loss": round(avg_ploss, 5),
            "value_loss": round(avg_vloss, 5),
            "num_sims": num_sims,
            "gen_time": round(elapsed, 2),
            "timestamp": time.time(),
        }
        with open(history_path, "a") as f:
            f.write(json.dumps(record) + "\n")
        with open(progress_path, "w") as f:
            record["num_generations"] = num_generations
            record["best_win_rate"] = round(max(best_win_rate, win_rate), 4)
            json.dump(record, f, indent=2)

        best_win_rate = max(best_win_rate, win_rate)

        # Checkpoints
        ckpt_data = {
            "gen": gen,
            "arch_meta": network.arch_meta(),
            "model_state_dict": network.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "win_rate": win_rate,
            "num_cards": num_cards,
        }
        latest_path = os.path.join(output_dir, "betaone_latest.pt")
        torch.save(ckpt_data, latest_path)
        if gen % 10 == 0 or win_rate >= best_win_rate:
            torch.save(ckpt_data, os.path.join(output_dir, f"betaone_gen{gen}.pt"))

        # Periodic eval curve: append eval.jsonl / value_eval.jsonl so the TUI
        # and downstream plots can track decision-quality progress across
        # training, not just win rate. WR is compressed near the top of the
        # skill curve; eval pass rate moves earlier and at higher resolution.
        if eval_every > 0 and gen % eval_every == 0:
            try:
                from .eval import run_eval, run_value_eval
                from .suite import compute_eval_suite, suite_id as _suite_id
                bench_dir = os.path.join(output_dir, "benchmarks")
                os.makedirs(bench_dir, exist_ok=True)
                _sid = _suite_id(compute_eval_suite())
                pol = run_eval(latest_path)
                val = run_value_eval(latest_path)
                # Mirror Experiment.save_eval / save_value_eval entry shape.
                pol_entry = {
                    "suite": _sid, "timestamp": time.time(), "gen": gen,
                    "passed": pol["passed"], "total": pol["total"],
                    "score": round(pol["passed"] / max(pol["total"], 1), 4),
                    "end_turn_avg": pol.get("end_turn_avg"),
                    "end_turn_high": pol.get("end_turn_high", 0),
                    "by_category": {
                        cat: {"passed": sum(1 for r in rs if r["passed"]), "total": len(rs)}
                        for cat, rs in pol.get("by_category", {}).items()
                    },
                }
                val_entry = {
                    "suite": _sid, "timestamp": time.time(), "gen": gen,
                    "passed": val["passed"], "total": val["total"],
                    "score": round(val["passed"] / max(val["total"], 1), 4),
                    "by_category": val.get("by_category", {}),
                }
                with open(os.path.join(bench_dir, "eval.jsonl"), "a", encoding="utf-8") as f:
                    f.write(json.dumps(pol_entry) + "\n")
                with open(os.path.join(bench_dir, "value_eval.jsonl"), "a", encoding="utf-8") as f:
                    f.write(json.dumps(val_entry) + "\n")
                print(f"       eval: {pol['passed']}/{pol['total']} "
                      f"({pol_entry['score']:.0%}) | value: {val['passed']}/{val['total']} "
                      f"({val_entry['score']:.0%})")
            except Exception as e:
                print(f"       [eval_every] skipped gen {gen}: {e}")

    print(f"\nTraining complete. Best win rate: {best_win_rate:.1%}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="BetaOne self-play training")
    parser.add_argument("--encounter-set", required=True,
                        help="Encounter set id (e.g. lean-decks-v1)")
    parser.add_argument("--generations", type=int, default=2000)
    parser.add_argument("--combats", type=int, default=256)
    parser.add_argument("--sims", type=int, default=150)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--output-dir", default="betaone_checkpoints")
    parser.add_argument("--cold-start", action="store_true",
                        help="Ignore existing checkpoint, start from scratch")
    parser.add_argument("--replay-capacity", type=int, default=200_000,
                        help="Replay buffer capacity in steps")
    args = parser.parse_args()

    train(
        encounter_set_id=args.encounter_set,
        num_generations=args.generations,
        combats_per_gen=args.combats,
        num_sims=args.sims,
        lr=args.lr,
        output_dir=args.output_dir,
        cold_start=args.cold_start,
        replay_capacity=args.replay_capacity,
    )


if __name__ == "__main__":
    main()
