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

from .calibrate import calibrate_all
from .curriculum import CombatCurriculum, TIER_CONFIGS
from .deck_gen import build_random_deck_json, _make_starter
from .network import (
    ACTION_DIM,
    MAX_ACTIONS,
    MAX_HAND,
    STATE_DIM,
    BetaOneNetwork,
    export_onnx,
)

# ---------------------------------------------------------------------------
# Data loading (shared with train.py)
# ---------------------------------------------------------------------------

_DATA_DIR = Path(__file__).resolve().parents[4] / "STS2-Agent" / "mcp_server" / "data" / "eng"
_SOLVER_DIR = Path(__file__).resolve().parents[1]


def _load_json(filename: str) -> str:
    path = _DATA_DIR / filename
    if not path.exists():
        return "[]"
    return path.read_text(encoding="utf-8")


def _load_solver_json(filename: str) -> str:
    path = _SOLVER_DIR / filename
    if not path.exists():
        return "{}"
    return path.read_text(encoding="utf-8")


def _build_monster_data_json() -> str:
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


def _build_card_vocab(output_dir: str) -> tuple[dict[str, int], str]:
    vocab_path = os.path.join(output_dir, "card_vocab.json")
    if os.path.exists(vocab_path):
        with open(vocab_path, encoding="utf-8") as f:
            vocab = json.load(f)
        return vocab, json.dumps(vocab)
    cards_raw = json.loads(_load_json("cards.json"))
    vocab: dict[str, int] = {"<PAD>": 0, "<UNK>": 1}
    for c in cards_raw:
        base_id = c["id"].rstrip("+")
        if base_id not in vocab:
            vocab[base_id] = len(vocab)
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, indent=2)
    print(f"Card vocab: {len(vocab)} entries")
    return vocab, json.dumps(vocab)


def _find_latest_checkpoint(output_dir: str) -> str | None:
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
    num_generations: int = 2000,
    combats_per_gen: int = 256,
    num_sims: int = 150,
    lr: float = 3e-4,
    value_coef: float = 1.0,
    train_epochs: int = 4,
    batch_size: int = 512,
    temperature: float = 1.0,
    output_dir: str = "betaone_checkpoints",
    skip_to_final: bool = False,
    recorded_encounters: bool = False,
    mixed: bool = False,
    recorded_frac: float = 0.5,
    replay_capacity: int = 200_000,
    cold_start: bool = False,
):
    os.makedirs(output_dir, exist_ok=True)
    onnx_dir = os.path.join(output_dir, "onnx")
    os.makedirs(onnx_dir, exist_ok=True)

    # Load game data
    monster_json = _build_monster_data_json()
    profiles_json = _load_solver_json("enemy_profiles.json")
    enc_pool_path = str(_SOLVER_DIR / "encounter_pool.json")

    # Card vocabulary
    card_vocab, card_vocab_json = _build_card_vocab(output_dir)
    num_cards = len(card_vocab)

    # Network + optimizer
    network = BetaOneNetwork(num_cards=num_cards)
    optimizer = torch.optim.Adam(network.parameters(), lr=lr)
    print(f"BetaOne self-play: {network.param_count():,} params, {num_cards} card vocab")

    # Curriculum
    # recorded_encounters.jsonl lives alongside checkpoints; resolve relative
    # to the project root (4 dirs up from this file) so it works regardless
    # of the working directory.
    project_root = Path(__file__).resolve().parents[4]
    recorded_path = str(project_root / output_dir / "recorded_encounters.jsonl")
    curriculum = CombatCurriculum(
        encounter_pool_path=enc_pool_path,
        recorded_encounters_path=recorded_path,
    )

    if mixed:
        if not curriculum.recorded_encounters:
            print("No recorded encounters found — run the game bot first to build the pool")
            return
        curriculum.tier = curriculum.max_tier
        n_rec = len(curriculum.recorded_encounters)
        print(f"Mixed mode: {recorded_frac:.0%} recorded ({n_rec} encounters) + {1-recorded_frac:.0%} final exam")
    elif recorded_encounters:
        if not curriculum.recorded_encounters:
            print("No recorded encounters found — run the game bot first to build the pool")
            return
        curriculum.use_recorded_only = True
        curriculum.tier = curriculum.max_tier
        print(f"Training on {len(curriculum.recorded_encounters)} recorded encounters")
    elif skip_to_final:
        curriculum.tier = curriculum.max_tier
        curriculum.consecutive_good = 0
        curriculum.gens_at_tier = 0

    best_win_rate = 0.0
    start_gen = 1

    history_path = os.path.join(output_dir, "betaone_history.jsonl")
    progress_path = os.path.join(output_dir, "betaone_progress.json")

    # Resume from checkpoint (unless cold start)
    if not cold_start:
        latest_ckpt = _find_latest_checkpoint(output_dir)
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
                loaded, skipped = 0, 0
                for key in new_state:
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
                print(f"Warm-start: {loaded} loaded, {skipped} new/skipped")
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

    # HP calibration for recorded encounters — skip on cold start
    # (random network can't win anything, so calibration fails).
    # Use existing calibrated_hp values from the file instead.
    avg_calibrated_hp: float | None = None
    if recorded_encounters and not cold_start:
        onnx_path = export_onnx(network, onnx_dir)
        print("Running initial HP calibration...")
        curriculum.recorded_encounters, avg_calibrated_hp = calibrate_all(
            curriculum.recorded_encounters, monster_json, profiles_json,
            card_vocab_json, onnx_path,
            encounters_path=recorded_path,
            num_sims=50, combats=32,
        )
        if avg_calibrated_hp is not None:
            print(f"Avg calibrated HP: {avg_calibrated_hp:.1f}")
    elif recorded_encounters:
        # Cold start: use existing calibrated HP from file, or default to max
        hp_vals = [r.get("calibrated_hp", 70) for r in curriculum.recorded_encounters]
        avg_calibrated_hp = sum(hp_vals) / len(hp_vals) if hp_vals else None
        print(f"Cold start: using existing calibrated HP (avg={avg_calibrated_hp:.1f})" if avg_calibrated_hp else
              "Cold start: no calibrated HP, using default 70")

    # Replay buffer
    replay = ReplayBuffer(max_steps=replay_capacity)
    print(f"Replay buffer: capacity {replay_capacity:,} steps")

    for gen in range(start_gen, num_generations + 1):
        t0 = time.time()

        # Export ONNX
        onnx_path = export_onnx(network, onnx_dir)

        # Recalibrate every 200 gens
        if recorded_encounters and gen % 200 == 0 and gen > start_gen:
            print(f"Gen {gen}: recalibrating recorded encounters...")
            curriculum.recorded_encounters, avg_calibrated_hp = calibrate_all(
                curriculum.recorded_encounters, monster_json, profiles_json,
                card_vocab_json, onnx_path,
                encounters_path=recorded_path,
                num_sims=50, combats=32,
            )

        # Sample encounters and decks
        cfg = curriculum.config
        seeds = [gen * 100_000 + i for i in range(combats_per_gen)]

        if mixed:
            # Mixed mode: split between recorded encounters and final exam
            n_recorded = int(combats_per_gen * recorded_frac)
            n_exam = combats_per_gen - n_recorded

            # Recorded encounters portion
            curriculum.use_recorded_only = True
            rec_enc = curriculum.sample_encounters(n_recorded)
            rec_samples = getattr(curriculum, "_recorded_samples", None)
            rec_decks = [json.loads(curriculum.sample_deck_json(combat_idx=i))
                         for i in range(n_recorded)]

            # Final exam portion
            curriculum.use_recorded_only = False
            curriculum.tier = curriculum.max_tier
            exam_enc = curriculum.sample_encounters(n_exam)
            exam_decks = [json.loads(curriculum.sample_deck_json(combat_idx=None))
                          for _ in range(n_exam)]

            encounters = rec_enc + exam_enc
            decks = rec_decks + exam_decks
        else:
            encounters = curriculum.sample_encounters(combats_per_gen)
            decks = [json.loads(curriculum.sample_deck_json(combat_idx=i))
                     for i in range(combats_per_gen)]
            rec_samples = getattr(curriculum, "_recorded_samples", None)
            n_recorded = combats_per_gen if recorded_encounters else 0

        # Group combats by HP level for batched self-play
        from collections import defaultdict
        hp_groups: dict[int, tuple[list, list, list]] = defaultdict(lambda: ([], [], []))
        for i in range(combats_per_gen):
            if mixed and i < n_recorded:
                rec = rec_samples[i] if rec_samples and i < len(rec_samples) else None
            elif not mixed and recorded_encounters:
                rec = rec_samples[i] if rec_samples and i < len(rec_samples) else None
            else:
                rec = None
            hp = rec.get("calibrated_hp", rec.get("player_hp", 70)) if rec else cfg.player_hp
            hp_groups[hp][0].append(encounters[i])
            hp_groups[hp][1].append(decks[i])
            hp_groups[hp][2].append(seeds[i])

        # Self-play: MCTS combats (one call per HP level)
        all_outcomes = []
        all_final_hps = []
        gen_states, gen_act_feat, gen_act_masks = [], [], []
        gen_hand_ids, gen_action_ids, gen_policies = [], [], []
        gen_combat_indices = []
        combat_offset = 0

        for hp, (grp_enc, grp_dks, grp_seeds) in hp_groups.items():
            rollout = sts2_engine.betaone_mcts_selfplay(
                encounters_json=json.dumps(grp_enc),
                decks_json=json.dumps(grp_dks),
                player_hp=hp,
                player_max_hp=70,
                player_max_energy=3,
                relics=[],
                potions_json="[]",
                monster_data_json=monster_json,
                enemy_profiles_json=profiles_json,
                onnx_path=onnx_path,
                card_vocab_json=card_vocab_json,
                num_sims=num_sims,
                temperature=temperature,
                seeds=grp_seeds,
                gen_id=gen,
                add_noise=True,
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
            gen_combat_indices.extend(ci)
            all_outcomes.extend(rollout["outcomes"])
            all_final_hps.extend(rollout["final_hps"])

        T = len(gen_states)
        if T == 0:
            print(f"Gen {gen}: no steps, skipping")
            continue

        # Build value targets from outcomes
        combat_indices = np.array(gen_combat_indices, dtype=np.int64)
        gen_values = np.zeros(T, dtype=np.float32)
        for ci, outcome in enumerate(all_outcomes):
            mask = combat_indices == ci
            gen_values[mask] = 1.0 if outcome == "win" else -1.0

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

        # Curriculum update
        tier_before = curriculum.tier
        tier_change = curriculum.update(win_rate) if not recorded_encounters else "hold"

        print(
            f"Gen {gen:4d} | "
            f"win {win_rate:5.1%} | "
            f"hp {avg_hp:4.1f} | "
            f"steps {T:5d} | "
            f"buf {buf_size:6d} | "
            f"pi {avg_ploss:.3f} | "
            f"v {avg_vloss:.3f} | "
            f"T{tier_before}{'UP' if tier_change == 'promoted' else '  '} | "
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
            "tier": tier_before,
            "tier_name": cfg.name,
            "tier_change": tier_change,
            "gens_at_tier": curriculum.gens_at_tier,
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
            "model_state_dict": network.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "win_rate": win_rate,
            "tier": curriculum.tier,
            "num_cards": num_cards,
        }
        torch.save(ckpt_data, os.path.join(output_dir, "betaone_latest.pt"))
        if gen % 10 == 0 or win_rate >= best_win_rate:
            torch.save(ckpt_data, os.path.join(output_dir, f"betaone_gen{gen}.pt"))

    print(f"\nTraining complete. Best win rate: {best_win_rate:.1%}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="BetaOne self-play training")
    parser.add_argument("--generations", type=int, default=2000)
    parser.add_argument("--combats", type=int, default=256)
    parser.add_argument("--sims", type=int, default=150)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--output-dir", default="betaone_checkpoints")
    parser.add_argument("--final-exam", action="store_true")
    parser.add_argument("--recorded-encounters", action="store_true",
                        help="Train on recorded death encounters from live games")
    parser.add_argument("--mixed", action="store_true",
                        help="Train on mix of recorded encounters and final exam")
    parser.add_argument("--recorded-frac", type=float, default=0.5,
                        help="Fraction of combats from recorded encounters in mixed mode")
    parser.add_argument("--cold-start", action="store_true",
                        help="Ignore existing checkpoint, start from scratch")
    parser.add_argument("--replay-capacity", type=int, default=200_000,
                        help="Replay buffer capacity in steps")
    args = parser.parse_args()

    train(
        num_generations=args.generations,
        combats_per_gen=args.combats,
        num_sims=args.sims,
        lr=args.lr,
        output_dir=args.output_dir,
        skip_to_final=args.final_exam,
        recorded_encounters=args.recorded_encounters,
        mixed=args.mixed,
        recorded_frac=args.recorded_frac,
        cold_start=args.cold_start,
        replay_capacity=args.replay_capacity,
    )


if __name__ == "__main__":
    main()
