"""BetaOne AlphaZero-style self-play training.

Replaces PPO with MCTS self-play: network learns from MCTS visit
distributions (policy) and game outcomes (value).

Usage:
    python -m sts2_solver.betaone.selfplay_train [--generations N] [--combats N]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import sts2_engine

from .curriculum import CombatCurriculum
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
    num_generations: int = 200,
    combats_per_gen: int = 256,
    num_sims: int = 100,
    lr: float = 3e-4,
    value_coef: float = 1.0,
    train_epochs: int = 10,
    batch_size: int = 256,
    temperature: float = 1.0,
    output_dir: str = "betaone_checkpoints",
    skip_to_final: bool = False,
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
    curriculum = CombatCurriculum(encounter_pool_path=enc_pool_path)
    if skip_to_final:
        curriculum.tier = curriculum.max_tier
        curriculum.consecutive_good = 0
        curriculum.gens_at_tier = 0

    best_win_rate = 0.0
    start_gen = 1

    history_path = os.path.join(output_dir, "betaone_history.jsonl")
    progress_path = os.path.join(output_dir, "betaone_progress.json")

    # Resume from checkpoint
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
            print("Architecture mismatch — cold start")
            for f in [history_path, progress_path]:
                if os.path.exists(f):
                    os.remove(f)
    else:
        print("Cold start — no checkpoint found")

    for gen in range(start_gen, num_generations + 1):
        t0 = time.time()

        # Export ONNX
        onnx_path = export_onnx(network, onnx_dir)

        # Sample encounters and decks
        cfg = curriculum.config
        encounters = curriculum.sample_encounters(combats_per_gen)
        decks = [json.loads(curriculum.sample_deck_json()) for _ in range(combats_per_gen)]
        seeds = [gen * 100_000 + i for i in range(combats_per_gen)]

        # Self-play: MCTS combats
        rollout = sts2_engine.betaone_mcts_selfplay(
            encounters_json=json.dumps(encounters),
            decks_json=json.dumps(decks),
            player_hp=cfg.player_hp,
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
            seeds=seeds,
            gen_id=gen,
            add_noise=True,
        )

        T = rollout["total_steps"]
        if T == 0:
            print(f"Gen {gen}: no steps, skipping")
            continue

        # Reshape data
        states = torch.tensor(rollout["states"], dtype=torch.float32).reshape(T, STATE_DIM)
        hand_ids = torch.tensor(rollout["hand_card_ids"], dtype=torch.long).reshape(T, MAX_HAND)
        action_ids = torch.tensor(rollout["action_card_ids"], dtype=torch.long).reshape(T, MAX_ACTIONS)
        act_feat = torch.tensor(rollout["action_features"], dtype=torch.float32).reshape(T, MAX_ACTIONS, ACTION_DIM)
        act_masks = torch.tensor(rollout["action_masks"]).reshape(T, MAX_ACTIONS)
        policies = torch.tensor(rollout["policies"], dtype=torch.float32).reshape(T, MAX_ACTIONS)
        combat_indices = torch.tensor(rollout["combat_indices"], dtype=torch.long)

        # Assign value targets: +1 for wins, -1 for losses
        outcomes = rollout["outcomes"]
        values_target = torch.zeros(T, dtype=torch.float32)
        combat_offset = 0
        for ci, outcome in enumerate(outcomes):
            mask = combat_indices == ci
            values_target[mask] = 1.0 if outcome == "win" else -1.0

        # Stats
        n_combats = len(outcomes)
        n_wins = sum(1 for o in outcomes if o == "win")
        win_rate = n_wins / max(n_combats, 1)
        win_hps = [hp for hp, o in zip(rollout["final_hps"], outcomes) if o == "win"]
        avg_hp = np.mean(win_hps) if win_hps else 0.0

        # Train
        network.train()
        indices = np.arange(T)
        total_ploss = 0.0
        total_vloss = 0.0
        n_updates = 0

        for _epoch in range(train_epochs):
            np.random.shuffle(indices)
            for start in range(0, T, batch_size):
                end = min(start + batch_size, T)
                b = torch.from_numpy(indices[start:end]).long()

                metrics = train_batch(
                    network, optimizer,
                    states[b], act_feat[b], act_masks[b],
                    hand_ids[b], action_ids[b],
                    policies[b], values_target[b],
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
        tier_change = curriculum.update(win_rate)

        print(
            f"Gen {gen:4d} | "
            f"win {win_rate:5.1%} | "
            f"hp {avg_hp:4.1f} | "
            f"steps {T:5d} | "
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
    parser.add_argument("--generations", type=int, default=200)
    parser.add_argument("--combats", type=int, default=256)
    parser.add_argument("--sims", type=int, default=100)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--output-dir", default="betaone_checkpoints")
    parser.add_argument("--final-exam", action="store_true")
    args = parser.parse_args()

    train(
        num_generations=args.generations,
        combats_per_gen=args.combats,
        num_sims=args.sims,
        lr=args.lr,
        output_dir=args.output_dir,
        skip_to_final=args.final_exam,
    )


if __name__ == "__main__":
    main()
