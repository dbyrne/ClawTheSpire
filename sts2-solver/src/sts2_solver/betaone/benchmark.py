"""Benchmark BetaOne checkpoints against fixed evaluation sets.

Two evaluation sets (columns):
  1. Final exam:  mixed encounters from all curriculum tiers, 70 HP
  2. Recorded:    43 death encounters from live games, frozen calibrated HP

Three evaluation modes (rows):
  1. PPO:           policy sampling (no MCTS tree search)
  2. PPO + MCTS:    same PPO checkpoint, MCTS at inference
  3. MCTS cold:     MCTS self-play trained checkpoint, MCTS at inference

Usage:
    python -m sts2_solver.betaone.benchmark                         # full 3x2 table
    python -m sts2_solver.betaone.benchmark --ppo path/to.pt        # specific PPO checkpoint
    python -m sts2_solver.betaone.benchmark --mcts path/to.pt       # specific MCTS checkpoint
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import torch

import sts2_engine

from .network import BetaOneNetwork, export_onnx
from .curriculum import CombatCurriculum, TIER_CONFIGS
from .deck_gen import build_random_deck_json, _make_starter, lookup_card

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_DATA_DIR = Path(__file__).resolve().parents[4] / "STS2-Agent" / "mcp_server" / "data" / "eng"
_SOLVER_DIR = Path(__file__).resolve().parents[1]
_CHECKPOINTS = Path(__file__).resolve().parents[4] / "betaone_checkpoints"
_BENCHMARK_FILE = _CHECKPOINTS / "benchmark_recorded.jsonl"


def _load_json(filename: str) -> str:
    path = _DATA_DIR / filename
    return path.read_text(encoding="utf-8") if path.exists() else "[]"


def _load_solver_json(filename: str) -> str:
    path = _SOLVER_DIR / filename
    return path.read_text(encoding="utf-8") if path.exists() else "{}"


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


def _build_card_vocab() -> tuple[dict, str]:
    vocab_path = _CHECKPOINTS / "card_vocab.json"
    with open(vocab_path, encoding="utf-8") as f:
        vocab = json.load(f)
    return vocab, json.dumps(vocab)


def _load_checkpoint(path: str) -> tuple[BetaOneNetwork, dict]:
    card_vocab, _ = _build_card_vocab()
    net = BetaOneNetwork(num_cards=len(card_vocab))
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    net.load_state_dict(ckpt["model_state_dict"])
    net.eval()
    return net, ckpt


# ---------------------------------------------------------------------------
# Eval primitives
# ---------------------------------------------------------------------------

def _sample_final_exam(combats: int, seed: int = 42) -> tuple[list, list]:
    """Sample encounters and decks for final exam."""
    random.seed(seed)
    cur = CombatCurriculum(encounter_pool_path=str(_SOLVER_DIR / "encounter_pool.json"))
    cur.tier = cur.max_tier
    encounters = cur.sample_encounters(combats)
    decks = [json.loads(cur.sample_deck_json(combat_idx=i)) for i in range(combats)]
    return encounters, decks


def _load_recorded_benchmark() -> list[dict]:
    """Load frozen recorded encounters benchmark."""
    if not _BENCHMARK_FILE.exists():
        return []
    with open(_BENCHMARK_FILE, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def eval_policy(
    onnx_path: str, card_vocab_json: str,
    monster_json: str, profiles_json: str,
    encounters: list, decks: list, player_hp: int,
    seeds: list[int],
) -> float:
    """Evaluate using policy sampling (PPO-style, no MCTS)."""
    r = sts2_engine.collect_betaone_rollouts(
        encounters_json=json.dumps(encounters),
        decks_json=json.dumps(decks),
        player_hp=player_hp, player_max_hp=70, player_max_energy=3,
        relics_json="[]", potions_json="[]",
        monster_data_json=monster_json,
        enemy_profiles_json=profiles_json,
        onnx_path=onnx_path,
        temperature=0.01,
        seeds=seeds,
        gen_id=0,
        card_vocab_json=card_vocab_json,
    )
    wins = sum(1 for o in r["outcomes"] if o == "win")
    return wins / max(len(r["outcomes"]), 1)


def eval_mcts(
    onnx_path: str, card_vocab_json: str,
    monster_json: str, profiles_json: str,
    encounters: list, decks: list, player_hp: int,
    seeds: list[int], num_sims: int = 400,
) -> float:
    """Evaluate using MCTS search."""
    r = sts2_engine.betaone_mcts_selfplay(
        encounters_json=json.dumps(encounters),
        decks_json=json.dumps(decks),
        player_hp=player_hp, player_max_hp=70, player_max_energy=3,
        relics_json="[]", potions_json="[]",
        monster_data_json=monster_json,
        enemy_profiles_json=profiles_json,
        onnx_path=onnx_path,
        card_vocab_json=card_vocab_json,
        num_sims=num_sims, temperature=0.0,
        seeds=seeds,
        add_noise=False,
    )
    wins = sum(1 for o in r["outcomes"] if o == "win")
    return wins / max(len(r["outcomes"]), 1)


def eval_final_exam(onnx_path: str, card_vocab_json: str,
                    monster_json: str, profiles_json: str,
                    use_mcts: bool, combats: int = 128) -> float:
    """Final exam benchmark."""
    encounters, decks = _sample_final_exam(combats)
    seeds = list(range(combats))
    if use_mcts:
        return eval_mcts(onnx_path, card_vocab_json, monster_json, profiles_json,
                         encounters, decks, 70, seeds)
    return eval_policy(onnx_path, card_vocab_json, monster_json, profiles_json,
                       encounters, decks, 70, seeds)


def eval_recorded(onnx_path: str, card_vocab_json: str,
                  monster_json: str, profiles_json: str,
                  use_mcts: bool, combats_per: int = 32) -> float:
    """Recorded encounters benchmark with frozen HP."""
    records = _load_recorded_benchmark()
    if not records:
        return 0.0

    total_wins = 0
    total_games = 0

    for i, rec in enumerate(records):
        enemy_ids = rec["enemy_ids"]
        hp = rec.get("calibrated_hp", 70)
        deck_ids = rec.get("deck", [])

        deck = []
        for cid in deck_ids:
            try:
                deck.append(lookup_card(cid.rstrip("+")))
            except Exception:
                pass
        if not deck:
            continue

        encounters = [enemy_ids] * combats_per
        decks = [deck] * combats_per
        seeds = [42 * 1000 + i * 100 + j for j in range(combats_per)]

        if use_mcts:
            wr = eval_mcts(onnx_path, card_vocab_json, monster_json, profiles_json,
                           encounters, decks, hp, seeds)
        else:
            wr = eval_policy(onnx_path, card_vocab_json, monster_json, profiles_json,
                             encounters, decks, hp, seeds)

        wins = int(wr * combats_per + 0.5)
        total_wins += wins
        total_games += combats_per

    return total_wins / max(total_games, 1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_benchmark(
    ppo_checkpoint: str | None = None,
    mcts_checkpoint: str | None = None,
    combats: int = 256,
    recorded_combats: int = 32,
):
    """Run 3x2 benchmark table."""
    card_vocab, card_vocab_json = _build_card_vocab()
    monster_json = _build_monster_data_json()
    profiles_json = _load_solver_json("enemy_profiles.json")

    # Find checkpoints
    if ppo_checkpoint is None:
        ppo_path = Path(__file__).resolve().parents[4] / "betaone_ppo_mixed" / "betaone_latest.pt"
        if not ppo_path.exists():
            ppo_path = _CHECKPOINTS / "betaone_latest.pt"
        ppo_checkpoint = str(ppo_path)

    if mcts_checkpoint is None:
        # Self-play checkpoint (solver-local dir)
        mcts_path = Path(__file__).resolve().parents[3] / "betaone_checkpoints" / "betaone_latest.pt"
        if not mcts_path.exists():
            mcts_path = None
        mcts_checkpoint = str(mcts_path) if mcts_path else None

    results = {}

    # Pre-sample encounters so all models face the same ones
    print("Sampling shared encounters...")
    fe_encounters, fe_decks = _sample_final_exam(combats)
    fe_seeds = list(range(combats))
    rec_records = _load_recorded_benchmark()

    # Collect checkpoints to eval: (label, path, use_mcts)
    evals: list[tuple[str, str, bool]] = []
    if Path(ppo_checkpoint).exists():
        evals.append(("PPO", ppo_checkpoint, False))
        evals.append(("PPO+MCTS", ppo_checkpoint, True))
    else:
        print(f"PPO checkpoint not found: {ppo_checkpoint}")
    if mcts_checkpoint and Path(mcts_checkpoint).exists():
        evals.append(("MCTS", mcts_checkpoint, True))
    else:
        print(f"MCTS checkpoint not found: {mcts_checkpoint}")

    last_ckpt_path = None
    for label, ckpt_path, use_mcts in evals:
        if ckpt_path != last_ckpt_path:
            net, ckpt = _load_checkpoint(ckpt_path)
            gen = ckpt.get("gen", "?")
            onnx_path = export_onnx(net, str(_CHECKPOINTS / "eval_onnx"))
            print(f"\n{label} checkpoint: gen {gen} ({Path(ckpt_path).parent.name})")
            last_ckpt_path = ckpt_path

        t0 = time.time()

        # Final exam — shared encounters
        if use_mcts:
            fe_wr = eval_mcts(onnx_path, card_vocab_json, monster_json, profiles_json,
                              fe_encounters, fe_decks, 70, fe_seeds)
        else:
            fe_wr = eval_policy(onnx_path, card_vocab_json, monster_json, profiles_json,
                                fe_encounters, fe_decks, 70, fe_seeds)

        # Recorded — shared encounters with frozen HP
        total_wins, total_games = 0, 0
        for i, rec in enumerate(rec_records):
            deck = []
            for cid in rec.get("deck", []):
                try:
                    deck.append(lookup_card(cid.rstrip("+")))
                except Exception:
                    pass
            if not deck:
                continue

            hp = rec.get("calibrated_hp", 70)
            enc = [rec["enemy_ids"]] * recorded_combats
            dks = [deck] * recorded_combats
            seeds = [42 * 1000 + i * 100 + j for j in range(recorded_combats)]

            if use_mcts:
                wr = eval_mcts(onnx_path, card_vocab_json, monster_json, profiles_json,
                               enc, dks, hp, seeds)
            else:
                wr = eval_policy(onnx_path, card_vocab_json, monster_json, profiles_json,
                                 enc, dks, hp, seeds)
            total_wins += int(wr * recorded_combats + 0.5)
            total_games += recorded_combats

        rec_wr = total_wins / max(total_games, 1)

        # Eval scenarios — raw policy quality (skip for PPO+MCTS since it uses search)
        eval_score = None
        if label != "PPO+MCTS":
            from .eval import run_eval
            import io, contextlib
            # Suppress eval's verbose output
            f = io.StringIO()
            with contextlib.redirect_stdout(f):
                eval_result = run_eval(ckpt_path)
            eval_score = eval_result["passed"] / max(eval_result["total"], 1)

        elapsed = time.time() - t0
        results[label] = (fe_wr, rec_wr, eval_score)
        mode = "MCTS" if use_mcts else "policy"
        eval_str = f"  eval={eval_score:5.1%}" if eval_score is not None else "  eval=  N/A"
        print(f"  {label:18s} final={fe_wr:5.1%}  recorded={rec_wr:5.1%}{eval_str}  ({mode}, {elapsed:.0f}s)")

    # --- Summary table ---
    if results:
        print(f"\n{'':20s} {'Final Exam':>10s}  {'Recorded':>10s}  {'Eval':>7s}")
        print("-" * 53)
        for name, (fe, rec, ev) in results.items():
            ev_str = f"{ev:6.1%}" if ev is not None else "   N/A"
            print(f"  {name:18s} {fe:9.1%}  {rec:9.1%}  {ev_str}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark BetaOne checkpoints")
    parser.add_argument("--ppo", default=None, help="Path to PPO checkpoint")
    parser.add_argument("--mcts", default=None, help="Path to MCTS self-play checkpoint")
    parser.add_argument("--combats", type=int, default=256, help="Combats for final exam")
    parser.add_argument("--recorded-combats", type=int, default=32, help="Combats per recorded encounter")
    args = parser.parse_args()

    run_benchmark(
        ppo_checkpoint=args.ppo,
        mcts_checkpoint=args.mcts,
        combats=args.combats,
        recorded_combats=args.recorded_combats,
    )


if __name__ == "__main__":
    main()
