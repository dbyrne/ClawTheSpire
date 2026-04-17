"""Benchmark BetaOne checkpoints against encounter sets.

One code path: load encounter set, run each encounter N times, report WR with CI.

Usage:
    python -m sts2_solver.betaone.benchmark --checkpoint path/to.pt --encounter-set name
"""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

import sts2_engine

from .data_utils import build_monster_data_json, load_solver_json
from .network import (
    BetaOneNetwork, export_onnx, load_checkpoint, network_kwargs_from_meta,
)
from .paths import SOLVER_ROOT, SOLVER_PKG, BENCHMARK_DIR
from .deck_gen import lookup_card


_CHECKPOINTS = SOLVER_ROOT.parent / "betaone_checkpoints"


def _build_card_vocab() -> tuple[dict, str]:
    for candidate in [BENCHMARK_DIR / "card_vocab.json", _CHECKPOINTS / "card_vocab.json"]:
        if candidate.exists():
            with open(candidate, encoding="utf-8") as f:
                vocab = json.load(f)
            return vocab, json.dumps(vocab)
    raise FileNotFoundError("card_vocab.json not found")


def _load_checkpoint(path: str) -> tuple[BetaOneNetwork, dict]:
    card_vocab, _ = _build_card_vocab()
    # Peek at arch_meta first so we construct a network matching the
    # checkpoint's architecture (value_head_layers may differ between
    # experiments now that depth is tunable).
    import torch
    ckpt_peek = torch.load(path, map_location="cpu", weights_only=False)
    kwargs = network_kwargs_from_meta(ckpt_peek.get("arch_meta"))
    net = BetaOneNetwork(num_cards=len(card_vocab), **kwargs)
    ckpt = load_checkpoint(path, network=net, strict=False)
    net.eval()
    return net, ckpt


def _eval_batch(
    onnx_path: str, card_vocab_json: str,
    monster_json: str, profiles_json: str,
    encounters: list, decks: list, player_hp: int,
    seeds: list[int], use_mcts: bool, num_sims: int,
    c_puct: float = 2.5,
    pomcp: bool = False,
    turn_boundary_eval: bool = False,
    pw_k: float = 1.0,
    batch_size_mcts: int = 1,
) -> int:
    """Run a batch of combats at a single HP level. Returns win count."""
    if use_mcts:
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
            c_puct=c_puct,
            pomcp=pomcp,
            turn_boundary_eval=turn_boundary_eval,
            pw_k=pw_k,
            batch_size=batch_size_mcts,
        )
    else:
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
    return sum(1 for o in r["outcomes"] if o == "win")


def benchmark_checkpoint(
    checkpoint_path: str,
    encounter_set: list[dict],
    mode: str = "both",
    repeats: int = 1,
    num_sims: int = 400,
    c_puct: float = 2.5,
    pomcp: bool = False,
    turn_boundary_eval: bool = False,
    pw_k: float = 1.0,
    batch_size_mcts: int = 1,
) -> list[dict]:
    """Benchmark a checkpoint against an encounter set.

    Args:
        checkpoint_path: Path to .pt checkpoint.
        encounter_set: List of {enemies, deck, hp, relics} dicts.
        mode: "policy", "mcts", or "both".
        repeats: Times to repeat each encounter (for statistical power).
        num_sims: MCTS simulations per decision.
        c_puct, pomcp, turn_boundary_eval, pw_k: MCTS inference config.
            Should match the model's training config — defaulting to stock
            MCTS settings here would silently change inference semantics
            from what the model was optimized for.

    Returns:
        List of result dicts, one per mode.
    """
    card_vocab, card_vocab_json = _build_card_vocab()
    monster_json = build_monster_data_json()
    profiles_json = load_solver_json("enemy_profiles.json")

    net, ckpt = _load_checkpoint(checkpoint_path)
    gen = ckpt.get("gen", "?")
    onnx_path = export_onnx(net, str(_CHECKPOINTS / "eval_onnx"))

    modes = []
    if mode in ("policy", "both"):
        modes.append("policy")
    if mode in ("mcts", "both"):
        modes.append("mcts")

    results = []
    for m in modes:
        use_mcts = m == "mcts"
        t0 = time.time()

        # Group encounters by HP for batched execution
        from collections import defaultdict
        hp_groups: dict[int, list[tuple[int, dict]]] = defaultdict(list)
        for i, enc in enumerate(encounter_set):
            hp = enc.get("hp", enc.get("calibrated_hp", 70))
            hp_groups[hp].append((i, enc))

        wins, n_games = 0, 0
        for hp, enc_list in hp_groups.items():
            batch_enc, batch_dks, batch_seeds = [], [], []
            for i, enc in enc_list:
                deck = enc.get("deck", [])
                if deck and isinstance(deck[0], str):
                    deck = [lookup_card(cid.rstrip("+")) for cid in deck
                            if lookup_card(cid.rstrip("+")) is not None]
                if not deck:
                    continue
                enemies = enc.get("enemies", enc.get("enemy_ids", []))
                for j in range(repeats):
                    batch_enc.append(enemies)
                    batch_dks.append(deck)
                    batch_seeds.append(42 * 1000 + i * 100 + j)

            if not batch_enc:
                continue

            batch_wins = _eval_batch(
                onnx_path, card_vocab_json, monster_json, profiles_json,
                batch_enc, batch_dks, hp, batch_seeds, use_mcts, num_sims,
                c_puct=c_puct, pomcp=pomcp,
                turn_boundary_eval=turn_boundary_eval, pw_k=pw_k,
                batch_size_mcts=batch_size_mcts,
            )
            wins += batch_wins
            n_games += len(batch_enc)

        wr = wins / max(n_games, 1)

        # Wilson 95% CI
        z = 1.96
        n = n_games
        p_hat = wins / max(n, 1)
        denom = 1 + z * z / n
        center = (p_hat + z * z / (2 * n)) / denom
        margin = z * math.sqrt((p_hat * (1 - p_hat) + z * z / (4 * n)) / n) / denom
        ci_lo = max(0, center - margin)
        ci_hi = min(1, center + margin)

        elapsed = time.time() - t0
        print(f"  {m:<10s} win_rate={wr:5.1%}  95%CI=[{ci_lo:.1%},{ci_hi:.1%}]  n={n_games}  ({elapsed:.0f}s)")

        results.append({
            "mode": m,
            "gen": gen,
            "win_rate": round(wr, 4),
            "wins": wins,
            "games": n_games,
            "ci95_lo": round(ci_lo, 4),
            "ci95_hi": round(ci_hi, 4),
            "mcts_sims": num_sims if use_mcts else 0,
            "pw_k": pw_k if use_mcts else None,
            "c_puct": c_puct if use_mcts else None,
            "pomcp": pomcp if use_mcts else None,
            "turn_boundary_eval": turn_boundary_eval if use_mcts else None,
            "batch_size_mcts": batch_size_mcts if use_mcts else None,
            "elapsed": round(elapsed, 1),
        })

    return results
