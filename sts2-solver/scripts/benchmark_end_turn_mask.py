"""A/B combat benchmark: mask EndTurn from candidate actions vs. don't.

Tests user's hypothesis that v3 g88 would win more on lean-decks-v1 if
EndTurn were excluded from the candidate set whenever any other action is
playable. Engine change: `enumerate_actions` skips pushing Action::EndTurn
when the new MASK_END_TURN atomic is set. The atomic is plumbed through
`betaone_mcts_selfplay`/`betaone_mcts_fight_combat` as a per-call flag.

Auto-end-turn semantics are preserved: when no cards/potions are playable,
the caller's empty-actions branch ends the turn without a network decision.

Usage:
    PYTHONIOENCODING=utf-8 python scripts/benchmark_end_turn_mask.py \\
        --checkpoint C:/coding-projects/sts2-reanalyse-v3/sts2-solver/experiments/reanalyse-v3/betaone_gen88.pt \\
        --encounter-set lean-decks-v1 \\
        --num-sims 1000 --repeats 2

Reports both win rates with Wilson CIs and the delta. Same seeds across
the A and B runs so encounter/RNG variance cancels and only the EndTurn
masking is being measured.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import sts2_engine

# Ensure printing emoji/arrows on Windows cp1252 doesn't crash
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from sts2_solver.betaone.benchmark import _build_card_vocab, _load_checkpoint
from sts2_solver.betaone.data_utils import build_monster_data_json, load_solver_json
from sts2_solver.betaone.deck_gen import lookup_card
from sts2_solver.betaone.network import export_onnx
from sts2_solver.betaone.paths import SOLVER_ROOT, BENCHMARK_DIR


def _wilson_ci(wins: int, n: int) -> tuple[float, float, float]:
    if n <= 0:
        return 0.0, 0.0, 0.0
    z = 1.96
    p = wins / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
    return p, max(0.0, center - margin), min(1.0, center + margin)


def _resolve_encounter_set(name_or_path: str) -> Path:
    p = Path(name_or_path)
    if p.exists():
        return p
    candidate = BENCHMARK_DIR / "encounter_sets" / f"{name_or_path}.jsonl"
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"Encounter set not found: {name_or_path}")


def _load_encounters(path: Path) -> list[dict]:
    encs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                encs.append(json.loads(line))
    return encs


def _resolve_decks(encounters: list[dict]) -> list[list[dict]]:
    """Encounter decks may be card-id strings; resolve to full Card dicts."""
    decks = []
    for enc in encounters:
        deck = enc.get("deck", [])
        if deck and isinstance(deck[0], str):
            deck = [
                lookup_card(cid.rstrip("+"))
                for cid in deck
                if lookup_card(cid.rstrip("+")) is not None
            ]
        decks.append(deck)
    return decks


def run_batch(
    onnx_path: str,
    card_vocab_json: str,
    monster_json: str,
    profiles_json: str,
    encounters: list[dict],
    decks: list[list[dict]],
    num_sims: int,
    repeats: int,
    mask_end_turn: bool,
    deterministic: bool = False,
) -> tuple[int, int, float, dict[tuple[int, int], int]]:
    """Run all encounters at their natural HP buckets, repeats per encounter.
    Returns (wins, games, elapsed_s, per_game_outcomes).
    per_game_outcomes maps (encounter_idx, repeat_idx) -> 1 if win, 0 if loss.
    Used by the paired-analysis step in main(): with the same seed schema
    across mask=False and mask=True runs, identical (enc_idx, repeat_idx)
    pairs hit the same RNG path so any outcome flip is attributable to the
    masking change rather than encounter/RNG variance.
    """
    from collections import defaultdict

    hp_groups: dict[int, list[int]] = defaultdict(list)
    for i, enc in enumerate(encounters):
        hp = enc.get("hp", enc.get("calibrated_hp", 70))
        hp_groups[hp].append(i)

    t0 = time.time()
    wins = 0
    n_games = 0
    per_game: dict[tuple[int, int], int] = {}
    for hp, idxs in hp_groups.items():
        batch_enc, batch_dks, batch_seeds, batch_keys = [], [], [], []
        for i in idxs:
            deck = decks[i]
            if not deck:
                continue
            enemies = encounters[i].get("enemies", encounters[i].get("enemy_ids", []))
            for j in range(repeats):
                batch_enc.append(enemies)
                batch_dks.append(deck)
                # Same seeds across A and B runs so RNG/encounter variance
                # cancels — only the mask flag differs between them.
                batch_seeds.append(42 * 1000 + i * 100 + j)
                batch_keys.append((i, j))

        if not batch_enc:
            continue

        r = sts2_engine.betaone_mcts_selfplay(
            encounters_json=json.dumps(batch_enc),
            decks_json=json.dumps(batch_dks),
            player_hp=hp,
            player_max_hp=70,
            player_max_energy=3,
            relics_json="[]",
            potions_json="[]",
            monster_data_json=monster_json,
            enemy_profiles_json=profiles_json,
            onnx_path=onnx_path,
            card_vocab_json=card_vocab_json,
            num_sims=num_sims,
            temperature=0.0,
            seeds=batch_seeds,
            add_noise=False,
            c_puct=2.5,
            pomcp=False,
            turn_boundary_eval=False,
            pw_k=1.0,
            mask_end_turn=mask_end_turn,
            deterministic=deterministic,
        )
        for key, outcome in zip(batch_keys, r["outcomes"]):
            per_game[key] = 1 if outcome == "win" else 0
        wins += sum(1 for o in r["outcomes"] if o == "win")
        n_games += len(batch_enc)
        # Light progress trace per HP bucket
        sub_wins = sum(1 for o in r["outcomes"] if o == "win")
        sub_n = len(batch_enc)
        sub_wr = sub_wins / max(sub_n, 1)
        tag = "MASKED" if mask_end_turn else "BASE  "
        print(
            f"  [{tag}] hp={hp:>2} bucket: {sub_wins}/{sub_n} = {sub_wr*100:.1f}% "
            f"(running {wins}/{n_games} = {wins/max(n_games,1)*100:.1f}%)",
            flush=True,
        )

    return wins, n_games, time.time() - t0, per_game


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--encounter-set", default="lean-decks-v1")
    p.add_argument("--num-sims", type=int, default=1000)
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument(
        "--noise-floor", action="store_true",
        help="Run BASE in both arms (mask flag never set). Same seeds, same "
             "engine config — discordant pairs measure MCTS determinism / "
             "noise floor, not the mask effect."
    )
    p.add_argument(
        "--deterministic", action="store_true",
        help="Force ORT deterministic-compute kernels + sequential inter-op "
             "execution. Eliminates ONNX run-to-run float variance that "
             "produces ~15%% paired-discordance noise floor at intra=1. "
             "Slower per-combat; use for benchmark precision, not training."
    )
    args = p.parse_args()

    print(f"Loading checkpoint: {args.checkpoint}")
    net, ckpt = _load_checkpoint(args.checkpoint)
    gen = ckpt.get("gen", "?")
    card_vocab, card_vocab_json = _build_card_vocab()
    monster_json = build_monster_data_json()
    profiles_json = load_solver_json("enemy_profiles.json")
    onnx_path = export_onnx(net, str(SOLVER_ROOT.parent / "betaone_checkpoints" / "eval_onnx"))
    print(f"  gen={gen}, params={sum(p.numel() for p in net.parameters()):,}")

    enc_path = _resolve_encounter_set(args.encounter_set)
    encounters = _load_encounters(enc_path)
    decks = _resolve_decks(encounters)
    print(f"Encounter set: {enc_path.name}  ({len(encounters)} encounters x {args.repeats} repeats)")
    print(f"MCTS sims: {args.num_sims}\n")

    arm_b_mask = False if args.noise_floor else True
    arm_b_label = "BASE (noise-floor)" if args.noise_floor else "MASKED"
    arm_b_desc = (
        "BASE again, identical seeds — measures MCTS determinism / search noise"
        if args.noise_floor
        else "EndTurn excluded when any other action is valid"
    )

    print("Run A: BASE (EndTurn always offered when other plays exist)")
    base_wins, base_n, base_t, base_per_game = run_batch(
        onnx_path, card_vocab_json, monster_json, profiles_json,
        encounters, decks, args.num_sims, args.repeats,
        mask_end_turn=False, deterministic=args.deterministic,
    )
    base_wr, base_lo, base_hi = _wilson_ci(base_wins, base_n)
    print(
        f"\n  BASE   : {base_wins}/{base_n} = {base_wr*100:.2f}% "
        f"(95% CI {base_lo*100:.2f}-{base_hi*100:.2f})  [{base_t:.0f}s]\n"
    )

    print(f"Run B: {arm_b_label} ({arm_b_desc})")
    mask_wins, mask_n, mask_t, mask_per_game = run_batch(
        onnx_path, card_vocab_json, monster_json, profiles_json,
        encounters, decks, args.num_sims, args.repeats,
        mask_end_turn=arm_b_mask, deterministic=args.deterministic,
    )
    mask_wr, mask_lo, mask_hi = _wilson_ci(mask_wins, mask_n)
    print(
        f"\n  MASKED : {mask_wins}/{mask_n} = {mask_wr*100:.2f}% "
        f"(95% CI {mask_lo*100:.2f}-{mask_hi*100:.2f})  [{mask_t:.0f}s]\n"
    )

    delta = mask_wr - base_wr
    arrow = "UP" if delta > 0 else ("DOWN" if delta < 0 else "FLAT")
    print(f"Aggregate Δ: {delta*100:+.2f}pp ({arrow})")
    print(
        f"  base = {base_wr*100:.2f}% [{base_lo*100:.2f}, {base_hi*100:.2f}]\n"
        f"  mask = {mask_wr*100:.2f}% [{mask_lo*100:.2f}, {mask_hi*100:.2f}]"
    )
    if base_hi < mask_lo or mask_hi < base_lo:
        print("  Wilson CIs disjoint.")
    else:
        print("  Wilson CIs overlap (independent-sample assumption).")

    # Paired analysis: same seed pair -> identical RNG/encounter, so any
    # outcome flip is attributable to the mask flag. McNemar's test on the
    # discordant-pair counts is the right power lens here, not naive
    # overlap-of-CIs (which assumes independent samples and discards the
    # huge variance reduction from same-seed pairing).
    keys = sorted(set(base_per_game) & set(mask_per_game))
    n_paired = len(keys)
    both_win = both_lose = base_only_win = mask_only_win = 0
    for k in keys:
        b, m = base_per_game[k], mask_per_game[k]
        if b == 1 and m == 1:
            both_win += 1
        elif b == 0 and m == 0:
            both_lose += 1
        elif b == 1 and m == 0:
            base_only_win += 1
        else:
            mask_only_win += 1

    discordant = base_only_win + mask_only_win
    net_flip = mask_only_win - base_only_win  # positive => mask helps
    paired_delta = net_flip / max(n_paired, 1)

    # Wilson CI for the discordant-flip proportion using McNemar framing.
    # Standard error of net_flip/n_paired under H0 is sqrt(discordant)/n
    # for the difference in proportions; report that as a sanity bound.
    if n_paired > 0 and discordant > 0:
        se_diff = math.sqrt(discordant) / n_paired
        delta_lo = paired_delta - 1.96 * se_diff
        delta_hi = paired_delta + 1.96 * se_diff
    else:
        delta_lo = delta_hi = paired_delta

    print()
    print(f"Paired analysis (same seed across arms, n={n_paired}):")
    print(f"  Both win        : {both_win}")
    print(f"  Both lose       : {both_lose}")
    print(f"  Base wins only  : {base_only_win}  (mask hurt)")
    print(f"  Mask wins only  : {mask_only_win}  (mask helped)")
    print(f"  Discordant pairs: {discordant} ({100*discordant/max(n_paired,1):.1f}%)")
    print(
        f"  Paired Δ        : {paired_delta*100:+.2f}pp  "
        f"(95% CI {delta_lo*100:+.2f}, {delta_hi*100:+.2f})"
    )

    # Two-sided exact-binomial p (McNemar): under H0 each discordant pair
    # is 50/50. P(at least |net_flip| in either direction).
    if discordant > 0:
        from math import comb
        k = max(base_only_win, mask_only_win)
        # Sum tail probs from k..discordant (or all-zero..min) symmetric.
        tail = sum(comb(discordant, i) for i in range(k, discordant + 1))
        p = 2 * tail / (2 ** discordant)
        p = min(p, 1.0)
        print(f"  McNemar p-value : {p:.4f}")
    else:
        print(f"  McNemar p-value : 1.0000  (no discordant pairs)")


if __name__ == "__main__":
    main()
