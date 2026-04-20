"""Append more gap-filler encounters to uber-decks-v1.jsonl without rerunning the base.

Use after build_uber_decks.py to add combo-pair specs for cards that came out
under-covered. Only generates new decks + calibrates + appends — preserves the
existing 487 calibrated encounters.

Usage:
    PYTHONIOENCODING=utf-8 python scripts/extend_uber_decks.py \\
        --target experiments/_benchmark/encounter_sets/uber-decks-v1.jsonl \\
        --checkpoint C:/coding-projects/sts2-reanalyse-v3/sts2-solver/experiments/reanalyse-v3/betaone_gen88.pt
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

from build_uber_decks import (
    EVAL_REQUIRED_CARDS, STATUS_CARDS,
    load_encounter_set, load_game_data, load_card_vocab,
    export_ckpt_onnx, calibrate_hp,
)

from sts2_solver.betaone.deck_gen import build_random_deck


# Additional gap-filler specs targeting cards still under 20% in v1:
# - Combo pairs for failing P-Eval scenarios
# - Individual top-ups for cards under 15%
# Top-up round (after first extend): D&R diluted back to 7.1%, DAGGER_SPRAY 8.6%.
# Both are common STS cards that'll appear in real decks; bump coverage.
EXTEND_SPECS = [
    (["DODGE_AND_ROLL"],    80, "dodge_topup"),
    (["DAGGER_SPRAY"],      80, "spray_topup"),
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--target", required=True, type=Path,
                   help="Existing uber-decks jsonl to append to")
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--seed", type=int, default=44)
    p.add_argument("--sims", type=int, default=50)
    p.add_argument("--combats", type=int, default=32)
    args = p.parse_args()

    rng = random.Random(args.seed)

    # Load existing for enemy/relic pool
    existing = load_encounter_set(args.target)
    print(f"  existing: {len(existing)} encounters")
    enemy_pool = [tuple(e.get("enemies", [])) for e in existing if e.get("enemies")]
    relic_pool = [tuple(sorted(e.get("relics", []))) for e in existing]

    # Generate new gap-fillers
    gap_encounters = []
    for core_cards, count, label in EXTEND_SPECS:
        for _ in range(count):
            deck = build_random_deck(rng=rng, core_cards=core_cards)
            enemies = list(rng.choice(enemy_pool))
            relics = list(rng.choice(relic_pool))
            gap_encounters.append({
                "enemies": enemies, "deck": deck, "hp": 50,
                "relics": relics, "_gap_label": label,
            })
    print(f"  generated: {len(gap_encounters)} new gap-filler decks")

    # Export ONNX
    print(f"  exporting ONNX from {args.checkpoint.name}...")
    onnx_dir = Path(tempfile.gettempdir()) / "uber_decks_onnx"
    onnx_path = export_ckpt_onnx(args.checkpoint, onnx_dir)

    monster_json, profiles_json = load_game_data()
    _, card_vocab_json = load_card_vocab(args.checkpoint)

    # Calibrate
    calibrated = []
    print(f"  calibrating {len(gap_encounters)} encounters...")
    t0 = time.time()
    for i, enc in enumerate(gap_encounters):
        hp = calibrate_hp(
            enemies=enc["enemies"], deck=enc["deck"], relics=enc.get("relics", []),
            monster_json=monster_json, profiles_json=profiles_json,
            card_vocab_json=card_vocab_json, onnx_path=onnx_path,
            sims=args.sims, combats=args.combats,
        )
        if hp is None:
            continue
        calibrated.append({
            "enemies": enc["enemies"], "deck": enc["deck"],
            "hp": hp, "relics": enc.get("relics", []),
        })
        elapsed = time.time() - t0
        rate = (i + 1) / elapsed
        eta = (len(gap_encounters) - (i + 1)) / max(rate, 0.001)
        print(f"    [{i+1}/{len(gap_encounters)}] hp={hp}  "
              f"{len(calibrated)}/{i+1} kept  "
              f"{elapsed/60:.1f}m  ETA {eta/60:.1f}m",
              flush=True)

    # Append to existing file
    with open(args.target, "a", encoding="utf-8") as f:
        for enc in calibrated:
            f.write(json.dumps(enc) + "\n")
    new_total = len(existing) + len(calibrated)
    print(f"\n  appended {len(calibrated)} → {new_total} total encounters in {args.target.name}")

    # Re-audit
    all_encs = existing + calibrated
    card_counts = Counter()
    for enc in all_encs:
        ids = {c.get("id") if isinstance(c, dict) else c for c in enc.get("deck", [])}
        for cid in ids:
            card_counts[cid] += 1

    n = new_total
    audit_path = args.target.with_suffix(".audit.txt")
    lines = [f"Coverage audit: {args.target.name}", f"n = {n} encounters (post-extend)", ""]
    lines.append(f"{'card':25} | {'count':>6} | {'pct':>6}")
    lines.append("-" * 50)
    gap_count = 0
    for card in sorted(EVAL_REQUIRED_CARDS - STATUS_CARDS):
        c = card_counts.get(card, 0)
        pct = c / n * 100
        flag = " ← GAP" if pct < 20 else ""
        if pct < 20:
            gap_count += 1
        lines.append(f"{card:25} | {c:>6} | {pct:>5.1f}%{flag}")
    lines.append("")
    lines.append(f"gaps (<20% coverage): {gap_count} of {len(EVAL_REQUIRED_CARDS - STATUS_CARDS)}")
    audit_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  audit → {audit_path}")
    print()
    print("\n".join(lines[-5:]))


if __name__ == "__main__":
    main()
