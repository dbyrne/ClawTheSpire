"""Build gamma-train-v1: lean-decks-v2 (latest packages) + sampled uber-decks-v1
(diversity) with potions added to ~50% of encounters.

Avoids re-calibrating HP for a new generation; reuses HPs from the source sets
which were calibrated against reanalyse-v3 g88 (well-understood baseline).
Each source encounter keeps its original enemies/deck/hp/relics; we only add
a sampled `potions` field.
"""
from __future__ import annotations

import argparse
import random
import sys

from sts2_solver.betaone.encounter_set import load_encounter_set, save_encounter_set
from sts2_solver.betaone.potions import sample_potions


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="gamma-train-v1")
    p.add_argument("--lean-v2-take", type=int, default=-1,
                   help="How many lean-decks-v2 encounters to take (-1 = all).")
    p.add_argument("--uber-sample", type=int, default=250,
                   help="How many uber-decks-v1 encounters to sample (0 = none).")
    p.add_argument("--potion-rate", type=float, default=0.5)
    p.add_argument("--potion-max", type=int, default=2)
    p.add_argument("--seed", type=int, default=12345)
    args = p.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    rng = random.Random(args.seed)

    lean_v2 = load_encounter_set("lean-decks-v2")
    uber_v1 = load_encounter_set("uber-decks-v1")

    if args.lean_v2_take >= 0 and args.lean_v2_take < len(lean_v2):
        lean_take = rng.sample(lean_v2, args.lean_v2_take)
    else:
        lean_take = list(lean_v2)

    if args.uber_sample > 0:
        uber_take = rng.sample(uber_v1, min(args.uber_sample, len(uber_v1)))
    else:
        uber_take = []

    print(f"Source encounters: lean-decks-v2 {len(lean_take)} + uber-decks-v1 {len(uber_take)}")
    print(f"Potion rate: {args.potion_rate}, max per encounter: {args.potion_max}")

    merged = []
    potion_counts = [0, 0, 0]  # count of encounters with 0 / 1 / 2+ potions
    for enc in lean_take + uber_take:
        new_enc = dict(enc)
        potions = sample_potions(rng, rate=args.potion_rate, max_count=args.potion_max)
        new_enc["potions"] = potions
        n = len(potions)
        potion_counts[min(n, 2)] += 1
        merged.append(new_enc)

    rng.shuffle(merged)

    print(f"Total encounters: {len(merged)}")
    print(f"  with 0 potions: {potion_counts[0]} ({100*potion_counts[0]/len(merged):.0f}%)")
    print(f"  with 1 potion:  {potion_counts[1]} ({100*potion_counts[1]/len(merged):.0f}%)")
    print(f"  with 2+ potions: {potion_counts[2]} ({100*potion_counts[2]/len(merged):.0f}%)")

    save_encounter_set(
        name=args.name,
        encounters=merged,
        source={
            "type": "merged_with_potions",
            "base_sets": ["lean-decks-v2", "uber-decks-v1"],
            "lean_v2_taken": len(lean_take),
            "uber_sampled": len(uber_take),
            "potion_rate": args.potion_rate,
            "potion_max_count": args.potion_max,
            "rng_seed": args.seed,
        },
    )
    print(f"\nEncounter set saved: {args.name}")


if __name__ == "__main__":
    main()
