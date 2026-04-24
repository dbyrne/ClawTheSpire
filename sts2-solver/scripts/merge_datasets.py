"""Merge two distillation dataset.pkl files into one.

Used for DAgger iteration k+1: combine existing distilled dataset
(teacher-trajectory states) with new DAgger dataset (student-trajectory
states + teacher labels). This augments rather than replaces — classic
Ross 2011 DAgger aggregates datasets across iterations.

Usage:
    python -m scripts.merge_datasets \\
        --inputs experiments/distill-v1/dataset.pkl \\
                 experiments/distill-dagger-v2/dataset.pkl \\
        --output experiments/distill-dagger-v2/dataset_merged.pkl
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", nargs="+", required=True, help="Dataset pkl paths to merge")
    p.add_argument("--output", required=True)
    args = p.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    keys = ["states", "action_features", "action_masks",
            "hand_card_ids", "action_card_ids",
            "target_policies", "target_values"]

    merged = {k: [] for k in keys}
    metas = []
    total = 0
    for path in args.inputs:
        with open(path, "rb") as f:
            d = pickle.load(f)
        n = len(d["states"])
        print(f"  {Path(path).parent.name}: {n} states")
        total += n
        for k in keys:
            merged[k].append(d[k])
        metas.append(d.get("meta", {}))

    for k in keys:
        merged[k] = np.concatenate(merged[k], axis=0)

    merged["meta"] = {
        "merged_from": [str(p) for p in args.inputs],
        "n_states": total,
        "component_metas": metas,
    }

    with open(args.output, "wb") as f:
        pickle.dump(merged, f, protocol=pickle.HIGHEST_PROTOCOL)
    size_mb = Path(args.output).stat().st_size / (1024 * 1024)
    print(f"\nWrote {total} merged states to {args.output} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
