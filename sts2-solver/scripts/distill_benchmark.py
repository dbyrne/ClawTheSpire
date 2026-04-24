"""Tight-CI policy-only benchmark for distill-v1 student vs v3 g88 teacher.

Runs both checkpoints at N repeats × lean-decks-v1 (189 encounters) and
reports WR with 95% CI so we can cleanly see whether the student closed
the policy-MCTS gap.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sts2_solver.betaone.benchmark import benchmark_checkpoint
from sts2_solver.betaone.encounter_set import load_encounter_set


def _ci95(wins: int, n: int) -> tuple[float, float, float]:
    """Wilson 95% CI for win rate."""
    if n == 0:
        return 0.0, 0.0, 0.0
    p = wins / n
    z = 1.96
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    halfw = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return p, max(0.0, center - halfw), min(1.0, center + halfw)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoints", nargs="+", required=True,
                   help="Checkpoint paths to benchmark (labeled by basename)")
    p.add_argument("--encounter-set", default="lean-decks-v1")
    p.add_argument("--repeats", type=int, default=50)
    p.add_argument("--mode", default="policy", choices=["policy", "mcts"])
    p.add_argument("--num-sims", type=int, default=1000,
                   help="MCTS sims (only used when --mode=mcts)")
    args = p.parse_args()

    sys.stdout.reconfigure(encoding="utf-8")
    encounter_set = load_encounter_set(args.encounter_set)
    print(f"Encounter set: {args.encounter_set} ({len(encounter_set)} encounters)")
    print(f"Repeats: {args.repeats} → n = {args.repeats * len(encounter_set)} combats per checkpoint")
    print()

    results = []
    for ckpt in args.checkpoints:
        label = Path(ckpt).parent.name + "/" + Path(ckpt).stem
        print(f"=== {label} ===")
        import time
        t0 = time.time()
        res = benchmark_checkpoint(
            checkpoint_path=ckpt,
            encounter_set=encounter_set,
            mode=args.mode,
            repeats=args.repeats,
            num_sims=args.num_sims if args.mode == "mcts" else 0,
            c_puct=1.5,
            pomcp=True,
            turn_boundary_eval=True,
            pw_k=2.0,
        )
        dt = time.time() - t0
        # benchmark_checkpoint returns list of per-mode results
        r = res[0] if res else None
        if r is None:
            print("  NO RESULT")
            continue
        wins = r["wins"]
        games = r["games"]
        wr, lo, hi = _ci95(wins, games)
        print(f"  {wins}/{games} = {100*wr:.2f}% [95% CI: {100*lo:.2f}%, {100*hi:.2f}%]")
        print(f"  elapsed: {dt/60:.1f} min")
        results.append((label, wins, games, wr, lo, hi))
        print()

    # Comparison
    if len(results) >= 2:
        print("=== COMPARISON ===")
        for label, wins, games, wr, lo, hi in results:
            print(f"  {label:40s}: {100*wr:5.2f}% [{100*lo:5.2f}, {100*hi:5.2f}] n={games}")
        if len(results) == 2:
            a, b = results[0], results[1]
            delta = 100 * (b[3] - a[3])
            # Pooled SE for delta
            pa, pb = a[3], b[3]
            na, nb = a[2], b[2]
            se = 100 * math.sqrt(pa*(1-pa)/na + pb*(1-pb)/nb)
            print(f"  Delta ({b[0]} − {a[0]}): {delta:+.2f}pp (SE ≈ {se:.2f}pp)")
            if abs(delta) > 1.96 * se:
                print(f"  → SIGNIFICANT at p<0.05")
            else:
                print(f"  → within noise")


if __name__ == "__main__":
    main()
