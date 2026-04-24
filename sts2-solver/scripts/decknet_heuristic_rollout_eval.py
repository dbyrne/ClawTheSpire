"""Rollout-based heuristic eval: score candidates via actual MCTS combats.

Same 15-scenario eval as decknet_heuristic_eval.py but uses
`betaone_mcts_fight_combat` FFI to run real combats against probe encounters,
scoring by mean HP preserved (final_hp / max_hp).

Must run from a venv with an sts2_engine wheel matching the checkpoint's
state_dim. For trunk-baseline-v2 use:
    PYTHONPATH=C:/coding-projects/STS2/sts2-solver/src \\
    C:/coding-projects/sts2-trunk-baseline-v2/sts2-solver/.venv/Scripts/python.exe \\
    scripts/decknet_heuristic_rollout_eval.py --checkpoint <path>
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

from sts2_solver.decknet.eval import build_scenarios
from sts2_solver.decknet.heuristic import load_encounter_pool
from sts2_solver.decknet.heuristic_rollout import (
    RolloutConfig,
    prepare_oracle,
    score_candidates_rollout,
)
from sts2_solver.decknet.state import ModKind


def _mod_label(mod) -> str:
    if mod.kind == ModKind.IDENTITY:
        return "SKIP"
    if mod.kind == ModKind.ADD and mod.card is not None:
        return f"ADD {mod.card.id}"
    if mod.kind == ModKind.REMOVE and mod.card is not None:
        return f"REMOVE {mod.card.id}"
    return str(mod.kind.value)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--pool", default=None)
    p.add_argument("--k-next", type=int, default=3)
    p.add_argument("--num-seeds", type=int, default=1)
    p.add_argument("--num-sims", type=int, default=50)
    p.add_argument("--probe-seed", type=int, default=0)
    p.add_argument("--no-boss", action="store_true")
    p.add_argument("--verbose", "-v", action="store_true")
    args = p.parse_args()

    print(f"Loading oracle from: {args.checkpoint}")
    t0 = time.time()
    oracle = prepare_oracle(args.checkpoint)
    print(f"  -> ready in {time.time()-t0:.1f}s, known enemies: {len(oracle.known_enemies)}")

    pool = load_encounter_pool(args.pool)
    print(f"Encounter pool: {len(pool)} entries")

    scenarios = build_scenarios()
    print(f"DeckNet eval: {len(scenarios)} scenarios\n")

    config = RolloutConfig(
        k_next=args.k_next,
        include_boss=not args.no_boss,
        num_seeds=args.num_seeds,
        num_sims=args.num_sims,
        probe_seed=args.probe_seed,
    )
    probes_per_cand = (args.k_next + (0 if args.no_boss else 1)) * args.num_seeds
    print(f"Config: probes/candidate={probes_per_cand}, sims/combat={args.num_sims}\n")

    correct = 0
    avoided_bad = 0
    bad_total = 0
    hit_bad = 0
    t_start = time.time()

    from collections import defaultdict
    by_cat: dict[str, dict] = defaultdict(lambda: {"correct": 0, "total": 0, "bad": 0})

    print(f"{'#':>3}  {'category':<18} {'scenario':<38} {'pick':<26} {'correct':<6}")
    print("-" * 110)

    for i, sc in enumerate(scenarios):
        t0 = time.time()
        scores, details = score_candidates_rollout(
            sc.state, sc.candidates,
            oracle=oracle, pool=pool, config=config, return_detail=True,
        )
        dt = time.time() - t0
        pick = max(range(len(scores)), key=lambda j: scores[j])
        is_correct = pick == sc.best_idx or pick in sc.acceptable_idx
        is_bad = pick in sc.bad_idx
        cat = sc.category
        by_cat[cat]["total"] += 1
        if is_correct:
            correct += 1
            by_cat[cat]["correct"] += 1
        if sc.bad_idx:
            bad_total += 1
            if is_bad:
                hit_bad += 1
                by_cat[cat]["bad"] += 1
            else:
                avoided_bad += 1

        labels = [_mod_label(m) for m in sc.candidates]
        status = "OK  " if is_correct else ("BAD!" if is_bad else "miss")
        print(f"{i:>3}  {cat:<18} {sc.name:<38} {labels[pick][:26]:<26} {status:<6} [{dt:.1f}s]")

        if args.verbose:
            for j, d in enumerate(details):
                marker = "*" if j == sc.best_idx else (" ")
                bad_marker = "!" if j in sc.bad_idx else " "
                print(f"       {marker}{bad_marker} cand[{j}] {labels[j]:<26} "
                      f"score={scores[j]:.3f}  wr={d['wins']}/{d['trials']}  hp={d['mean_hp_frac']:.2f}")

    n = len(scenarios)
    print("\n" + "=" * 80)
    print(f"Overall: {correct}/{n} ({100*correct/n:.1f}%)")
    if bad_total:
        print(f"Avoided bad picks: {avoided_bad}/{bad_total} ({100*avoided_bad/bad_total:.0f}%)")
    print(f"Total time: {time.time()-t_start:.1f}s")
    print("Baseline (15-scn forward-pass): untrained=67%, forward-pass=47%, rollout=67%")
    print()
    print(f"{'Category':<20} {'Acc':<10} {'n':>4}  {'Bad':>3}")
    print("-" * 50)
    for cat, stats in sorted(by_cat.items()):
        tot = stats["total"]
        pct = 100 * stats["correct"] / tot if tot else 0.0
        bad = stats["bad"]
        bad_str = f"{bad}" if bad else ""
        print(f"{cat:<20} {stats['correct']}/{tot} ({pct:.0f}%)  {tot:>4}  {bad_str:>3}")


if __name__ == "__main__":
    main()
