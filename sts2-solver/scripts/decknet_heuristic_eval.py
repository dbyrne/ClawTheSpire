"""Evaluate the combat-net-oracle heuristic against the 15-scenario DeckNet eval.

Hypothesis: a well-trained combat net already encodes deck-quality understanding.
If we query it properly (score candidate decks via probe-encounter rollouts),
we should beat the 67% untrained baseline without training anything.

Usage:
    python scripts/decknet_heuristic_eval.py \\
        --checkpoint C:/coding-projects/sts2-trunk-baseline-v2/sts2-solver/experiments/trunk-baseline-v2/betaone_latest.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

from sts2_solver.decknet.eval import build_scenarios
from sts2_solver.decknet.heuristic import (
    ProbeConfig,
    load_combat_net,
    load_encounter_pool,
    score_candidates,
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
    p.add_argument("--checkpoint", required=True, help="Path to betaone_latest.pt")
    p.add_argument("--pool", default=None, help="Override encounter pool path")
    p.add_argument("--k-next", type=int, default=3, help="Next-K probe encounters")
    p.add_argument("--num-draws", type=int, default=3, help="Random draws per probe")
    p.add_argument("--probe-seed", type=int, default=0, help="Probe selection seed")
    p.add_argument("--device", default="cpu")
    p.add_argument("--no-boss", action="store_true", help="Skip boss proxy probe")
    p.add_argument("--current-hp", action="store_true",
                   help="Use scenario.player.hp instead of max_hp in probes")
    p.add_argument("--verbose", "-v", action="store_true")
    p.add_argument("--untrained", action="store_true",
                   help="Use random-init network (vocab + arch from checkpoint)")
    args = p.parse_args()

    print(f"Loading combat-net from: {args.checkpoint}"
          + (" [UNTRAINED]" if args.untrained else ""))
    net, card_vocab = load_combat_net(args.checkpoint, device=args.device,
                                       untrained=args.untrained)
    print(f"  -> {sum(q.numel() for q in net.parameters())} params, "
          f"vocab size {len(card_vocab)}")

    pool = load_encounter_pool(args.pool)
    print(f"Encounter pool: {len(pool)} entries, "
          f"floor range [{min(e['floor'] for e in pool)}, "
          f"{max(e['floor'] for e in pool)}]")

    scenarios = build_scenarios()
    print(f"DeckNet eval: {len(scenarios)} scenarios\n")

    config = ProbeConfig(
        k_next=args.k_next,
        include_boss=not args.no_boss,
        num_draws=args.num_draws,
        probe_seed=args.probe_seed,
        use_max_hp=not args.current_hp,
    )

    correct = 0
    avoided_bad = 0
    bad_total = 0
    hit_bad = 0

    from collections import defaultdict
    by_cat: dict[str, dict] = defaultdict(lambda: {"correct": 0, "total": 0, "bad": 0})

    print(f"{'#':>3}  {'category':<18} {'scenario':<40} {'pick':<24} {'correct':<6}")
    print("-" * 110)

    for i, sc in enumerate(scenarios):
        scores, details = score_candidates(
            sc.state, sc.candidates,
            net=net, card_vocab=card_vocab, pool=pool,
            config=config, device=args.device, return_detail=True,
        )
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
        print(f"{i:>3}  {cat:<18} {sc.name:<40} {labels[pick][:24]:<24} {status:<6}")

        if args.verbose:
            for j, d in enumerate(details):
                marker = "*" if j == sc.best_idx else " "
                bad_marker = "!" if j in sc.bad_idx else " "
                print(f"       {marker}{bad_marker} cand[{j}] {labels[j]:<22} "
                      f"v={scores[j]:+.3f}  std={d['std']:.3f}")

    n = len(scenarios)
    print("\n" + "=" * 80)
    print(f"Overall: {correct}/{n} ({100*correct/n:.1f}%)")
    if bad_total:
        print(f"Avoided bad picks: {avoided_bad}/{bad_total} "
              f"({100*avoided_bad/bad_total:.0f}%)")
    print()
    print(f"{'Category':<20} {'Acc':<10} {'n':>4}  {'Bad':>3}")
    print("-" * 50)
    for cat, stats in sorted(by_cat.items()):
        tot = stats["total"]
        pct = 100 * stats["correct"] / tot if tot else 0.0
        bad_str = f"{stats['bad']}" if stats['bad'] else ""
        print(f"{cat:<20} {stats['correct']}/{tot} ({pct:.0f}%)  {tot:>4}  {bad_str:>3}")


if __name__ == "__main__":
    main()
