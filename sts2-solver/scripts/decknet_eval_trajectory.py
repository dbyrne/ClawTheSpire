"""Evaluate a DeckNet experiment across saved checkpoints on the 74-scn suite.

Reports per-gen overall accuracy + per-category accuracy for the final gen.
Use: python scripts/decknet_eval_trajectory.py --experiment decknet-rootval-trunkv2g60
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

import torch

from sts2_solver.decknet.encoder import encode_batch
from sts2_solver.decknet.eval import build_scenarios
from sts2_solver.decknet.network import DeckNet
from sts2_solver.decknet.state import apply_mod


def eval_checkpoint(ckpt_path: Path, card_vocab: dict, scenarios) -> dict:
    net = DeckNet(num_cards=len(card_vocab))
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    net.load_state_dict(ckpt["model_state_dict"])
    net.eval()

    by_cat = defaultdict(lambda: {"correct": 0, "total": 0, "bad": 0})
    correct = 0
    bad_picks = 0
    bad_total = 0

    with torch.no_grad():
        for sc in scenarios:
            candidate_states = [apply_mod(sc.state, m) for m in sc.candidates]
            batch = encode_batch(candidate_states, card_vocab)
            values = net(
                batch["card_ids"], batch["card_stats"],
                batch["deck_mask"], batch["global_state"],
            )
            pick = int(torch.argmax(values).item())
            is_correct = pick == sc.best_idx or pick in sc.acceptable_idx
            is_bad = pick in sc.bad_idx

            by_cat[sc.category]["total"] += 1
            if is_correct:
                correct += 1
                by_cat[sc.category]["correct"] += 1
            if sc.bad_idx:
                bad_total += 1
                if is_bad:
                    bad_picks += 1
                    by_cat[sc.category]["bad"] += 1

    return {
        "correct": correct,
        "total": len(scenarios),
        "accuracy": correct / len(scenarios),
        "bad_picks": bad_picks,
        "bad_total": bad_total,
        "by_category": dict(by_cat),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--experiment", required=True,
                   help="Experiment name (under experiments/)")
    p.add_argument("--gens", type=int, nargs="+",
                   help="Specific gens to eval (default: all saved)")
    args = p.parse_args()

    exp_dir = Path("experiments") / args.experiment
    vocab_path = exp_dir / "card_vocab.json"
    if not vocab_path.exists():
        vocab_path = Path("src/sts2_solver/card_vocab.json")
    card_vocab = json.loads(vocab_path.read_text())

    scenarios = build_scenarios()
    print(f"Eval suite: {len(scenarios)} scenarios")

    if args.gens:
        gens = args.gens
    else:
        gens = sorted(
            int(p.stem.replace("decknet_gen", ""))
            for p in exp_dir.glob("decknet_gen*.pt")
        )

    print(f"Gens to eval: {gens}\n")

    print(f"{'Gen':<6} {'Acc':<18} {'Bad picks':<14}")
    print("-" * 40)
    all_results = {}
    for gen in gens:
        ckpt_path = exp_dir / f"decknet_gen{gen}.pt"
        if not ckpt_path.exists():
            print(f"{gen:<6}  (no checkpoint)")
            continue
        r = eval_checkpoint(ckpt_path, card_vocab, scenarios)
        all_results[gen] = r
        print(f"{gen:<6} {r['correct']}/{r['total']} ({100*r['accuracy']:.1f}%)"
              f"   {r['bad_picks']}/{r['bad_total']}")

    # Per-category breakdown for the latest gen
    if gens and gens[-1] in all_results:
        latest = all_results[gens[-1]]
        print(f"\nPer-category (gen {gens[-1]}):")
        print(f"{'Category':<20} {'Acc':<14} {'n':>4}  {'Bad':>3}")
        print("-" * 50)
        for cat in sorted(latest["by_category"].keys()):
            s = latest["by_category"][cat]
            pct = 100 * s["correct"] / s["total"] if s["total"] else 0.0
            bad_str = f"{s['bad']}" if s["bad"] else ""
            print(f"{cat:<20} {s['correct']}/{s['total']} ({pct:.0f}%)   {s['total']:>4}  {bad_str:>3}")

    # Reference baselines
    print("\nReference baselines on 74-scn suite:")
    print("  Random choice                   : 35.7%")
    print("  Always-SKIP                     : 21.6%")
    print("  Untrained combat-net forward-pass: 33.8%")
    print("  Rollout heuristic (gen 88)      : 39.2%")
    print("  Combat-net forward-pass (gen 88): 53.0%")


if __name__ == "__main__":
    main()
