"""Compare base eval to a hypothetical 'mask end_turn when other plays exist'.

Question: how much would v3 g88 improve if we simply forbade end_turn at
inference whenever any non-end-turn action is valid? End-turn bias is a
documented failure mode (see project_card_data_bugs.md, the human-label
session showing 17/20 labeled bad picks were end_turn).

This is a CHEAP proxy test using the curated P-Eval suite, NOT real
combat WR. Combat WR with masking would need engine changes; the eval
harness shows the directional answer in seconds.

Per scenario we:
  1. Run the network forward to get logits over all action slots
  2. Identify end_turn slots via action_features[..., _FLAG_END_TURN]
  3. If any non-end-turn slot is valid, set end_turn logits to -inf
  4. Argmax → masked policy pick
  5. Compare to scenario.best_actions (same as run_eval)

Output: A/B P-Eval, plus per-category deltas + end_turn metric drops.
"""

from __future__ import annotations

import argparse
import sys
import json
from pathlib import Path

import numpy as np
import torch

from sts2_solver.betaone.network import (
    BetaOneNetwork,
    network_kwargs_from_meta,
    MAX_ACTIONS,
    MAX_HAND,
    ACTION_DIM,
    STATE_DIM,
    CARD_STATS_DIM,
)
from sts2_solver.betaone.eval import (
    build_scenarios,
    encode_state,
    encode_action,
    _card_id_lookup,
    _FLAG_END_TURN,
)


def _load_network(ckpt_path: Path) -> BetaOneNetwork:
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    arch = ckpt.get("arch_meta") or {}
    kwargs = network_kwargs_from_meta(arch)
    cv_path = ckpt_path.parent / "card_vocab.json"
    with open(cv_path, encoding="utf-8") as f:
        vocab = json.load(f)
    net = BetaOneNetwork(num_cards=len(vocab), **kwargs)
    net.load_state_dict(ckpt["model_state_dict"], strict=False)
    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)
    return net, vocab


def policy_pick_with_optional_mask(
    sc, net: BetaOneNetwork, card_vocab: dict, *, mask_end_turn: bool
) -> int:
    """Policy argmax. If mask_end_turn and any non-end-turn slot is valid, set
    end_turn logits to -inf before argmax."""
    sv = encode_state(sc)
    st = torch.tensor([sv], dtype=torch.float32)
    af = torch.zeros(1, MAX_ACTIONS, ACTION_DIM)
    am = torch.ones(1, MAX_ACTIONS, dtype=torch.bool)  # True = INVALID per network convention
    hi = torch.zeros(1, MAX_HAND, dtype=torch.long)
    ai = torch.zeros(1, MAX_ACTIONS, dtype=torch.long)
    for i, c in enumerate(sc.hand[:MAX_HAND]):
        cid = c.get("id", "") + ("+" if c.get("upgraded") else "")
        hi[0, i] = card_vocab.get(cid, 0)
    n = len(sc.actions)
    end_turn_slots: list[int] = []
    for i, a in enumerate(sc.actions[:MAX_ACTIONS]):
        for j, v in enumerate(encode_action(a, sc.enemies)):
            af[0, i, j] = v
        am[0, i] = False  # valid slot
        if a.action_type == "end_turn":
            end_turn_slots.append(i)
        if a.card is not None:
            cid = a.card.get("id", "") + ("+" if a.card.get("upgraded") else "")
            ai[0, i] = card_vocab.get(cid, 0)

    with torch.no_grad():
        out = net(st, af, am, hi, ai)
    logits = out[0][0].numpy()

    if mask_end_turn and end_turn_slots:
        non_end_turn_valid = any(
            i < n and i not in end_turn_slots for i in range(n)
        )
        if non_end_turn_valid:
            for slot in end_turn_slots:
                logits[slot] = -1e9

    pick = int(logits[:n].argmax())
    return pick


def run_eval(net, vocab: dict, *, mask_end_turn: bool) -> dict:
    """Re-implementation of eval.run_eval that supports end_turn-mask flag.

    Mirrors the original's pass/fail semantics but only computes the headline
    summary (no per-scenario printout) — keeps stdout clean for the A/B.
    """
    scenarios = build_scenarios()
    by_cat: dict[str, list[bool]] = {}
    n_pass = 0
    n_total = 0
    end_turn_picked_when_bad = 0
    end_turn_bad_scenarios = 0
    end_turn_probs_on_bad: list[float] = []

    for sc in scenarios:
        pick = policy_pick_with_optional_mask(sc, net, vocab, mask_end_turn=mask_end_turn)
        passed = pick in sc.best_actions
        by_cat.setdefault(sc.category, []).append(passed)
        n_pass += int(passed)
        n_total += 1

        # End-turn bias diagnostics (mirror eval.run_eval's fields):
        # is end_turn in bad_actions for this scenario?
        et_slots = [i for i, a in enumerate(sc.actions) if a.action_type == "end_turn"]
        if et_slots and any(s in sc.bad_actions for s in et_slots):
            end_turn_bad_scenarios += 1
            if pick in et_slots:
                end_turn_picked_when_bad += 1

    return {
        "passed": n_pass,
        "total": n_total,
        "by_category": {
            cat: {"passed": sum(rs), "total": len(rs)} for cat, rs in by_cat.items()
        },
        "end_turn_bad_scenarios": end_turn_bad_scenarios,
        "end_turn_picked_when_bad": end_turn_picked_when_bad,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    args = p.parse_args()

    net, vocab = _load_network(Path(args.checkpoint))
    print(f"Loaded {args.checkpoint}")
    print()

    base = run_eval(net, vocab, mask_end_turn=False)
    masked = run_eval(net, vocab, mask_end_turn=True)

    print(f"BASE       : P-Eval {base['passed']}/{base['total']} ({100*base['passed']/base['total']:.1f}%)")
    print(f"MASK ETURN : P-Eval {masked['passed']}/{masked['total']} ({100*masked['passed']/masked['total']:.1f}%)")
    print()
    print(f"End-turn-bad scenarios: {base['end_turn_bad_scenarios']}")
    print(f"  base picked end_turn there:    {base['end_turn_picked_when_bad']}/{base['end_turn_bad_scenarios']}")
    print(f"  masked picked end_turn there:  {masked['end_turn_picked_when_bad']}/{masked['end_turn_bad_scenarios']}")
    print()
    print("Per-category P-Eval deltas (only categories that moved):")
    for cat in sorted(base["by_category"].keys()):
        b = base["by_category"][cat]["passed"]
        m = masked["by_category"][cat]["passed"]
        if b != m:
            arrow = "↑" if m > b else "↓"
            print(f"  {cat:18s}: {b:3d} -> {m:3d}  ({m-b:+d} {arrow})")


if __name__ == "__main__":
    main()
