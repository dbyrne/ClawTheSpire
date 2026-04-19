"""P-Eval with MCTS-1000 verification.

For each combat scenario, compares:
  - Policy-head pick (raw forward pass through the value/policy net)
  - MCTS-1000 pick (full tree search via betaone_mcts_search FFI)

Classifies each scenario:
  - CLEAN: both policy and MCTS pick a best action
  - ECHO:  policy picks BAD, MCTS also picks BAD (search amplifies the bias)
  - FIXED: policy picks BAD, MCTS picks OK (search corrects the error)
  - BROKE: policy picks OK, MCTS picks BAD (search degrades)

ECHO is the echo-chamber diagnostic — search has no corrective pressure
because value-head leaf evaluations share the policy-head's biases.

Usage:
    PYTHONIOENCODING=utf-8 python scripts/combat_eval_mcts.py \\
        --checkpoint <path/to/betaone.pt> \\
        [--num-sims 1000] [--only-echo] [--scenarios name1,name2]
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

import torch

import sts2_engine

from sts2_solver.betaone.deck_gen import lookup_card
from sts2_solver.betaone.eval import (
    ActionSpec,
    Scenario,
    build_scenarios,
    encode_action,
    encode_state,
)
from sts2_solver.betaone.network import (
    ACTION_DIM, MAX_ACTIONS, MAX_HAND,
    BetaOneNetwork, export_onnx, network_kwargs_from_meta,
)


# ---------------------------------------------------------------------------
# Scenario → Rust CombatState JSON
# ---------------------------------------------------------------------------

def _dummy_card() -> dict:
    return lookup_card("STRIKE_SILENT")


def scenario_to_state_json(sc: Scenario) -> str:
    """Serialize a Scenario into a CombatState-compatible JSON for Rust FFI.

    Hand + enemy + player are direct mappings. Draw/discard/exhaust piles are
    filled with dummy Strikes to match the scenario's stated sizes (MCTS may
    end the turn and need pile contents for the next draw).
    """
    pending = None
    if sc.pending_choice is not None:
        pending = {
            "choice_type": sc.pending_choice.get("choice_type", "discard_from_hand"),
            "num_choices": sc.pending_choice.get("num_choices", 1),
            "source_card_id": sc.pending_choice.get("source_card_id", ""),
            "valid_indices": sc.pending_choice.get("valid_indices"),
            "chosen_so_far": sc.pending_choice.get("chosen_so_far", []),
        }

    player = {
        "hp": sc.player.get("hp", 70),
        "max_hp": sc.player.get("max_hp", 70),
        "block": sc.player.get("block", 0),
        "energy": sc.player.get("energy", 3),
        "max_energy": sc.player.get("max_energy", 3),
        "powers": sc.player.get("powers", {}),
        "hand": list(sc.hand),
        "draw_pile": [_dummy_card() for _ in range(sc.draw_size)],
        "discard_pile": [_dummy_card() for _ in range(sc.discard_size)],
        "exhaust_pile": [_dummy_card() for _ in range(sc.exhaust_size)],
        "potions": [],
    }

    enemies = []
    for i, e in enumerate(sc.enemies):
        enemies.append({
            "id": e.get("id", f"TEST_ENEMY_{i}"),
            "name": e.get("name", "Test"),
            "hp": e.get("hp", 30),
            "max_hp": e.get("max_hp", e.get("hp", 30)),
            "block": e.get("block", 0),
            "powers": e.get("powers", {}),
            "intent_type": e.get("intent_type", "Attack"),
            "intent_damage": e.get("intent_damage", 10),
            "intent_hits": e.get("intent_hits", 1),
        })

    state = {
        "player": player,
        "enemies": enemies,
        "turn": sc.turn,
        "cards_played_this_turn": 0,
        "attacks_played_this_turn": 0,
        "cards_drawn_this_turn": 0,
        "discards_this_turn": 0,
        "last_x_cost": 0,
        "relics": list(sc.relics),
        "floor": 1,
        "gold": 0,
        "pending_choice": pending,
        "act_id": "",
        "boss_id": "",
        "map_path": [],
    }
    return json.dumps(state)


# ---------------------------------------------------------------------------
# Policy-head forward pass
# ---------------------------------------------------------------------------

def policy_pick(sc: Scenario, net: BetaOneNetwork, card_vocab: dict) -> tuple[int, list[float]]:
    """Run the policy head forward pass; return (pick_idx, probs)."""
    sv = encode_state(sc)
    st = torch.tensor([sv], dtype=torch.float32)
    af = torch.zeros(1, MAX_ACTIONS, ACTION_DIM)
    am = torch.ones(1, MAX_ACTIONS, dtype=torch.bool)
    hi = torch.zeros(1, MAX_HAND, dtype=torch.long)
    ai = torch.zeros(1, MAX_ACTIONS, dtype=torch.long)
    for i, c in enumerate(sc.hand[:MAX_HAND]):
        cid = c.get("id", "") + ("+" if c.get("upgraded") else "")
        hi[0, i] = card_vocab.get(cid, 0)
    for i, a in enumerate(sc.actions[:MAX_ACTIONS]):
        for j, v in enumerate(encode_action(a, sc.enemies)):
            af[0, i, j] = v
        am[0, i] = False
        if a.card is not None:
            cid = a.card.get("id", "") + ("+" if a.card.get("upgraded") else "")
            ai[0, i] = card_vocab.get(cid, 0)
    with torch.no_grad():
        out = net(st, af, am, hi, ai)
    logits = out[0][0].numpy()
    n = len(sc.actions)
    probs = torch.softmax(out[0][0][:n], dim=-1).numpy()
    pick = int(logits[:n].argmax())
    return pick, probs.tolist()


# ---------------------------------------------------------------------------
# MCTS-1000 via FFI
# ---------------------------------------------------------------------------

def mcts_pick(
    sc: Scenario, onnx_path: str, card_vocab_json: str,
    num_sims: int = 1000, seed: int = 42,
) -> tuple[int | None, dict]:
    """Run MCTS via FFI. Returns (scenario_action_idx or None, raw_result)."""
    state_json = scenario_to_state_json(sc)
    result = sts2_engine.betaone_mcts_search(
        state_json=state_json,
        onnx_path=onnx_path,
        card_vocab_json=card_vocab_json,
        num_sims=num_sims, temperature=0.0, seed=seed, gen_id=0,
    )
    pick_idx = _match_mcts_to_scenario(result, sc)
    return pick_idx, result


def _match_mcts_to_scenario(mcts_result: dict, sc: Scenario) -> int | None:
    """Map MCTS's chosen action back to a scenario.actions index.

    MCTS returns (action_type, card_idx [into hand], target_idx, choice_idx).
    Scenario actions are a list of ActionSpec with (action_type, card dict, target_idx).
    Match on action_type + card id + target (or choice_idx for discard).
    """
    a_type = mcts_result.get("action_type")
    card_idx = mcts_result.get("card_idx")
    target = mcts_result.get("target_idx")
    choice = mcts_result.get("choice_idx")

    picked_card_id = None
    if card_idx is not None and card_idx < len(sc.hand):
        picked_card_id = sc.hand[card_idx].get("id", "")
    picked_choice_id = None
    if choice is not None and choice < len(sc.hand):
        picked_choice_id = sc.hand[choice].get("id", "")

    for i, a in enumerate(sc.actions):
        if a.action_type != a_type:
            continue
        if a_type == "end_turn":
            return i
        if a_type == "play_card":
            if a.card is None:
                continue
            if a.card.get("id", "") == picked_card_id and a.target_idx == target:
                return i
        if a_type == "choose_card":
            # Match by hand card id (covers both "Sly discard" patterns and
            # generic discard-from-hand). First matching action wins; duplicates
            # in the scenario list are equivalent.
            if a.card is None:
                continue
            if a.card.get("id", "") == picked_choice_id:
                return i
        # use_potion not currently in scenarios
    return None


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def classify(scenario: Scenario, policy_idx: int, mcts_idx: int | None) -> str:
    p_ok = policy_idx in scenario.best_actions or policy_idx in getattr(scenario, 'acceptable_idx', [])
    p_bad = policy_idx in scenario.bad_actions
    if mcts_idx is None:
        return "NOMATCH"
    m_ok = mcts_idx in scenario.best_actions or mcts_idx in getattr(scenario, 'acceptable_idx', [])
    m_bad = mcts_idx in scenario.bad_actions

    if p_bad and m_bad:
        return "ECHO"
    if p_bad and m_ok:
        return "FIXED"
    if p_ok and m_bad:
        return "BROKE"
    if p_ok and m_ok:
        return "CLEAN"
    return "MIXED"  # miss (neither ok nor bad)


def load_net(ckpt_path: Path, device: str = "cpu") -> tuple[BetaOneNetwork, dict, str]:
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    cv_path = ckpt_path.parent / "card_vocab.json"
    card_vocab = json.loads(cv_path.read_text())
    kwargs = network_kwargs_from_meta(ckpt.get("arch_meta"))
    net = BetaOneNetwork(num_cards=len(card_vocab), **kwargs)
    net.load_state_dict(ckpt["model_state_dict"])
    net.eval()
    onnx_dir = Path(tempfile.gettempdir()) / "combat_eval_mcts_onnx"
    onnx_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = export_onnx(net, str(onnx_dir))
    return net, card_vocab, onnx_path


def _label(action: ActionSpec) -> str:
    if action.action_type == "end_turn":
        return "End turn"
    if action.action_type == "play_card" and action.card is not None:
        return f"play {action.card.get('id', '?')}"
    if action.action_type == "choose_card" and action.card is not None:
        return f"discard {action.card.get('id', '?')}"
    return action.label or action.action_type


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--num-sims", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--only-echo", action="store_true",
                   help="Only print ECHO/BROKE/FIXED (disagreements)")
    p.add_argument("--scenarios",
                   help="Comma-separated scenario names (default: all)")
    p.add_argument("--category",
                   help="Only scenarios with this category")
    args = p.parse_args()

    ckpt = Path(args.checkpoint)
    print(f"Loading net from {ckpt}")
    net, card_vocab, onnx_path = load_net(ckpt)
    vocab_json = json.dumps(card_vocab)
    print(f"  ONNX exported to {onnx_path}")

    scenarios = build_scenarios()
    if args.scenarios:
        wanted = set(args.scenarios.split(","))
        scenarios = [s for s in scenarios if s.name in wanted]
    if args.category:
        scenarios = [s for s in scenarios if s.category == args.category]
    print(f"Evaluating {len(scenarios)} scenarios @ {args.num_sims} sims")

    tally = defaultdict(int)
    by_cat_tally = defaultdict(lambda: defaultdict(int))
    t0 = time.time()

    for i, sc in enumerate(scenarios):
        policy_idx, probs = policy_pick(sc, net, card_vocab)
        mcts_idx, mcts_raw = mcts_pick(sc, onnx_path, vocab_json,
                                        num_sims=args.num_sims, seed=args.seed)
        verdict = classify(sc, policy_idx, mcts_idx)
        tally[verdict] += 1
        by_cat_tally[sc.category][verdict] += 1

        if args.only_echo and verdict not in ("ECHO", "FIXED", "BROKE", "NOMATCH"):
            continue

        p_label = _label(sc.actions[policy_idx]) if policy_idx < len(sc.actions) else "?"
        p_tag = "OK" if policy_idx in sc.best_actions or policy_idx in getattr(sc, 'acceptable_idx', []) else (
            "BAD" if policy_idx in sc.bad_actions else "--")

        if mcts_idx is not None and mcts_idx < len(sc.actions):
            m_label = _label(sc.actions[mcts_idx])
            m_tag = "OK" if mcts_idx in sc.best_actions or mcts_idx in getattr(sc, 'acceptable_idx', []) else (
                "BAD" if mcts_idx in sc.bad_actions else "--")
            visits = mcts_raw.get("policy", [])
            top_visit = max(visits) if visits else 0.0
            m_viz = f"{m_label} ({100*top_visit:.0f}% visits, v={mcts_raw.get('root_value', 0):+.2f})"
        else:
            m_viz = f"NOMATCH (action_type={mcts_raw.get('action_type')})"
            m_tag = "?"

        print(f"\n[{verdict}] {sc.name}  ({sc.category})")
        print(f"  policy:   [{p_tag}]  {p_label}  ({100*probs[policy_idx]:.1f}%)")
        print(f"  mcts-{args.num_sims}: [{m_tag}]  {m_viz}")

    print("\n" + "=" * 70)
    print(f"Total: {sum(tally.values())} scenarios in {time.time()-t0:.1f}s")
    for v in ("CLEAN", "ECHO", "FIXED", "BROKE", "MIXED", "NOMATCH"):
        if tally[v]:
            print(f"  {v:<8} {tally[v]:>3}")
    print()
    print("Per-category ECHO rate (policy BAD → MCTS also BAD):")
    for cat in sorted(by_cat_tally.keys()):
        stats = by_cat_tally[cat]
        total = sum(stats.values())
        echo_pct = 100 * stats["ECHO"] / total if total else 0
        fixed_pct = 100 * stats["FIXED"] / total if total else 0
        print(f"  {cat:<20} echo={stats['ECHO']}/{total} ({echo_pct:.0f}%)  fixed={stats['FIXED']}/{total} ({fixed_pct:.0f}%)")


if __name__ == "__main__":
    main()
