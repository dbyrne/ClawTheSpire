"""Validate tight-margin scenarios against trunk-baseline-v1 gen 50.

For each new margin scenario, report top-action probability (target: 0.35-0.65
for a good discriminating scenario). Uses the same encode_state / encode_action
helpers that the eval harness uses.

Run from trunk-baseline-v1 worktree's venv.
"""
import sys, torch
sys.stdout.reconfigure(encoding="utf-8")

from sts2_solver.betaone.eval import (
    build_scenarios, encode_state, encode_action,
    _card_id_lookup, _load_card_vocab,
)
from sts2_solver.betaone.network import (
    BetaOneNetwork, network_kwargs_from_meta,
    MAX_HAND, MAX_ACTIONS, ACTION_DIM,
)

MARGIN_NAMES = {
    "damage_dagger_strike_over_skewer_2",
    "damage_split_over_skewer_3_small_margin",
    "damage_predator_over_skewer_2_vulnerable",
    "damage_exact_lethal_predator_over_split",
    "damage_dagger_strike_over_predator_at_3e",
    "damage_sucker_punch_over_strike_weak_worthless_dead",
    "block_backflip_over_defend_on_thin_pile",
    "block_defend_over_backflip_exhausted_pile",
    "block_two_defends_over_single_defend_at_damage_14",
    "block_one_defend_enough_at_damage_5",
    "combo_accuracy_before_blade_dance_single_turn",
    "combo_expose_before_predator_not_strike",
    "combo_burst_before_acrobatics_skill_double",
    "synergy_accelerant_minimal_poison_skip",
    "synergy_accelerant_medium_poison_worth_it",
    "discard_wound_over_slimed_both_status",
}


def _load_net(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    net_kwargs = network_kwargs_from_meta(ckpt.get("arch_meta"))
    card_vocab = _load_card_vocab()
    net = BetaOneNetwork(num_cards=len(card_vocab), **net_kwargs)
    net.load_state_dict(ckpt["model_state_dict"])
    net.eval()
    return net, card_vocab, ckpt.get("gen", "?")


def _score_scenario(net, card_vocab, sc):
    state_t = torch.tensor([encode_state(sc)], dtype=torch.float32)
    action_t = torch.zeros(1, MAX_ACTIONS, ACTION_DIM)
    mask_t = torch.ones(1, MAX_ACTIONS, dtype=torch.bool)
    action_ids = torch.zeros(1, MAX_ACTIONS, dtype=torch.long)
    for i, act in enumerate(sc.actions):
        action_t[0, i] = torch.tensor(encode_action(act, sc.enemies))
        mask_t[0, i] = False
        if act.card is not None:
            action_ids[0, i] = _card_id_lookup(act.card, card_vocab)
    hand_ids = torch.zeros(1, MAX_HAND, dtype=torch.long)
    for i, c in enumerate(sc.hand[:MAX_HAND]):
        hand_ids[0, i] = _card_id_lookup(c, card_vocab)
    with torch.no_grad():
        logits, _value = net(state_t, action_t, mask_t, hand_ids, action_ids)
    n = len(sc.actions)
    probs = torch.softmax(logits[0, :n], dim=0).tolist()
    top_idx = max(range(n), key=lambda i: probs[i])
    return probs, top_idx, top_idx in sc.best_actions


def main():
    base = r"C:/coding-projects/sts2-trunk-baseline-v1/sts2-solver/experiments/trunk-baseline-v1"
    net50, vocab, gen50 = _load_net(f"{base}/betaone_gen50.pt")
    net60, _, gen60 = _load_net(f"{base}/betaone_gen60.pt")

    scenarios = [s for s in build_scenarios() if s.name in MARGIN_NAMES]
    print(f"Comparing {len(scenarios)} margin scenarios: gen {gen50} vs gen {gen60}\n")
    print(f"{'name':<55} {'g50_p':>6} {'g50':>4}  {'g60_p':>6} {'g60':>4}  {'Δp':>6}")
    print("-" * 100)

    flip_to_pass = []    # gen 50 fail → gen 60 pass
    flip_to_fail = []    # gen 50 pass → gen 60 fail
    both_pass = 0
    both_fail = 0

    for sc in scenarios:
        p50, idx50, pass50 = _score_scenario(net50, vocab, sc)
        p60, idx60, pass60 = _score_scenario(net60, vocab, sc)
        top50 = p50[idx50]; top60 = p60[idx60]
        delta = top60 - top50

        status = f"{'YES' if pass50 else 'no':<3} -> {'YES' if pass60 else 'no':<3}"
        if pass50 and pass60: both_pass += 1
        elif (not pass50) and (not pass60): both_fail += 1
        elif pass60: flip_to_pass.append(sc.name)
        else: flip_to_fail.append(sc.name)

        mark = "++" if pass60 and not pass50 else ("--" if pass50 and not pass60 else "  ")
        print(f"{mark} {sc.name:<52} {top50:>5.2f}  {'Y' if pass50 else 'n':>3}   {top60:>5.2f}  {'Y' if pass60 else 'n':>3}   {delta:+.2f}")

    print()
    print(f"=== Summary: gen {gen50} vs gen {gen60} ===")
    print(f"  both passed : {both_pass}")
    print(f"  both failed : {both_fail}")
    print(f"  gen 50->60 FLIP TO PASS : {len(flip_to_pass)}")
    for n in flip_to_pass: print(f"    ++ {n}")
    print(f"  gen 50->60 FLIP TO FAIL : {len(flip_to_fail)}")
    for n in flip_to_fail: print(f"    -- {n}")


if __name__ == "__main__":
    main()
