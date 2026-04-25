"""Run GammaTwo MCTS rescue diagnostics with training-matched search settings.

This script evaluates a checkpoint on the curated BetaOne/GammaTwo scenario set.
It compares the policy-head pick to MCTS picks from betaone_mcts_reanalyse,
which exposes the same knobs used during training: POMCP, turn-boundary eval,
c_puct, and pw_k.

The older betaone_mcts_search helper does not expose those knobs, so its rescue
rate can disagree with the actual training-time search.
"""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path

import torch

import sts2_engine

from sts2_solver.betaone.eval import (
    ACTION_DIM,
    MAX_ACTIONS,
    MAX_HAND,
    BetaOneNetwork,
    _match_mcts_to_scenario,
    _scenario_to_state_json,
    build_scenarios,
    encode_action,
    encode_state,
)
from sts2_solver.betaone.network import export_onnx, network_kwargs_from_meta
from sts2_solver.betaone.data_utils import load_solver_json


def _load_card_vocab(checkpoint: Path) -> dict[str, int]:
    for path in (
        checkpoint.parent / "card_vocab.json",
        checkpoint.parent.parent / "card_vocab.json",
    ):
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    raise FileNotFoundError(f"card_vocab.json not found near {checkpoint}")


def _policy_pick(sc, net: BetaOneNetwork, card_vocab: dict[str, int]) -> int:
    state = torch.tensor([encode_state(sc)], dtype=torch.float32)
    action_features = torch.zeros(1, MAX_ACTIONS, ACTION_DIM)
    action_mask = torch.ones(1, MAX_ACTIONS, dtype=torch.bool)
    hand_ids = torch.zeros(1, MAX_HAND, dtype=torch.long)
    action_ids = torch.zeros(1, MAX_ACTIONS, dtype=torch.long)

    for i, card in enumerate(sc.hand[:MAX_HAND]):
        cid = card.get("id", "") + ("+" if card.get("upgraded") else "")
        hand_ids[0, i] = card_vocab.get(cid, 0)

    for i, action in enumerate(sc.actions[:MAX_ACTIONS]):
        for j, value in enumerate(encode_action(action, sc.enemies)):
            action_features[0, i, j] = value
        action_mask[0, i] = False
        if action.card is not None:
            cid = action.card.get("id", "") + ("+" if action.card.get("upgraded") else "")
            action_ids[0, i] = card_vocab.get(cid, 0)

    from sts2_solver.betaone.eval import _pile_ids_from_scenario

    draw_ids, discard_ids, exhaust_ids = _pile_ids_from_scenario(sc, card_vocab)

    with torch.no_grad():
        logits, _value = net(
            state,
            action_features,
            action_mask,
            hand_ids,
            action_ids,
            draw_ids,
            discard_ids,
            exhaust_ids,
        )
    return int(logits[0, : len(sc.actions)].argmax())


def _classify(sc, policy_idx: int, mcts_idx: int | None) -> str:
    policy_ok = policy_idx in sc.best_actions or policy_idx in getattr(sc, "acceptable_idx", [])
    policy_bad = policy_idx in sc.bad_actions
    if mcts_idx is None:
        return "NOMATCH"
    mcts_ok = mcts_idx in sc.best_actions or mcts_idx in getattr(sc, "acceptable_idx", [])
    mcts_bad = mcts_idx in sc.bad_actions
    if policy_bad and mcts_bad:
        return "ECHO"
    if policy_bad and mcts_ok:
        return "FIXED"
    if policy_ok and mcts_bad:
        return "BROKE"
    if policy_ok and mcts_ok:
        return "CLEAN"
    return "MIXED"


def _contribution(tally: dict[str, int]) -> float:
    denom = tally["FIXED"] + tally["ECHO"] + tally["BROKE"]
    return (tally["FIXED"] - tally["BROKE"]) / denom if denom else 0.0


def _load_net(checkpoint: Path, card_vocab: dict[str, int]) -> tuple[BetaOneNetwork, int, str]:
    ckpt = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    kwargs = network_kwargs_from_meta(ckpt.get("arch_meta"))
    net = BetaOneNetwork(num_cards=len(card_vocab), **kwargs)
    net.load_state_dict(ckpt["model_state_dict"], strict=False)
    net.eval()

    onnx_dir = Path(tempfile.gettempdir()) / "gammatwo_true_rescue_onnx"
    onnx_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = export_onnx(net, str(onnx_dir))
    return net, int(ckpt.get("gen", 0)), onnx_path


def run_one_config(
    *,
    scenarios,
    policy_picks: list[int],
    onnx_path: str,
    card_vocab_json: str,
    enemy_profiles_json: str,
    sims: int,
    seed: int,
    c_puct: float,
    pomcp: bool,
    turn_boundary_eval: bool,
    pw_k: float,
) -> dict:
    states = [_scenario_to_state_json(sc) for sc in scenarios]
    seeds = [seed + i for i in range(len(states))]
    out = sts2_engine.betaone_mcts_reanalyse(
        state_jsons=states,
        enemy_profiles_json=enemy_profiles_json,
        onnx_path=onnx_path,
        card_vocab_json=card_vocab_json,
        num_sims=sims,
        temperature=0.0,
        seeds=seeds,
        gen_id=int(time.time()) % 1_000_000,
        turn_boundary_eval=turn_boundary_eval,
        c_puct=c_puct,
        pomcp=pomcp,
        pw_k=pw_k,
    )

    policies = out["policies"]
    ok = out["ok"]
    tally = {k: 0 for k in ("CLEAN", "ECHO", "FIXED", "BROKE", "MIXED", "NOMATCH")}
    by_category: dict[str, dict[str, int]] = {}

    for i, sc in enumerate(scenarios):
        if not ok[i]:
            mcts_idx = None
        else:
            row = policies[i * MAX_ACTIONS : (i + 1) * MAX_ACTIONS]
            n = len(sc.actions)
            mcts_idx = max(range(n), key=lambda j: row[j]) if n else None
        verdict = _classify(sc, policy_picks[i], mcts_idx)
        tally[verdict] += 1
        by_category.setdefault(sc.category, {k: 0 for k in tally})[verdict] += 1

    return {
        "sims": sims,
        "c_puct": c_puct,
        "pomcp": pomcp,
        "turn_boundary_eval": turn_boundary_eval,
        "pw_k": pw_k,
        "total": len(scenarios),
        **{k.lower(): v for k, v in tally.items()},
        "rescue_rate": round(_contribution(tally), 4),
        "by_category": by_category,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--sims", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint)
    card_vocab = _load_card_vocab(checkpoint)
    card_vocab_json = json.dumps(card_vocab)
    enemy_profiles_json = load_solver_json("enemy_profiles.json")
    net, gen, onnx_path = _load_net(checkpoint, card_vocab)

    scenarios = build_scenarios()
    policy_picks = [_policy_pick(sc, net, card_vocab) for sc in scenarios]

    configs = [
        {
            "name": "search-default",
            "c_puct": 2.5,
            "pomcp": False,
            "turn_boundary_eval": False,
            "pw_k": 1.0,
        },
        {
            "name": "training-matched",
            "c_puct": 1.5,
            "pomcp": True,
            "turn_boundary_eval": True,
            "pw_k": 1.0,
        },
        {
            "name": "training-no-tbe",
            "c_puct": 1.5,
            "pomcp": True,
            "turn_boundary_eval": False,
            "pw_k": 1.0,
        },
        {
            "name": "policy-heavier",
            "c_puct": 3.0,
            "pomcp": True,
            "turn_boundary_eval": True,
            "pw_k": 1.0,
        },
    ]

    rows = []
    for cfg in configs:
        started = time.time()
        row = run_one_config(
            scenarios=scenarios,
            policy_picks=policy_picks,
            onnx_path=onnx_path,
            card_vocab_json=card_vocab_json,
            enemy_profiles_json=enemy_profiles_json,
            sims=args.sims,
            seed=args.seed,
            c_puct=cfg["c_puct"],
            pomcp=cfg["pomcp"],
            turn_boundary_eval=cfg["turn_boundary_eval"],
            pw_k=cfg["pw_k"],
        )
        row.update(
            {
                "name": cfg["name"],
                "checkpoint": str(checkpoint),
                "gen": gen,
                "elapsed_s": round(time.time() - started, 2),
                "timestamp": time.time(),
            }
        )
        rows.append(row)
        print(
            f"{cfg['name']}: rescue={row['rescue_rate']:+.3f} "
            f"clean={row['clean']} fixed={row['fixed']} "
            f"echo={row['echo']} broke={row['broke']} mixed={row['mixed']}"
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
