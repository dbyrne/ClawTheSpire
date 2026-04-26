"""Build a labelable decision pool from existing self-play rollouts.

For human-in-the-loop labeling: we want to surface decisions where the
network's argmax pick can be reviewed by a human and flagged as "bad".

This script:
  1. Loads a checkpoint (the network whose policy we want labeled)
  2. Walks an experiment's recent shard rollouts, extracts every decision
  3. Re-runs the network forward pass on each state to get the network's
     argmax (the policy's preferred action; this differs from MCTS-played
     when MCTS rescues a wrong policy pick)
  4. Saves one decision record per state to a label-pool JSONL the UI
     reads

Each record carries:
- state_json (dict): rich state from sts2_engine — player, enemies, hand
- actions: per-slot card_id + visit count + Q value + is_played + is_network_argmax
- network_argmax_idx, mcts_played_idx
- mcts_value (root value for context)

The UI ("Skip" / "Bad") records labels against this pool, which then
feed both the eval harness (as P-Eval scenarios with bad_actions populated)
and a fine-tuning loss (negative-example penalty on the labeled action).

Usage:
  PYTHONIOENCODING=utf-8 python scripts/build_label_pool.py \\
      --checkpoint <path/to/betaone.pt> \\
      --shards-dir <path/to/experiments/<name>/shards> \\
      --output <path/to/pool.jsonl> \\
      [--max-records 2000]
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import pickle
import sys
import time
import uuid
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
)


def _load_card_id_to_name() -> dict[int, str]:
    """Inverse of card_vocab.json — maps vocab id to card_id string."""
    # The vocab is shared across experiments; pick any one.
    candidates = [
        Path("C:/coding-projects/STS2/sts2-solver/betaone_checkpoints/card_vocab.json"),
        Path("C:/coding-projects/sts2-reanalyse-v3/sts2-solver/experiments/reanalyse-v3/card_vocab.json"),
    ]
    for c in candidates:
        if c.exists():
            with open(c, encoding="utf-8") as f:
                vocab = json.load(f)
            return {int(v): str(k) for k, v in vocab.items()}
    raise FileNotFoundError("card_vocab.json not found in expected locations")


def _load_network(checkpoint_path: Path) -> tuple[BetaOneNetwork, dict, str]:
    ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    arch = ckpt.get("arch_meta") or {}
    kwargs = network_kwargs_from_meta(arch)
    cv_path = checkpoint_path.parent / "card_vocab.json"
    with open(cv_path, encoding="utf-8") as f:
        card_vocab = json.load(f)
    net = BetaOneNetwork(num_cards=len(card_vocab), **kwargs)
    net.load_state_dict(ckpt["model_state_dict"], strict=False)
    net.eval()
    gen = ckpt.get("gen", "?")
    return net, card_vocab, str(gen)


def _action_type_from_features(action_features_row: np.ndarray) -> str:
    """Decode action type from sparse flags in action_features.

    Action encoding (per src/sts2_solver/betaone/eval.py::encode_action):
      [0:28]   card stats
      [28:32]  target info (hp_frac, intent, vulnerable_flag, present_flag)
      [32]     _FLAG_END_TURN
      [33]     _FLAG_USE_POTION
      [34]     _FLAG_IS_DISCARD
    play_card has no flag set (default branch with card stats populated).
    """
    if action_features_row[34] > 0.5:
        return "choose_card"  # discard from hand
    if action_features_row[33] > 0.5:
        return "use_potion"
    if action_features_row[32] > 0.5:
        return "end_turn"
    return "play_card"


def _action_target_idx(action_features_row: np.ndarray) -> int | None:
    """Recover target enemy index from action_features when set.

    The target encoding only stores hp_frac/intent/vulnerable/present flags,
    not the literal index — so we can't recover which enemy slot was targeted
    from action_features alone. Returns None for now; UI can show the
    'targeted enemy' inferred from card.target type if needed.
    """
    if action_features_row[31] > 0.5:  # present flag
        # Note: this only tells us a target was specified, not which slot.
        return -1  # marker for "had a target"
    return None


def extract_decisions_from_rollout(
    rollout: dict,
    *,
    net: BetaOneNetwork,
    card_id_to_name: dict[int, str],
    source_path: str,
) -> list[dict]:
    """Convert one rollout's per-step data into labelable decision records."""
    state_jsons = rollout.get("state_jsons", []) or []
    if not state_jsons:
        return []  # rollouts without state_jsons can't be reviewed by humans

    n = int(rollout.get("total_steps") or len(state_jsons))
    if n == 0:
        return []

    states = np.array(rollout["states"], dtype=np.float32).reshape(-1, STATE_DIM)
    action_features = np.array(rollout["action_features"], dtype=np.float32).reshape(
        -1, MAX_ACTIONS, ACTION_DIM
    )
    action_masks = np.array(rollout["action_masks"]).reshape(-1, MAX_ACTIONS).astype(bool)
    hand_ids = np.array(rollout["hand_card_ids"], dtype=np.int64).reshape(-1, MAX_HAND)
    action_ids = np.array(rollout["action_card_ids"], dtype=np.int64).reshape(-1, MAX_ACTIONS)
    visits = np.array(rollout["child_visits"], dtype=np.int64).reshape(-1, MAX_ACTIONS)
    q_values = np.array(rollout["child_q_values"], dtype=np.float32).reshape(-1, MAX_ACTIONS)
    mcts_values = np.array(rollout["mcts_values"], dtype=np.float32)
    combat_indices = np.array(rollout["combat_indices"], dtype=np.int64)

    out: list[dict] = []
    state_t = torch.from_numpy(states)
    af_t = torch.from_numpy(action_features)
    am_t = torch.from_numpy(action_masks)
    hi_t = torch.from_numpy(hand_ids)
    ai_t = torch.from_numpy(action_ids)

    with torch.no_grad():
        # Forward in one batch — the network takes (B,*) tensors.
        logits_b, _value_b, *_ = net(state_t, af_t, am_t, hi_t, ai_t)
        logits = logits_b.cpu().numpy()  # (n, MAX_ACTIONS)

    for i in range(n):
        sj_raw = state_jsons[i]
        if not sj_raw:
            continue
        try:
            state = json.loads(sj_raw)
        except Exception:
            continue

        # action_masks uses True = MASKED (invalid); valid action slots are False
        valid = ~action_masks[i]
        if not valid.any():
            continue

        masked_logits = np.where(valid, logits[i], -np.inf)
        net_argmax = int(np.argmax(masked_logits))
        net_probs = np.exp(masked_logits - np.max(masked_logits[valid]))
        net_probs = (net_probs * valid) / max(net_probs.sum(), 1e-9)

        mcts_visits = visits[i].copy()
        mcts_visits[~valid] = 0
        mcts_played = int(np.argmax(mcts_visits)) if mcts_visits.sum() > 0 else net_argmax

        actions_out = []
        for slot in range(MAX_ACTIONS):
            if not valid[slot]:
                continue
            cid = int(action_ids[i, slot])
            actions_out.append({
                "slot": slot,
                "type": _action_type_from_features(action_features[i, slot]),
                "card_id": card_id_to_name.get(cid, "<UNK>"),
                "card_vocab_id": cid,
                "mcts_visits": int(mcts_visits[slot]),
                "mcts_q": float(q_values[i, slot]),
                "net_prob": float(net_probs[slot]),
                "is_network_argmax": slot == net_argmax,
                "is_mcts_played": slot == mcts_played,
            })

        out.append({
            "id": f"{Path(source_path).stem}:c{int(combat_indices[i])}:s{i}",
            "source": source_path,
            "combat_index": int(combat_indices[i]),
            "step_index": i,
            "state": state,
            "actions": actions_out,
            "network_argmax_slot": net_argmax,
            "mcts_played_slot": mcts_played,
            "mcts_value": float(mcts_values[i]),
            "policy_mcts_disagree": net_argmax != mcts_played,
        })
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Build a labelable decision pool")
    p.add_argument("--checkpoint", required=True, help="Network checkpoint to evaluate")
    p.add_argument(
        "--shards-dir",
        required=True,
        help="Path to experiments/<name>/shards/ (will walk all gen<N>/results/*.pkl.gz)",
    )
    p.add_argument("--output", required=True, help="Output JSONL path")
    p.add_argument(
        "--max-records",
        type=int,
        default=2000,
        help="Stop after this many decisions (default 2000 — enough for first labeling pass)",
    )
    p.add_argument(
        "--prefer-disagreements",
        action="store_true",
        help="Prioritize decisions where network argmax != MCTS played (where labels are most informative)",
    )
    args = p.parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"Error: checkpoint {ckpt_path} not found", file=sys.stderr)
        sys.exit(1)
    shards_dir = Path(args.shards_dir)
    if not shards_dir.exists():
        print(f"Error: shards dir {shards_dir} not found", file=sys.stderr)
        sys.exit(1)

    print(f"Loading checkpoint {ckpt_path}")
    net, card_vocab, gen = _load_network(ckpt_path)
    card_id_to_name = {v: k for k, v in card_vocab.items()}
    print(f"  gen {gen}, {sum(p.numel() for p in net.parameters()):,} params")

    rollout_files = sorted(shards_dir.glob("gen*/results/*.pkl.gz"))
    print(f"Found {len(rollout_files)} rollout files under {shards_dir}")
    if not rollout_files:
        sys.exit(1)

    batch_id = uuid.uuid4().hex[:12]
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    started = time.perf_counter()
    with open(out_path, "w", encoding="utf-8") as out_f:
        # Header line — useful for reproducibility
        header = {
            "_meta": {
                "batch_id": batch_id,
                "checkpoint": str(ckpt_path),
                "checkpoint_gen": gen,
                "shards_dir": str(shards_dir),
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "max_records": args.max_records,
                "prefer_disagreements": args.prefer_disagreements,
            },
        }
        out_f.write(json.dumps(header) + "\n")

        # Two passes when --prefer-disagreements: first emit all disagreements,
        # then fill remaining slots with agreements.
        all_records: list[dict] = []
        for rf in rollout_files:
            try:
                with gzip.open(rf, "rb") as f:
                    rollout = pickle.load(f)
            except Exception as exc:
                print(f"  skipping {rf.name}: {exc}", file=sys.stderr)
                continue
            decisions = extract_decisions_from_rollout(
                rollout,
                net=net,
                card_id_to_name=card_id_to_name,
                source_path=str(rf),
            )
            all_records.extend(decisions)
            if len(all_records) >= args.max_records * 5 and not args.prefer_disagreements:
                break  # we have plenty; stop reading more files

        if args.prefer_disagreements:
            disagreements = [r for r in all_records if r["policy_mcts_disagree"]]
            agreements = [r for r in all_records if not r["policy_mcts_disagree"]]
            ordered = disagreements + agreements
        else:
            ordered = all_records

        for rec in ordered[: args.max_records]:
            out_f.write(json.dumps(rec) + "\n")
            written += 1

    elapsed = time.perf_counter() - started
    n_disagree = sum(1 for r in ordered[:written] if r["policy_mcts_disagree"])
    print(
        f"Wrote {written} records to {out_path} in {elapsed:.1f}s "
        f"({n_disagree} where network and MCTS disagreed)"
    )


if __name__ == "__main__":
    main()
