"""Fine-tune a checkpoint with human "bad action" labels.

Uses the labels written by the /label UI as negative examples on top of a
KL-anchor against the base checkpoint:

  L = sum over labeled (state, bad_slot):
        lambda_neg * log(softmax(policy_new(state))[bad_slot])
    + beta * E[ KL(policy_new(state) || policy_base(state)) ]
                                                 over batch states

The KL anchor keeps the policy close to the base everywhere it's not
explicitly being told otherwise. The negative-example loss pushes
probability mass off labeled bad actions. Running on top of v3 g88,
this directly counteracts the MCTS-distillation noise that's been
pulling the policy toward bad picks (per the labeling session that
showed 25/29 BROKE labels — network argmax already correct, but
MCTS-played differs).

The decision pool's states are reused as the training distribution
(labeled + unlabeled both contribute to the KL anchor; only labeled
get the negative term). This is enough for ~hundreds of labels; for
larger label sets a broader state distribution would be appropriate.

Usage:
  PYTHONIOENCODING=utf-8 python scripts/finetune_with_labels.py \\
      --base-checkpoint <path/to/v3_g88.pt> \\
      --pool <experiments/_labels/pool/v3_g88_initial.jsonl> \\
      --labels <experiments/_labels/labels.jsonl> \\
      --output <path/to/finetuned.pt> \\
      [--steps 500] [--lr 1e-4] [--neg-coef 1.0] [--kl-coef 0.5]

Output: a checkpoint with the same arch_meta as the base, plus a
training-side `_finetune_meta` dict for traceability.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from sts2_solver.betaone.network import (
    BetaOneNetwork,
    network_kwargs_from_meta,
    MAX_ACTIONS,
    MAX_HAND,
    ACTION_DIM,
    STATE_DIM,
    CARD_STATS_DIM,
)
from sts2_solver.betaone.eval import encode_state, encode_action

# Note on encoding: when we re-derive (state, action_features, ...) tensors
# from raw state JSONs we need to mirror the encode_* helpers. But each pool
# decision already has the SAME tensors that were computed during self-play
# (states, action_features, action_masks, hand_card_ids, action_card_ids).
# So we re-encode from the state_json — except we DON'T have the original
# rollout's per-decision tensors in the pool. The pool stores state_json
# but not the action_feature tensors. So we re-encode at training time.

# Actually simpler: the eval.encode_state expects a Scenario object.
# The state_json we have is the engine's combat-state format, not a
# Scenario. We need a different decoder.
#
# For this MVP, the simplest path is to re-encode using the engine's own
# logic, which means going through sts2_engine. But that requires loading
# Rust, slower path.
#
# Cleanest: ship-then-encode round trip. We can extend the pool builder
# to ALSO save the per-decision tensors (states, action_features, etc.)
# so we don't need to re-encode at training time. But that's a build
# change.
#
# Pragmatic v0: run rollouts again, looking up labels by state_json hash.
# Keeps fine-tune script self-contained at cost of recompute.


def _state_json_hash(state_json: dict) -> str:
    """Stable hash for matching decisions to their stored tensors."""
    import hashlib
    return hashlib.sha256(json.dumps(state_json, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def _load_network(ckpt_path: Path, *, trainable: bool) -> BetaOneNetwork:
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    arch = ckpt.get("arch_meta") or {}
    kwargs = network_kwargs_from_meta(arch)
    cv_path = ckpt_path.parent / "card_vocab.json"
    with open(cv_path, encoding="utf-8") as f:
        vocab = json.load(f)
    net = BetaOneNetwork(num_cards=len(vocab), **kwargs)
    net.load_state_dict(ckpt["model_state_dict"], strict=False)
    if not trainable:
        net.eval()
        for p in net.parameters():
            p.requires_grad_(False)
    return net


def _load_pool(pool_path: Path) -> tuple[dict, list[dict]]:
    """Return (header_meta, [decisions])."""
    with open(pool_path, encoding="utf-8") as f:
        lines = [json.loads(l) for l in f if l.strip()]
    header = lines[0].get("_meta", {})
    decisions = lines[1:]
    return header, decisions


def _load_labels(labels_path: Path) -> dict[str, dict]:
    """Latest label per decision_id."""
    out: dict[str, dict] = {}
    if not labels_path.exists():
        return out
    with open(labels_path, encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except Exception:
                continue
            did = rec.get("decision_id")
            if did:
                out[str(did)] = rec
    return out


def build_training_tensors(
    decisions: list[dict],
    labels: dict[str, dict],
    *,
    rollouts_root: Path | None,
) -> tuple[torch.Tensor, ...]:
    """Re-derive the (state, action_features, action_masks, hand_ids, action_ids)
    tensors for each decision by reading them from the original rollout file.

    The pool decisions carry source paths pointing back to shards/gen<N>/results/
    *.pkl.gz files. Each rollout has the per-decision tensors at the step_index
    position. Walk the unique sources, load tensors, index by step.
    """
    import gzip, pickle
    from collections import defaultdict

    by_source: dict[str, list[int]] = defaultdict(list)
    for i, d in enumerate(decisions):
        by_source[d["source"]].append(i)

    n = len(decisions)
    state_t = torch.zeros((n, STATE_DIM), dtype=torch.float32)
    af_t = torch.zeros((n, MAX_ACTIONS, ACTION_DIM), dtype=torch.float32)
    am_t = torch.ones((n, MAX_ACTIONS), dtype=torch.bool)  # default masked-out
    hi_t = torch.zeros((n, MAX_HAND), dtype=torch.long)
    ai_t = torch.zeros((n, MAX_ACTIONS), dtype=torch.long)
    bad_slot_t = torch.full((n,), -1, dtype=torch.long)
    has_label_t = torch.zeros((n,), dtype=torch.bool)

    for source, idxs in by_source.items():
        try:
            with gzip.open(source, "rb") as f:
                rollout = pickle.load(f)
        except Exception as exc:
            print(f"  skip rollout {Path(source).name}: {exc}", file=sys.stderr)
            continue
        states = np.array(rollout["states"], dtype=np.float32).reshape(-1, STATE_DIM)
        af = np.array(rollout["action_features"], dtype=np.float32).reshape(-1, MAX_ACTIONS, ACTION_DIM)
        am = np.array(rollout["action_masks"]).reshape(-1, MAX_ACTIONS).astype(bool)
        hi = np.array(rollout["hand_card_ids"], dtype=np.int64).reshape(-1, MAX_HAND)
        ai = np.array(rollout["action_card_ids"], dtype=np.int64).reshape(-1, MAX_ACTIONS)
        for di in idxs:
            si = decisions[di]["step_index"]
            state_t[di] = torch.from_numpy(states[si])
            af_t[di] = torch.from_numpy(af[si])
            am_t[di] = torch.from_numpy(am[si])
            hi_t[di] = torch.from_numpy(hi[si])
            ai_t[di] = torch.from_numpy(ai[si])
            lab = labels.get(decisions[di]["id"])
            if lab and lab.get("label") == "bad" and lab.get("bad_action_slot") is not None:
                bad_slot_t[di] = int(lab["bad_action_slot"])
                has_label_t[di] = True

    return state_t, af_t, am_t, hi_t, ai_t, bad_slot_t, has_label_t


def main() -> None:
    p = argparse.ArgumentParser(description="Fine-tune with human bad-action labels")
    p.add_argument("--base-checkpoint", required=True)
    p.add_argument("--pool", required=True)
    p.add_argument("--labels", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--neg-coef", type=float, default=1.0,
                   help="Weight on log-prob penalty for labeled bad actions")
    p.add_argument("--kl-coef", type=float, default=0.5,
                   help="Weight on KL(new || base) over batch states")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    base_path = Path(args.base_checkpoint)
    pool_path = Path(args.pool)
    labels_path = Path(args.labels)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading base: {base_path}")
    base_net = _load_network(base_path, trainable=False)
    print(f"  {sum(p.numel() for p in base_net.parameters()):,} params (frozen reference)")

    train_net = _load_network(base_path, trainable=True)
    train_net.train()
    print(f"  cloned for fine-tune")

    print(f"Loading pool {pool_path.name}")
    header, decisions = _load_pool(pool_path)
    labels = _load_labels(labels_path)
    n_bad = sum(1 for l in labels.values() if l.get("label") == "bad")
    print(f"  {len(decisions)} decisions, {len(labels)} labeled ({n_bad} bad)")

    if n_bad == 0:
        print("No bad labels — nothing to fine-tune on. Exiting.", file=sys.stderr)
        sys.exit(1)

    print("Re-deriving training tensors from source rollouts...")
    started = time.perf_counter()
    state_t, af_t, am_t, hi_t, ai_t, bad_slot_t, has_label_t = build_training_tensors(
        decisions, labels, rollouts_root=None,
    )
    n = len(decisions)
    n_labeled = int(has_label_t.sum())
    print(f"  built {n} state tensors ({n_labeled} with bad-action labels) in {time.perf_counter()-started:.1f}s")

    optim = torch.optim.Adam(train_net.parameters(), lr=args.lr)
    rng = np.random.default_rng(args.seed)

    print(f"\nFine-tuning: {args.steps} steps, batch={args.batch_size}, "
          f"neg_coef={args.neg_coef}, kl_coef={args.kl_coef}")

    log_every = max(args.steps // 20, 1)
    last_neg = last_kl = 0.0
    for step in range(args.steps):
        # Sample a batch — biased toward labeled states so the negative loss
        # gets enough signal even when most pool states aren't labeled.
        labeled_idxs = np.where(has_label_t.numpy())[0]
        unlabeled_idxs = np.where(~has_label_t.numpy())[0]
        n_lab_in_batch = min(len(labeled_idxs), args.batch_size // 2)
        n_unlab_in_batch = args.batch_size - n_lab_in_batch
        chosen = np.concatenate([
            rng.choice(labeled_idxs, n_lab_in_batch, replace=False) if n_lab_in_batch > 0 else np.array([], dtype=np.int64),
            rng.choice(unlabeled_idxs, n_unlab_in_batch, replace=True) if n_unlab_in_batch > 0 and len(unlabeled_idxs) > 0 else np.array([], dtype=np.int64),
        ]).astype(np.int64)
        idx = torch.from_numpy(chosen)

        state_b = state_t[idx]
        af_b = af_t[idx]
        am_b = am_t[idx]
        hi_b = hi_t[idx]
        ai_b = ai_t[idx]
        bad_slot_b = bad_slot_t[idx]
        has_label_b = has_label_t[idx]

        # Forward both nets
        train_logits, _train_value, *_ = train_net(state_b, af_b, am_b, hi_b, ai_b)
        with torch.no_grad():
            base_logits, _base_value, *_ = base_net(state_b, af_b, am_b, hi_b, ai_b)

        # Mask: True = invalid action; we want -inf there before softmax
        valid = ~am_b
        train_masked = torch.where(valid, train_logits, torch.full_like(train_logits, -1e9))
        base_masked = torch.where(valid, base_logits, torch.full_like(base_logits, -1e9))

        train_logp = F.log_softmax(train_masked, dim=-1)
        base_p = F.softmax(base_masked, dim=-1)
        # KL(base || train) = E_base[ log base - log train ]
        # Equivalent direction (forward KL) prevents train from dropping
        # mass on actions base supports — anchors strongly to base.
        kl_per = (base_p * (base_p.clamp_min(1e-9).log() - train_logp)).sum(dim=-1)
        kl_loss = kl_per.mean()

        # Negative-example loss: -log p(bad_slot) per labeled state.
        # We want to MAXIMIZE -log p, equivalently MINIMIZE log p (push down).
        if has_label_b.any():
            labeled_train_logp = train_logp[has_label_b]
            labeled_bad = bad_slot_b[has_label_b]
            bad_logp = labeled_train_logp.gather(1, labeled_bad.unsqueeze(1)).squeeze(1)
            neg_loss = bad_logp.mean()
        else:
            neg_loss = torch.tensor(0.0)

        loss = args.kl_coef * kl_loss + args.neg_coef * neg_loss

        optim.zero_grad()
        loss.backward()
        optim.step()

        last_neg = float(neg_loss.detach())
        last_kl = float(kl_loss.detach())

        if (step + 1) % log_every == 0 or step == 0:
            print(f"  step {step+1:4d}/{args.steps}: kl={last_kl:.4f}, "
                  f"neg_logp_bad={last_neg:.4f} (lower=more pushed-down), "
                  f"total={float(loss.detach()):.4f}")

    # Save fine-tuned checkpoint with same arch_meta
    ckpt = torch.load(str(base_path), map_location="cpu", weights_only=False)
    ckpt["model_state_dict"] = train_net.state_dict()
    ckpt["_finetune_meta"] = {
        "base_checkpoint": str(base_path),
        "pool": str(pool_path),
        "labels": str(labels_path),
        "labels_total": len(labels),
        "labels_bad": n_bad,
        "labels_bad_used": n_labeled,
        "steps": args.steps,
        "lr": args.lr,
        "neg_coef": args.neg_coef,
        "kl_coef": args.kl_coef,
        "batch_size": args.batch_size,
        "final_kl": last_kl,
        "final_neg_logp": last_neg,
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    torch.save(ckpt, str(out_path))

    print(f"\nSaved fine-tuned checkpoint: {out_path}")
    print(f"  base_kl(train||base) at end: {last_kl:.4f}")
    print(f"  log p(bad) at end: {last_neg:.4f}")
    print()
    print(f"Next: eval the fine-tuned checkpoint to compare vs base")
    print(f"  cp {out_path} <experiment-dir>/betaone_finetune.pt")
    print(f"  PYTHONIOENCODING=utf-8 python -m sts2_solver.betaone.experiment_cli eval <experiment-name> --checkpoint <path>")


if __name__ == "__main__":
    main()
