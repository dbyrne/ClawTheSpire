"""DAgger-style distillation dataset generation.

Closes the imitation-learning distribution-shift gap: instead of training the
student on states the TEACHER visits (which under-represents the close-call
decisions where policy/MCTS disagree), train on states the STUDENT visits
with teacher MCTS labels. Student's policy is only evaluated at states IT
reaches, so those are the states that need correct labels.

Two phases:
  1. Sampler self-play — run the current student (with exploration noise +
     optional MCTS) through combats on the encounter set. Record state_jsons
     + encoded features at every decision point.
  2. Teacher labeling — use `betaone_mcts_reanalyse` to run teacher
     (v3 g88 + MCTS-N) on each recorded state_json. The returned visit
     distribution + root value become the training targets.

Output: dataset.pkl with the same schema as distill_generate.py, so it drops
into distill_c51.py / distill_transformer.py unchanged.

Usage:
    python -m scripts.distill_dagger \\
        --sampler experiments/distill-c51-diverse-v1/betaone_latest.pt \\
        --teacher C:/coding-projects/sts2-reanalyse-v3/sts2-solver/experiments/reanalyse-v3/betaone_gen88.pt \\
        --card-vocab C:/coding-projects/sts2-reanalyse-v3/sts2-solver/experiments/reanalyse-v3/card_vocab.json \\
        --encounter-set uber-decks-v1 \\
        --n-combats 1000 \\
        --sampler-sims 1000 \\
        --teacher-sims 2000 \\
        --output experiments/distill-dagger-v1
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import sts2_engine

from sts2_solver.betaone.network import (
    BetaOneNetwork,
    STATE_DIM,
    ACTION_DIM,
    MAX_ACTIONS,
    MAX_HAND,
    export_onnx,
    network_kwargs_from_meta,
)
from sts2_solver.betaone.encounter_set import load_encounter_set
from sts2_solver.betaone.data_utils import load_solver_json, build_monster_data_json


def _load_network(checkpoint_path: str):
    """Load a checkpoint with the appropriate network class (scalar/distributional/transformer)."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    arch_meta = ckpt.get("arch_meta") or {}
    kwargs = network_kwargs_from_meta(arch_meta)
    num_cards = ckpt.get("num_cards") or arch_meta.get("num_cards") or 120

    is_transformer = bool(arch_meta.get("transformer_trunk"))
    is_distributional = bool(arch_meta.get("distributional_value"))

    if is_transformer:
        from distill_transformer import TransformerStudentNetwork
        net = TransformerStudentNetwork(num_cards=num_cards, **kwargs)
        tag = "[transformer]"
    elif is_distributional:
        from distill_c51 import DistStudentNetwork
        net = DistStudentNetwork(num_cards=num_cards, **kwargs)
        tag = "[distributional]"
    else:
        net = BetaOneNetwork(num_cards=num_cards, **kwargs)
        tag = "[scalar]"

    net.load_state_dict(ckpt["model_state_dict"])
    net.eval()
    print(f"Loaded {tag} {Path(checkpoint_path).name}: {net.param_count():,} params")
    return net, num_cards


def _sample_state_batches(encounters, n_combats, seeds_per_encounter):
    """Group (encounter, seed) pairs by HP for the engine's batched self-play.

    Picks n_combats (encounter, seed) pairs cycling through encounters so we
    get balanced coverage. Returns list of (enemies, decks, relics, hp, seeds).
    """
    from collections import defaultdict
    rng = np.random.RandomState(42)
    pairs = []
    for i in range(n_combats):
        enc = encounters[i % len(encounters)]
        seed = (i // len(encounters)) * 10_000 + i  # deterministic
        pairs.append((enc, seed))

    hp_groups: dict[int, tuple[list, list, list, list]] = defaultdict(
        lambda: ([], [], [], [])
    )
    for enc, seed in pairs:
        hp = enc.get("hp", 70)
        hp_groups[hp][0].append(enc["enemies"])
        hp_groups[hp][1].append(enc["deck"])
        hp_groups[hp][2].append(enc.get("relics", []))
        hp_groups[hp][3].append(seed)
    return [(encs, dks, rels, hp, seeds)
            for hp, (encs, dks, rels, seeds) in sorted(hp_groups.items())]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sampler", required=True, help="Student checkpoint (drives state distribution)")
    p.add_argument("--teacher", required=True, help="Teacher checkpoint (labels each state)")
    p.add_argument("--card-vocab", required=True, help="Path to card_vocab.json")
    p.add_argument("--encounter-set", required=True, help="Encounter set name")
    p.add_argument("--n-combats", type=int, default=1000, help="Sampler combats to run")
    p.add_argument("--sampler-sims", type=int, default=1000,
                   help="MCTS sims during sampler rollouts (higher = closer to deployment, slower)")
    p.add_argument("--teacher-sims", type=int, default=2000,
                   help="MCTS sims for teacher labeling (the main quality lever)")
    p.add_argument("--temperature", type=float, default=1.5,
                   help="Sampler action-selection temperature (>1 broadens state distribution)")
    p.add_argument("--noise-frac", type=float, default=0.5,
                   help="Dirichlet noise on sampler root priors")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--c-puct", type=float, default=1.5)
    p.add_argument("--pw-k", type=float, default=2.0)
    p.add_argument("--player-max-hp", type=int, default=70)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    sys.stdout.reconfigure(encoding="utf-8")
    os.makedirs(args.output, exist_ok=True)

    # ----- Phase 1: Sampler rollouts to generate states -----
    print("=" * 70)
    print("Phase 1: SAMPLER self-play (state distribution generation)")
    print("=" * 70)
    sampler_net, num_cards = _load_network(args.sampler)
    sampler_onnx_dir = os.path.join(args.output, "sampler_onnx")
    os.makedirs(sampler_onnx_dir, exist_ok=True)
    sampler_onnx = export_onnx(sampler_net, sampler_onnx_dir)
    print(f"Sampler ONNX: {sampler_onnx}")

    with open(args.card_vocab, encoding="utf-8") as f:
        card_vocab_json = f.read()
    monster_json = build_monster_data_json()
    profiles_json = load_solver_json("enemy_profiles.json")

    encounters = load_encounter_set(args.encounter_set)
    print(f"Encounter set: {args.encounter_set} — {len(encounters)} encs, {args.n_combats} combats")

    batches = _sample_state_batches(encounters, args.n_combats, 1)
    total_combats = sum(len(b[0]) for b in batches)
    print(f"Total combats: {total_combats} across {len(batches)} HP-groups")

    all_states = []
    all_act_feat = []
    all_act_masks = []
    all_hand_ids = []
    all_action_ids = []
    all_state_jsons = []
    total_steps = 0

    t_start = time.time()
    combat_done = 0
    for hp_idx, (b_enc, b_dks, b_rels, b_hp, b_seeds) in enumerate(batches):
        n = len(b_enc)
        for chunk_start in range(0, n, args.batch_size):
            chunk_end = min(chunk_start + args.batch_size, n)
            sub_enc = b_enc[chunk_start:chunk_end]
            sub_dks = b_dks[chunk_start:chunk_end]
            sub_rels = b_rels[chunk_start:chunk_end]
            sub_seeds = b_seeds[chunk_start:chunk_end]
            t0 = time.time()
            rollout = sts2_engine.betaone_mcts_selfplay(
                encounters_json=json.dumps(sub_enc),
                decks_json=json.dumps(sub_dks),
                player_hp=b_hp,
                player_max_hp=args.player_max_hp,
                player_max_energy=3,
                relics_json=json.dumps(sub_rels),
                potions_json="[]",
                monster_data_json=monster_json,
                enemy_profiles_json=profiles_json,
                onnx_path=sampler_onnx,
                card_vocab_json=card_vocab_json,
                num_sims=args.sampler_sims,
                temperature=args.temperature,  # HIGH for exploration
                seeds=sub_seeds,
                gen_id=0,
                add_noise=True,
                turn_boundary_eval=True,
                c_puct=args.c_puct,
                pomcp=True,
                noise_frac=args.noise_frac,
                pw_k=args.pw_k,
            )
            n_steps = rollout["total_steps"]
            if n_steps > 0:
                all_states.append(np.array(rollout["states"], dtype=np.float32).reshape(-1, STATE_DIM))
                all_act_feat.append(np.array(rollout["action_features"], dtype=np.float32).reshape(-1, MAX_ACTIONS, ACTION_DIM))
                all_act_masks.append(np.array(rollout["action_masks"]).reshape(-1, MAX_ACTIONS))
                all_hand_ids.append(np.array(rollout["hand_card_ids"], dtype=np.int64).reshape(-1, MAX_HAND))
                all_action_ids.append(np.array(rollout["action_card_ids"], dtype=np.int64).reshape(-1, MAX_ACTIONS))
                all_state_jsons.extend(rollout.get("state_jsons", [""] * n_steps))
                total_steps += n_steps
            combat_done += (chunk_end - chunk_start)
            dt = time.time() - t0
            elapsed = time.time() - t_start
            rate = combat_done / max(elapsed, 1e-6)
            eta = (total_combats - combat_done) / max(rate, 1e-6)
            print(
                f"  HP {b_hp} {chunk_start}-{chunk_end}: {n_steps} steps in {dt:.1f}s | "
                f"{combat_done}/{total_combats} combats, {total_steps} states, "
                f"{rate:.1f} c/s, ETA {eta/60:.1f}min",
                flush=True,
            )

    states = np.concatenate(all_states, axis=0)
    act_feat = np.concatenate(all_act_feat, axis=0)
    act_masks = np.concatenate(all_act_masks, axis=0)
    hand_ids = np.concatenate(all_hand_ids, axis=0)
    action_ids = np.concatenate(all_action_ids, axis=0)
    state_jsons = [s for s in all_state_jsons if s]  # keep order; filter empties

    # Keep only states that have a state_json (needed for teacher labeling).
    # Engine should return state_json for every step — filter as safety.
    if len(state_jsons) != len(states):
        print(f"WARNING: mismatch — {len(state_jsons)} jsons vs {len(states)} states; aligning")
        valid = [i for i, s in enumerate(all_state_jsons) if s]
        states = states[valid]
        act_feat = act_feat[valid]
        act_masks = act_masks[valid]
        hand_ids = hand_ids[valid]
        action_ids = action_ids[valid]
        state_jsons = [all_state_jsons[i] for i in valid]

    phase1_time = time.time() - t_start
    print(f"Phase 1 done: {len(states)} states in {phase1_time/60:.1f}min")

    # ----- Phase 2: Teacher MCTS labels each collected state -----
    print()
    print("=" * 70)
    print("Phase 2: TEACHER MCTS labeling (per-state reanalyse)")
    print("=" * 70)
    teacher_net, teacher_num_cards = _load_network(args.teacher)
    teacher_onnx_dir = os.path.join(args.output, "teacher_onnx")
    os.makedirs(teacher_onnx_dir, exist_ok=True)
    teacher_onnx = export_onnx(teacher_net, teacher_onnx_dir)
    print(f"Teacher ONNX: {teacher_onnx}")
    print(f"Labeling {len(state_jsons)} states with MCTS-{args.teacher_sims}...")

    t_label_start = time.time()
    # Seeds are deterministic for reproducibility
    seeds = [i * 7919 for i in range(len(state_jsons))]
    label_result = sts2_engine.betaone_mcts_reanalyse(
        state_jsons=state_jsons,
        enemy_profiles_json=profiles_json,
        onnx_path=teacher_onnx,
        card_vocab_json=card_vocab_json,
        num_sims=args.teacher_sims,
        temperature=1.0,
        seeds=seeds,
        gen_id=0,
        turn_boundary_eval=True,
        c_puct=args.c_puct,
        pomcp=True,
        pw_k=args.pw_k,
    )

    teacher_policies = np.array(label_result["policies"], dtype=np.float32).reshape(-1, MAX_ACTIONS)
    teacher_values = np.array(label_result["mcts_values"], dtype=np.float32)
    ok_mask = np.array(label_result["ok"], dtype=bool)
    phase2_time = time.time() - t_label_start
    n_ok = int(ok_mask.sum())
    print(f"Phase 2 done: {n_ok}/{len(state_jsons)} labels in {phase2_time/60:.1f}min "
          f"({len(state_jsons)/max(phase2_time,1):.1f} states/sec)")

    # Filter to only successfully-labeled states
    if n_ok < len(state_jsons):
        print(f"  dropping {len(state_jsons) - n_ok} unlabeled states")
    states = states[ok_mask]
    act_feat = act_feat[ok_mask]
    act_masks = act_masks[ok_mask]
    hand_ids = hand_ids[ok_mask]
    action_ids = action_ids[ok_mask]
    teacher_policies = teacher_policies[ok_mask]
    teacher_values = teacher_values[ok_mask]

    # ----- Save dataset -----
    dataset = {
        "states": states,
        "action_features": act_feat,
        "action_masks": act_masks,
        "hand_card_ids": hand_ids,
        "action_card_ids": action_ids,
        "target_policies": teacher_policies,
        "target_values": teacher_values,
        "meta": {
            "dagger": True,
            "sampler": args.sampler,
            "teacher": args.teacher,
            "encounter_set": args.encounter_set,
            "n_combats": args.n_combats,
            "n_states": int(states.shape[0]),
            "sampler_sims": args.sampler_sims,
            "teacher_sims": args.teacher_sims,
            "temperature": args.temperature,
            "noise_frac": args.noise_frac,
            "c_puct": args.c_puct,
            "pw_k": args.pw_k,
            "phase1_time_s": phase1_time,
            "phase2_time_s": phase2_time,
        },
    }
    out_path = os.path.join(args.output, "dataset.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)
    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print()
    print(f"Wrote {len(states)} states to {out_path} ({size_mb:.1f} MB)")
    print(f"Total time: {(phase1_time+phase2_time)/60:.1f}min "
          f"(phase 1: {phase1_time/60:.1f}min, phase 2: {phase2_time/60:.1f}min)")


if __name__ == "__main__":
    main()
