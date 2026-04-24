"""Generate a supervised distillation dataset from a frontier checkpoint.

Runs teacher (e.g. reanalyse-v3 g88) + MCTS-N on every encounter in an
encounter set with K seeds each, collecting the MCTS visit distribution
and root value as teacher targets for each decision state.

Output: a pickle dataset at <output_dir>/dataset.pkl containing tensors
suitable for supervised training:
    states        (N, STATE_DIM) float32
    action_features (N, MAX_ACTIONS, ACTION_DIM) float32
    action_masks  (N, MAX_ACTIONS) bool
    hand_card_ids (N, MAX_HAND) int64
    action_card_ids (N, MAX_ACTIONS) int64
    target_policies (N, MAX_ACTIONS) float32  — MCTS visit distribution
    target_values (N,) float32                — MCTS root value

Usage:
    python -m scripts.distill_generate \\
        --checkpoint <path/to/betaone_gen88.pt> \\
        --card-vocab <path/to/card_vocab.json> \\
        --encounter-set lean-decks-v1 \\
        --num-sims 2000 \\
        --seeds-per-encounter 10 \\
        --output experiments/distill-v1
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

# Resolve package imports when invoked as a script from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

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


def _build_batches_by_hp(
    encounters: list[dict], seeds_per_encounter: int
) -> list[tuple[list, list, list, int, list[int]]]:
    """Group (encounter, seed) pairs by HP, produce batches suitable for the engine.

    Engine batches combats by HP level. Each encounter expanded to K seeds gives
    K combats with the same HP. Returns list of (enemies, decks, relics, hp, seeds).
    """
    hp_groups: dict[int, tuple[list, list, list, list]] = defaultdict(
        lambda: ([], [], [], [])
    )
    for enc_idx, enc in enumerate(encounters):
        hp = enc.get("hp", 70)
        for seed_idx in range(seeds_per_encounter):
            # Deterministic seed = (encounter_index * 1000 + seed_index). Unique
            # across the whole generation; matches engine's seed-per-combat model.
            seed = enc_idx * 1000 + seed_idx
            hp_groups[hp][0].append(enc["enemies"])
            hp_groups[hp][1].append(enc["deck"])
            hp_groups[hp][2].append(enc.get("relics", []))
            hp_groups[hp][3].append(seed)
    return [(encs, dks, rels, hp, seeds)
            for hp, (encs, dks, rels, seeds) in sorted(hp_groups.items())]


def _load_teacher(checkpoint_path: str) -> BetaOneNetwork:
    """Load the teacher network from a .pt checkpoint (scalar or C51 distributional)."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    arch_meta = ckpt.get("arch_meta") or {}
    kwargs = network_kwargs_from_meta(arch_meta)
    num_cards = ckpt.get("num_cards") or arch_meta.get("num_cards") or 120
    if arch_meta.get("distributional_value"):
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from distill_c51 import DistStudentNetwork  # type: ignore
        network = DistStudentNetwork(num_cards=num_cards, **kwargs)
        tag = " [distributional]"
    else:
        network = BetaOneNetwork(num_cards=num_cards, **kwargs)
        tag = ""
    network.load_state_dict(ckpt["model_state_dict"])
    network.eval()
    print(
        f"Loaded teacher{tag}: {network.param_count():,} params "
        f"(vhl={kwargs.get('value_head_layers')}, trunk={kwargs.get('trunk_layers')}x{kwargs.get('trunk_hidden')})"
    )
    return network


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, help="Path to teacher .pt")
    p.add_argument("--card-vocab", required=True, help="Path to card_vocab.json")
    p.add_argument("--encounter-set", required=True, help="Encounter set name (e.g. lean-decks-v1)")
    p.add_argument("--num-sims", type=int, default=2000, help="MCTS sims per decision")
    p.add_argument("--seeds-per-encounter", type=int, default=10, help="Random seeds per encounter")
    p.add_argument("--noise-frac", type=float, default=0.5, help="Dirichlet noise fraction for exploration")
    p.add_argument("--c-puct", type=float, default=1.5, help="MCTS C_PUCT")
    p.add_argument("--batch-size", type=int, default=128, help="Combats per engine batch")
    p.add_argument("--player-max-hp", type=int, default=70)
    p.add_argument("--output", required=True, help="Output directory (creates dataset.pkl there)")
    p.add_argument("--limit-encounters", type=int, default=0, help="Limit for timing tests (0 = all)")
    args = p.parse_args()

    sys.stdout.reconfigure(encoding="utf-8")
    os.makedirs(args.output, exist_ok=True)

    # Load teacher + export ONNX for the engine
    network = _load_teacher(args.checkpoint)
    onnx_dir = os.path.join(args.output, "teacher_onnx")
    os.makedirs(onnx_dir, exist_ok=True)
    onnx_path = export_onnx(network, onnx_dir)
    print(f"Teacher ONNX: {onnx_path}")

    # Load card vocab
    with open(args.card_vocab, encoding="utf-8") as f:
        card_vocab_json = f.read()
    print(f"Card vocab: {args.card_vocab}")

    # Game data
    monster_json = build_monster_data_json()
    profiles_json = load_solver_json("enemy_profiles.json")

    # Encounter set
    encounters = load_encounter_set(args.encounter_set)
    if args.limit_encounters:
        encounters = encounters[:args.limit_encounters]
    print(f"Encounter set: {args.encounter_set} — {len(encounters)} encounters × {args.seeds_per_encounter} seeds")

    batches = _build_batches_by_hp(encounters, args.seeds_per_encounter)
    total_combats = sum(len(b[0]) for b in batches)
    print(f"Total combats: {total_combats} across {len(batches)} HP-groups")

    # Collect outputs
    all_states: list[np.ndarray] = []
    all_act_feat: list[np.ndarray] = []
    all_act_masks: list[np.ndarray] = []
    all_hand_ids: list[np.ndarray] = []
    all_action_ids: list[np.ndarray] = []
    all_policies: list[np.ndarray] = []
    all_mcts_values: list[np.ndarray] = []
    total_steps = 0

    t_start = time.time()
    combat_done = 0

    for hp_idx, (b_enc, b_dks, b_rels, b_hp, b_seeds) in enumerate(batches):
        # Chunk into smaller engine batches (memory + parallelism)
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
                onnx_path=onnx_path,
                card_vocab_json=card_vocab_json,
                num_sims=args.num_sims,
                temperature=1.0,
                seeds=sub_seeds,
                gen_id=0,
                add_noise=True,
                turn_boundary_eval=True,
                c_puct=args.c_puct,
                pomcp=True,
                noise_frac=args.noise_frac,
                pw_k=2.0,
            )
            n_steps = rollout["total_steps"]
            if n_steps > 0:
                all_states.append(
                    np.array(rollout["states"], dtype=np.float32).reshape(-1, STATE_DIM)
                )
                all_act_feat.append(
                    np.array(rollout["action_features"], dtype=np.float32).reshape(
                        -1, MAX_ACTIONS, ACTION_DIM
                    )
                )
                all_act_masks.append(
                    np.array(rollout["action_masks"]).reshape(-1, MAX_ACTIONS)
                )
                all_hand_ids.append(
                    np.array(rollout["hand_card_ids"], dtype=np.int64).reshape(
                        -1, MAX_HAND
                    )
                )
                all_action_ids.append(
                    np.array(rollout["action_card_ids"], dtype=np.int64).reshape(
                        -1, MAX_ACTIONS
                    )
                )
                all_policies.append(
                    np.array(rollout["policies"], dtype=np.float32).reshape(
                        -1, MAX_ACTIONS
                    )
                )
                all_mcts_values.append(
                    np.array(rollout["mcts_values"], dtype=np.float32)
                )
                total_steps += n_steps
            combat_done += (chunk_end - chunk_start)
            dt = time.time() - t0
            elapsed = time.time() - t_start
            rate = combat_done / max(elapsed, 1e-6)
            eta = (total_combats - combat_done) / max(rate, 1e-6)
            print(
                f"  HP {b_hp} chunk {chunk_start}-{chunk_end}: "
                f"{n_steps} steps in {dt:.1f}s | "
                f"{combat_done}/{total_combats} combats, {total_steps} states, "
                f"{rate:.1f} combats/sec, ETA {eta/60:.1f}min",
                flush=True,
            )

    # Concatenate all shards
    dataset = {
        "states": np.concatenate(all_states, axis=0),
        "action_features": np.concatenate(all_act_feat, axis=0),
        "action_masks": np.concatenate(all_act_masks, axis=0),
        "hand_card_ids": np.concatenate(all_hand_ids, axis=0),
        "action_card_ids": np.concatenate(all_action_ids, axis=0),
        "target_policies": np.concatenate(all_policies, axis=0),
        "target_values": np.concatenate(all_mcts_values, axis=0),
    }
    dataset["meta"] = {
        "teacher_checkpoint": args.checkpoint,
        "encounter_set": args.encounter_set,
        "num_sims": args.num_sims,
        "seeds_per_encounter": args.seeds_per_encounter,
        "noise_frac": args.noise_frac,
        "c_puct": args.c_puct,
        "n_encounters": len(encounters),
        "n_combats": total_combats,
        "n_states": total_steps,
        "generation_time_s": time.time() - t_start,
    }

    out_path = os.path.join(args.output, "dataset.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)
    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f"\nWrote {total_steps} states to {out_path} ({size_mb:.1f} MB)")
    print(f"Total time: {(time.time() - t_start)/60:.1f}min")


if __name__ == "__main__":
    main()
