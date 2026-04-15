"""Shared data loading and training utilities for BetaOne.

Used by both train.py (PPO) and selfplay_train.py (MCTS self-play).
"""

from __future__ import annotations

import glob
import json
import os
import random as stdlib_random
from collections import defaultdict
from pathlib import Path

from .paths import GAME_DATA_DIR, SOLVER_PKG


def load_game_json(filename: str) -> str:
    """Load a JSON file from the STS2-Agent game data directory."""
    path = GAME_DATA_DIR / filename
    if not path.exists():
        return "[]"
    return path.read_text(encoding="utf-8")


def load_solver_json(filename: str) -> str:
    """Load a JSON file from the solver package directory."""
    path = SOLVER_PKG / filename
    if not path.exists():
        return "{}"
    return path.read_text(encoding="utf-8")


def build_monster_data_json() -> str:
    """Build monster data dict keyed by ID, as JSON string for Rust."""
    monsters_raw = json.loads(load_game_json("monsters.json"))
    monsters = {}
    for m in monsters_raw:
        mid = m.get("id", "")
        if mid:
            monsters[mid] = {
                "name": m.get("name", mid),
                "min_hp": m.get("min_hp") or 20,
                "max_hp": m.get("max_hp") or m.get("min_hp") or 20,
            }
    return json.dumps(monsters)


def build_card_vocab(output_dir: str) -> tuple[dict[str, int], str]:
    """Load or create card vocab. Returns (vocab_dict, vocab_json_str).

    If card_vocab.json exists in output_dir, loads it. Otherwise builds
    from the game card database and saves to output_dir.
    """
    vocab_path = os.path.join(output_dir, "card_vocab.json")
    if os.path.exists(vocab_path):
        with open(vocab_path, encoding="utf-8") as f:
            vocab = json.load(f)
        return vocab, json.dumps(vocab)

    cards_raw = json.loads(load_game_json("cards.json"))
    vocab: dict[str, int] = {"<PAD>": 0, "<UNK>": 1}
    for c in cards_raw:
        base_id = c["id"].rstrip("+")
        if base_id not in vocab:
            vocab[base_id] = len(vocab)

    os.makedirs(output_dir, exist_ok=True)
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, indent=2)
    print(f"Card vocab: {len(vocab)} entries (saved to {vocab_path})")
    return vocab, json.dumps(vocab)


def find_latest_checkpoint(output_dir: str) -> str | None:
    """Find the best resume checkpoint: latest.pt first, else highest gen."""
    latest = os.path.join(output_dir, "betaone_latest.pt")
    if os.path.exists(latest):
        return latest
    pattern = os.path.join(output_dir, "betaone_gen*.pt")
    ckpts = glob.glob(pattern)
    if not ckpts:
        return None

    def gen_num(p: str) -> int:
        base = os.path.basename(p)
        return int(base.replace("betaone_gen", "").replace(".pt", ""))

    return max(ckpts, key=gen_num)


def setup_training_data(
    output_dir: str,
    training_set_id: str | None,
    mixed: bool,
    recorded_encounters: bool,
    recorded_frac: float,
    skip_to_final: bool,
) -> dict:
    """Set up encounters and curriculum for training.

    Returns a dict with:
        curriculum, recorded_encounters (bool), mixed (bool),
        recorded_frac, ts_data (or None), enc_pool_path
    """
    from .curriculum import CombatCurriculum

    enc_pool_path = str(SOLVER_PKG / "encounter_pool.json")
    recorded_path = str(Path(output_dir) / "recorded_encounters.jsonl")
    curriculum = CombatCurriculum(
        encounter_pool_path=enc_pool_path,
        recorded_encounters_path=recorded_path,
    )

    ts_data = None
    if training_set_id:
        from .training_set import load_training_set
        ts_data = load_training_set(training_set_id)
        ts_recorded = ts_data.get("recorded_data", [])
        ts_packages = ts_data.get("packages_data", {})
        print(f"Training set: {training_set_id}")
        print(f"  Recorded: {len(ts_recorded)} encounters")
        print(f"  Packages: {sum(len(v) for v in ts_packages.values())} encounter-HP pairs")
        mixed = True
        recorded_encounters = True
        curriculum.recorded_encounters = ts_recorded
        curriculum.tier = curriculum.max_tier
        from .packages import PACKAGES
        for pkg in PACKAGES:
            pkg.calibrated_hps = ts_packages.get(pkg.name, {})
    elif mixed:
        recorded_encounters = True
        if not curriculum.recorded_encounters:
            raise RuntimeError("No recorded encounters found — run the game bot first")
        curriculum.tier = curriculum.max_tier
    elif recorded_encounters:
        if not curriculum.recorded_encounters:
            raise RuntimeError("No recorded encounters found — run the game bot first")
        curriculum.use_recorded_only = True
        curriculum.tier = curriculum.max_tier
    elif skip_to_final:
        curriculum.tier = curriculum.max_tier
        curriculum.consecutive_good = 0
        curriculum.gens_at_tier = 0

    return {
        "curriculum": curriculum,
        "recorded_encounters": recorded_encounters,
        "mixed": mixed,
        "recorded_frac": recorded_frac,
        "ts_data": ts_data,
        "enc_pool_path": enc_pool_path,
    }


def sample_combat_batches(
    curriculum,
    combats_per_gen: int,
    mixed: bool,
    recorded_encounters: bool,
    recorded_frac: float,
    gen: int,
) -> list[tuple[list, list, list, int, int]]:
    """Sample encounters and group into batches by HP level.

    Returns list of (encounters, decks, relics, hp, count) tuples.
    """
    batches: list[tuple[list, list, list, int, int]] = []

    def _extract_relics(rec) -> list[str]:
        if rec is None:
            return []
        return list(rec.get("relics", []))

    if mixed:
        from .packages import sample_packages_batch

        n_rec = int(combats_per_gen * recorded_frac)
        n_pkg = combats_per_gen - n_rec

        # Recorded portion
        curriculum.use_recorded_only = True
        rec_enc = curriculum.sample_encounters(n_rec)
        rec_samples = getattr(curriculum, "_recorded_samples", None)
        rec_dks = [json.loads(curriculum.sample_deck_json(combat_idx=i))
                    for i in range(n_rec)]

        # Package portion
        pkg_rng = stdlib_random.Random(gen * 7919)
        pkg_enc, pkg_decks, pkg_hps = sample_packages_batch(n_pkg, rng=pkg_rng)

        # Group recorded by calibrated HP
        hp_groups: dict[int, tuple[list, list, list]] = defaultdict(lambda: ([], [], []))
        for i in range(n_rec):
            rec = rec_samples[i] if rec_samples and i < len(rec_samples) else None
            hp = rec.get("calibrated_hp", rec.get("player_hp", 70)) if rec else 70
            hp_groups[hp][0].append(rec_enc[i])
            hp_groups[hp][1].append(rec_dks[i])
            hp_groups[hp][2].append(_extract_relics(rec))
        for hp, (encs, dks_list, rels_list) in hp_groups.items():
            batches.append((encs, dks_list, rels_list, hp, len(encs)))

        # Group packages by HP
        pkg_hp_groups: dict[int, tuple[list, list, list]] = defaultdict(lambda: ([], [], []))
        for i in range(n_pkg):
            pkg_hp_groups[pkg_hps[i]][0].append(pkg_enc[i])
            pkg_hp_groups[pkg_hps[i]][1].append(pkg_decks[i])
            pkg_hp_groups[pkg_hps[i]][2].append([])
        for hp, (encs, dks_list, rels_list) in pkg_hp_groups.items():
            batches.append((encs, dks_list, rels_list, hp, len(encs)))

    elif recorded_encounters:
        rec_enc = curriculum.sample_encounters(combats_per_gen)
        rec_samples = getattr(curriculum, "_recorded_samples", None)
        rec_dks = [json.loads(curriculum.sample_deck_json(combat_idx=i))
                    for i in range(combats_per_gen)]

        hp_groups: dict[int, tuple[list, list, list]] = defaultdict(lambda: ([], [], []))
        for i in range(combats_per_gen):
            rec = rec_samples[i] if rec_samples and i < len(rec_samples) else None
            hp = rec.get("calibrated_hp", rec.get("player_hp", 70)) if rec else 70
            hp_groups[hp][0].append(rec_enc[i])
            hp_groups[hp][1].append(rec_dks[i])
            hp_groups[hp][2].append(_extract_relics(rec))
        for hp, (encs, dks_list, rels_list) in hp_groups.items():
            batches.append((encs, dks_list, rels_list, hp, len(encs)))

    else:
        cfg = curriculum.config
        enc = curriculum.sample_encounters(combats_per_gen)
        dks = [json.loads(curriculum.sample_deck_json(combat_idx=i))
               for i in range(combats_per_gen)]
        batches.append((enc, dks, [[] for _ in range(combats_per_gen)],
                        cfg.player_hp, combats_per_gen))

    return batches
