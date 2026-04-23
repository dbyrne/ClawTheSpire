"""Shared data loading and training utilities for BetaOne.

Used by both train.py (PPO) and selfplay_train.py (MCTS self-play).
"""

from __future__ import annotations

import glob
import json
import os
import random as stdlib_random
from collections import defaultdict

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


def setup_training_data(encounter_set_id: str) -> dict:
    """Load a frozen encounter set for training.

    Returns a dict with the loaded encounter set and its id/name for logging.
    """
    if not encounter_set_id:
        raise ValueError(
            "encounter_set_id is required — training now runs exclusively against "
            "frozen encounter sets. Set data.encounter_set in your experiment yaml."
        )
    from .encounter_set import load_encounter_set, load_encounter_set_meta
    es = load_encounter_set(encounter_set_id)
    meta = load_encounter_set_meta(encounter_set_id) or {}
    name = meta.get("name", encounter_set_id)
    print(f"Encounter set: {name} ({len(es)} encounters, avg HP {meta.get('avg_hp', '?')})")
    return {
        "encounter_set_id": encounter_set_id,
        "encounter_set_name": name,
        "encounter_set": es,
    }


def sample_combat_batches(
    encounter_set: list[dict],
    combats_per_gen: int,
    gen: int,
) -> list[tuple[list, list, list, list, int, int]]:
    """Sample encounters from the set and group into batches.

    Returns list of (encounters, decks, relics, potions, hp, count) tuples.

    Grouped by (hp, potions_inventory) so that each Rust engine call shares a
    single potion list (the FFI takes one `potions_json` per batch). Encounters
    without potions or with identical potion inventories collapse to fewer
    groups; worst case is one Rust call per unique inventory.
    """
    import json as _json
    from .encounter_set import sample_encounters
    rng = stdlib_random.Random(gen * 7919)
    sampled = sample_encounters(encounter_set, combats_per_gen, rng=rng)

    # Key = (hp, serialized-potions). Serialize so the dict key is hashable
    # and two encounters with identical potion inventories group together.
    groups: dict[tuple[int, str], tuple[list, list, list, list]] = defaultdict(
        lambda: ([], [], [], [])
    )
    for enc in sampled:
        hp = enc.get("hp", 70)
        potions = enc.get("potions", []) or []
        key = (hp, _json.dumps(potions, sort_keys=True))
        groups[key][0].append(enc["enemies"])
        groups[key][1].append(enc["deck"])
        groups[key][2].append(enc.get("relics", []))
        groups[key][3].append(potions)
    # Each batch's shared potions is just the first encounter's list (all
    # encounters in a group have identical potion inventories by construction).
    return [
        (encs, dks, rels, pots[0] if pots else [], hp, len(encs))
        for (hp, _key), (encs, dks, rels, pots) in groups.items()
    ]
