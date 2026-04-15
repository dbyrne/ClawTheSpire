"""Shared data loading utilities for BetaOne training scripts.

Extracted from train.py and selfplay_train.py to eliminate duplication.
"""

from __future__ import annotations

import glob
import json
import os
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
