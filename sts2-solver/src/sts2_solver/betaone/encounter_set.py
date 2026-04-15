"""Encounter Sets: immutable, versioned collections of combat encounters.

An encounter set is a flat JSONL file where each line is a fully frozen
combat encounter: enemies, deck, player HP, relics. Used for both
training and benchmarking.

Two generators produce encounter sets:
  1. Package generator: archetypes -> random decks -> calibrate HP -> freeze
  2. Live game recorder: death encounters -> calibrate HP -> freeze

Format (one JSON line per encounter):
  {"enemies": ["BYRDONIS"], "deck": [{card_dict}, ...], "hp": 35, "relics": []}
"""

from __future__ import annotations

import hashlib
import json
import os
import random as stdlib_random
from pathlib import Path

import yaml

from .paths import BENCHMARK_DIR

ENCOUNTER_SETS_DIR = BENCHMARK_DIR / "encounter_sets"


def _content_hash(data: str) -> str:
    return hashlib.sha256(data.encode("utf-8")).hexdigest()[:12]


def load_encounter_set(es_id: str) -> list[dict]:
    """Load an encounter set by ID or friendly name. Returns list of encounter dicts."""
    jsonl_path = _resolve_path(es_id, ".jsonl")
    if jsonl_path is None:
        raise FileNotFoundError(f"Encounter set not found: {es_id}")
    with open(jsonl_path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def load_encounter_set_meta(es_id: str) -> dict | None:
    """Load metadata YAML for an encounter set."""
    yaml_path = _resolve_path(es_id, ".yaml")
    if yaml_path is None:
        return None
    with open(yaml_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _resolve_path(es_id: str, ext: str) -> Path | None:
    """Find an encounter set file by ID or friendly name."""
    ENCOUNTER_SETS_DIR.mkdir(parents=True, exist_ok=True)
    # Try direct match
    path = ENCOUNTER_SETS_DIR / f"{es_id}{ext}"
    if path.exists():
        return path
    # Try with es- prefix
    path = ENCOUNTER_SETS_DIR / f"es-{es_id}{ext}"
    if path.exists():
        return path
    # Try resolving friendly name from YAML (might point to a different ID)
    yaml_path = ENCOUNTER_SETS_DIR / f"{es_id}.yaml"
    if yaml_path.exists():
        with open(yaml_path, encoding="utf-8") as f:
            meta = yaml.safe_load(f)
        real_id = meta.get("encounter_set_id", es_id)
        path = ENCOUNTER_SETS_DIR / f"{real_id}{ext}"
        if path.exists():
            return path
    return None


def save_encounter_set(
    name: str,
    encounters: list[dict],
    source: dict | None = None,
) -> str:
    """Save an encounter set. Returns the encounter_set_id.

    Args:
        name: friendly name
        encounters: list of {enemies, deck, hp, relics} dicts
        source: metadata about how the set was generated
    """
    ENCOUNTER_SETS_DIR.mkdir(parents=True, exist_ok=True)

    # Write JSONL
    lines = [json.dumps(enc, separators=(",", ":")) for enc in encounters]
    content = "\n".join(lines) + "\n" if lines else ""
    es_id = f"es-{_content_hash(content)}"

    jsonl_path = ENCOUNTER_SETS_DIR / f"{es_id}.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        f.write(content)

    # Compute stats
    hps = [enc.get("hp", 70) for enc in encounters]
    avg_hp = sum(hps) / len(hps) if hps else 0

    # Write metadata
    meta = {
        "encounter_set_id": es_id,
        "name": name,
        "encounter_count": len(encounters),
        "avg_hp": round(avg_hp, 1),
        "source": source,
    }
    yaml_path = ENCOUNTER_SETS_DIR / f"{es_id}.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(meta, f, default_flow_style=False, sort_keys=False)

    # Friendly name alias
    if name != es_id:
        friendly_path = ENCOUNTER_SETS_DIR / f"{name}.yaml"
        with open(friendly_path, "w", encoding="utf-8") as f:
            yaml.dump(meta, f, default_flow_style=False, sort_keys=False)
        # Also write a friendly JSONL symlink (copy on Windows)
        friendly_jsonl = ENCOUNTER_SETS_DIR / f"{name}.jsonl"
        if not friendly_jsonl.exists():
            import shutil
            shutil.copy2(str(jsonl_path), str(friendly_jsonl))

    return es_id


def list_encounter_sets() -> list[dict]:
    """List all saved encounter sets."""
    if not ENCOUNTER_SETS_DIR.exists():
        return []
    results = []
    seen_ids = set()
    for f in sorted(ENCOUNTER_SETS_DIR.iterdir()):
        if f.suffix != ".yaml" or f.stem.startswith("_"):
            continue
        with open(f, encoding="utf-8") as fh:
            meta = yaml.safe_load(fh)
        if not meta or "encounter_set_id" not in meta:
            continue
        es_id = meta["encounter_set_id"]
        if es_id in seen_ids:
            continue
        seen_ids.add(es_id)
        results.append(meta)
    return results


def sample_encounters(encounters: list[dict], n: int, rng=None) -> list[dict]:
    """Sample n encounters from a set (with replacement if needed)."""
    if rng is None:
        rng = stdlib_random.Random()
    if not encounters:
        return []
    return [rng.choice(encounters) for _ in range(n)]
