"""Extract real combat encounters from logs for use in self-play training.

Builds a pool of (floor, enemies) pairs from properly-bounded combat logs.
Self-play can sample from this pool instead of using static encounter data,
ensuring training sees the same enemy distribution as the real game.

Usage:
    python -m sts2_solver.build_encounter_pool [logs_dir]
"""

from __future__ import annotations

import json
import logging
import sys
from collections import Counter
from pathlib import Path

log = logging.getLogger(__name__)


def _to_monster_id(name: str) -> str:
    return name.upper().replace(" ", "_").replace("-", "_").replace("(", "").replace(")", "")


def extract_encounters(logs_dir: Path) -> list[dict]:
    """Extract encounter data from all properly-bounded combat logs.

    Each encounter is a single combat with one set of enemies.
    Returns list of {floor, enemies, outcome, turns}.
    """
    from .replay_extractor import extract_all_runs

    runs = extract_all_runs(logs_dir)
    encounters: list[dict] = []

    for run in runs:
        for combat in run.combats:
            # Get enemy IDs from the combat_start event
            enemy_ids = []
            for e in combat.enemies:
                name = e.get("name", "")
                if name:
                    enemy_ids.append(_to_monster_id(name))

            if not enemy_ids:
                # Fall back to first snapshot if combat_start had no enemies
                if combat.turns and combat.turns[0].snapshot:
                    for e in combat.turns[0].snapshot.enemies:
                        eid = e.get("id", "")
                        if eid:
                            enemy_ids.append(eid)

            if not enemy_ids:
                continue

            encounters.append({
                "floor": combat.floor,
                "enemies": sorted(enemy_ids),
                "outcome": combat.outcome,
                "turns": combat.turn_count,
            })

    return encounters


def _default_pool_path() -> Path:
    return Path(__file__).resolve().parent / "encounter_pool.json"


def main(logs_dir: Path | None = None) -> int:
    if logs_dir is None:
        logs_dir = Path(__file__).resolve().parents[3] / "logs"

    encounters = extract_encounters(logs_dir)
    pool_path = _default_pool_path()

    # Merge with existing pool
    existing: list[dict] = []
    if pool_path.exists():
        with open(pool_path, encoding="utf-8") as f:
            existing = json.load(f)

    # Deduplicate: same floor + same enemy set = same encounter type
    seen: set[tuple] = set()
    unique: list[dict] = []
    for enc in existing + encounters:
        key = (enc["floor"], tuple(enc["enemies"]))
        if key not in seen:
            seen.add(key)
            unique.append(enc)

    with open(pool_path, "w", encoding="utf-8") as f:
        json.dump(unique, f, indent=None, separators=(",", ":"))

    # Summary
    floor_counts = Counter(e["floor"] for e in unique)
    total = len(unique)
    print(f"Encounter pool: {total} unique encounters")
    for floor in sorted(floor_counts):
        count = floor_counts[floor]
        enemies_at_floor = set()
        for e in unique:
            if e["floor"] == floor:
                enemies_at_floor.update(e["enemies"])
        print(f"  Floor {floor:2d}: {count:3d} encounters "
              f"({len(enemies_at_floor)} unique enemies)")

    return total


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    dir_arg = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    main(dir_arg)
