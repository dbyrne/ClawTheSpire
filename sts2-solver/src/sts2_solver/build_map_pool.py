"""Extract real game maps from logs for use in self-play training.

Parses map_revealed events to collect full map graphs (nodes + edges),
tagged with their act ID (inferred from the first combat's enemies).

Usage:
    python -m sts2_solver.build_map_pool [logs_dir]
"""

from __future__ import annotations

import json
import logging
import sys
from collections import Counter
from pathlib import Path

from .replay_extractor import _parse_events

log = logging.getLogger(__name__)


def _infer_act(events: list[dict]) -> str | None:
    """Infer the act from the first combat's enemy names.

    Matches enemy IDs against encounter data to determine which act
    the run is in.
    """
    from .simulator import _ensure_data_loaded, _ENCOUNTERS_BY_ID
    _ensure_data_loaded()

    # Build monster -> act mapping
    monster_acts: dict[str, set[str]] = {}
    for enc in _ENCOUNTERS_BY_ID.values():
        act = enc.get("act", "")
        if not act:
            continue
        for m in enc.get("monsters", []):
            mid = m["id"]
            if mid not in monster_acts:
                monster_acts[mid] = set()
            monster_acts[mid].add(act)

    # Find first combat and check enemies
    for ev in events:
        if ev.get("type") != "combat_start":
            continue
        enemies = ev.get("enemies", [])
        for e in enemies:
            name = e.get("name", "")
            mid = name.upper().replace(" ", "_").replace("(", "").replace(")", "")
            acts = monster_acts.get(mid, set())
            if len(acts) == 1:
                return acts.pop()
        break  # Only check first combat

    return None


# Normalize act names to act IDs
_ACT_NAME_TO_ID = {
    "Act 1 - Overgrowth": "OVERGROWTH",
    "Overgrowth": "OVERGROWTH",
    "Underdocks": "UNDERDOCKS",
    "Act 2 - Hive": "HIVE",
    "Hive": "HIVE",
    "Act 3 - Glory": "GLORY",
    "Glory": "GLORY",
}


def _extract_maps(logs_dir: Path) -> list[dict]:
    """Extract map graphs from all JSONL logs, tagged with act_id.

    Each map is stored as: {act_id, nodes: [{row, col, node_type, children}], rows, cols}
    """
    maps: list[dict] = []

    for path in sorted(logs_dir.rglob("run_*.jsonl")):
        events = _parse_events(path)

        act_name = _infer_act(events)
        act_id = _ACT_NAME_TO_ID.get(act_name) if act_name else None

        for event in events:
            if event.get("type") != "map_revealed":
                continue
            m = event.get("map", {})
            nodes = m.get("nodes", [])
            if not nodes:
                continue

            clean_nodes = []
            for n in nodes:
                clean_nodes.append({
                    "row": n["row"],
                    "col": n["col"],
                    "node_type": n["node_type"],
                    "children": n.get("children", []),
                })
            entry = {
                "rows": m.get("rows", 16),
                "cols": m.get("cols", 7),
                "nodes": clean_nodes,
            }
            if act_id:
                entry["act_id"] = act_id
            maps.append(entry)

    # Deduplicate by node layout
    seen = set()
    unique: list[dict] = []
    for m in maps:
        key = tuple((n["row"], n["col"], n["node_type"]) for n in m["nodes"])
        if key not in seen:
            seen.add(key)
            unique.append(m)

    return unique


def _default_pool_path() -> Path:
    return Path(__file__).resolve().parent / "map_pool.json"


def main(logs_dir: Path | None = None) -> int:
    if logs_dir is None:
        logs_dir = Path(__file__).resolve().parents[3] / "logs"

    maps = _extract_maps(logs_dir)
    pool_path = _default_pool_path()

    # Merge with existing pool
    existing: list[dict] = []
    if pool_path.exists():
        with open(pool_path, encoding="utf-8") as f:
            existing = json.load(f)

    # Deduplicate merged set
    seen = set()
    merged: list[dict] = []
    for m in existing + maps:
        key = tuple((n["row"], n["col"], n["node_type"]) for n in m["nodes"])
        if key not in seen:
            seen.add(key)
            merged.append(m)

    with open(pool_path, "w", encoding="utf-8") as f:
        json.dump(merged, f)

    # Summary by act
    act_counts = Counter(m.get("act_id", "unknown") for m in merged)
    total = len(merged)
    print(f"Map pool: {total} unique maps saved to {pool_path}")
    for act, count in sorted(act_counts.items()):
        print(f"  {act}: {count} maps")

    return total


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    dir_arg = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    main(dir_arg)
