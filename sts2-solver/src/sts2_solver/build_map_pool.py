"""Extract real game maps from logs for use in self-play training.

Parses map_revealed events to collect full map graphs (nodes + edges).
Self-play uses these instead of generating synthetic maps.

Usage:
    python -m sts2_solver.build_map_pool [logs_dir]
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

from .replay_extractor import _parse_events

log = logging.getLogger(__name__)


def _extract_maps(logs_dir: Path) -> list[dict]:
    """Extract map graphs from all JSONL logs.

    Each map is stored as: {nodes: [{row, col, node_type, children}], rows, cols}
    """
    maps: list[dict] = []

    for path in sorted(logs_dir.rglob("run_*.jsonl")):
        events = _parse_events(path)
        for event in events:
            if event.get("type") != "map_revealed":
                continue
            m = event.get("map", {})
            nodes = m.get("nodes", [])
            if not nodes:
                continue

            # Strip to just what self-play needs
            clean_nodes = []
            for n in nodes:
                clean_nodes.append({
                    "row": n["row"],
                    "col": n["col"],
                    "node_type": n["node_type"],
                    "children": n.get("children", []),
                })
            maps.append({
                "rows": m.get("rows", 16),
                "cols": m.get("cols", 7),
                "nodes": clean_nodes,
            })

    # Deduplicate by node layout (same map can appear in multiple runs
    # if the game reuses seeds)
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


def main(logs_dir: Path | None = None) -> None:
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

    print(f"Map pool: {len(merged)} unique maps saved to {pool_path}")
    # Show summary
    for i, m in enumerate(merged[:5]):
        from collections import Counter
        types = Counter(n["node_type"] for n in m["nodes"])
        print(f"  Map {i}: {dict(types)}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    dir_arg = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    main(dir_arg)
