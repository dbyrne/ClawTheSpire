"""Scan all run logs and rebuild dashboard/data.json.

Usage:
    python dashboard/update_data.py          # scan + write data.json
    python dashboard/update_data.py --deploy  # also deploy to Vercel
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

LOGS_ROOT = Path(__file__).resolve().parent.parent / "logs"
DATA_OUT = Path(__file__).resolve().parent / "data.json"

# Generation directories in order (logs/ itself is the "current" gen)
GEN_DIRS = [
    ("gen1", LOGS_ROOT / "gen1"),
    ("gen2", LOGS_ROOT / "gen2"),
    ("gen3", LOGS_ROOT / "gen3"),
    ("gen4", LOGS_ROOT / "gen4"),
    ("gen5", LOGS_ROOT / "gen5"),
    ("gen6", LOGS_ROOT / "gen6"),
    ("gen7", LOGS_ROOT / "gen7"),
    ("gen8", LOGS_ROOT / "gen8"),
]

# Also scan logs/ root for current/unarchived runs
CURRENT_GEN = "current"


def scan_runs() -> list[dict]:
    """Scan all gen directories and logs/ root for completed runs."""
    runs = []

    for gen_name, gen_dir in GEN_DIRS:
        if not gen_dir.is_dir():
            continue
        for fp in sorted(gen_dir.glob("run_*.jsonl")):
            if fp.name.startswith("run_TEST"):
                continue
            run = _parse_run(fp, gen_name)
            if run:
                runs.append(run)

    # Scan logs/ root for current gen
    for fp in sorted(LOGS_ROOT.glob("run_*.jsonl")):
        if fp.name.startswith("run_TEST"):
            continue
        run = _parse_run(fp, CURRENT_GEN)
        if run:
            runs.append(run)

    # Sort by timestamp from filename (run_XXXX_YYYYMMDD_HHMMSS.jsonl)
    runs.sort(key=lambda r: r["ts"])
    # Assign run numbers
    for i, r in enumerate(runs):
        r["run_number"] = i + 1

    return runs


def _parse_run(fp: Path, gen: str) -> dict | None:
    """Extract run metadata from a JSONL log file."""
    first = last = None
    try:
        with open(fp, encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                if d.get("type") == "run_start":
                    first = d
                if d.get("type") == "run_end":
                    last = d
    except Exception:
        return None

    if not first or not last:
        return None

    # Extract timestamp from filename
    parts = fp.stem.split("_")
    ts = "_".join(parts[-2:]) if len(parts) >= 3 else fp.stem

    return {
        "run_id": first.get("run_id", fp.stem),
        "gen": gen,
        "ts": ts,
        "floor": last.get("floor", 0),
        "model": first.get("advisor_model", "unknown"),
        "character": first.get("character", "?"),
    }


def build_data(runs: list[dict]) -> dict:
    """Build the chart data structure."""
    # Find gen boundaries (last run number in each gen)
    gen_boundaries = {}
    for r in runs:
        gen = r["gen"]
        gen_boundaries[gen] = r["run_number"]

    return {
        "runs": [
            {
                "run_number": r["run_number"],
                "floor": r["floor"],
                "gen": r["gen"],
                "model": r["model"],
                "run_id": r["run_id"],
            }
            for r in runs
        ],
        "gen_boundaries": gen_boundaries,
    }


def main():
    runs = scan_runs()
    data = build_data(runs)

    with open(DATA_OUT, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print(f"Wrote {len(runs)} runs to {DATA_OUT}")

    if "--deploy" in sys.argv:
        dashboard_dir = Path(__file__).resolve().parent
        print("Deploying to Vercel...")
        subprocess.run(
            ["npx", "vercel", "--prod", "--yes"],
            cwd=str(dashboard_dir),
            check=True,
        )
        print("Deployed!")


if __name__ == "__main__":
    main()
