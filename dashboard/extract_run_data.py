"""Extract per-run timeline data for the run viewer dashboard.

Parses JSONL log files and produces run_data.json with events,
network value estimates, HP, and floor progression.

Usage:
    python dashboard/extract_run_data.py              # all runs
    python dashboard/extract_run_data.py --latest 5   # last 5 runs
    python dashboard/extract_run_data.py --run-id XYZ # specific run
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

LOGS_ROOT = Path(__file__).resolve().parent.parent / "logs"
DATA_OUT = Path(__file__).resolve().parent / "run_data.json"


def find_run_files() -> list[Path]:
    """Find all run JSONL files, sorted by timestamp."""
    files = []
    for gen_dir in sorted(LOGS_ROOT.iterdir()):
        if gen_dir.is_dir() and gen_dir.name.startswith("gen"):
            for fp in sorted(gen_dir.glob("run_*.jsonl")):
                if not fp.name.startswith("run_TEST"):
                    files.append(fp)
    for fp in sorted(LOGS_ROOT.glob("run_*.jsonl")):
        if not fp.name.startswith("run_TEST"):
            files.append(fp)
    return files


def extract_run(fp: Path) -> dict | None:
    """Parse a single JSONL log into a timeline structure."""
    events = []
    meta = {}
    hp = None
    max_hp = None
    floor = 0
    combat_enemies = []

    try:
        with open(fp, encoding="utf-8") as f:
            for line in f:
                ev = json.loads(line)
                t = ev.get("type")

                if t == "run_start":
                    meta = {
                        "run_id": ev.get("run_id", fp.stem),
                        "character": ev.get("character", "?"),
                        "checkpoint": ev.get("checkpoint", "?"),
                        "file": str(fp),
                    }
                    hp = ev.get("hp")
                    max_hp = ev.get("max_hp")
                    floor = ev.get("floor", 1)
                    events.append({
                        "type": "run_start",
                        "floor": floor,
                        "hp": hp,
                        "max_hp": max_hp,
                    })

                elif t == "hp_change":
                    hp = ev.get("hp")
                    max_hp = ev.get("max_hp", max_hp)

                elif t == "combat_start":
                    floor = ev.get("floor", floor)
                    enemies = ev.get("enemies", [])
                    combat_enemies = [
                        e.get("name", "?") for e in enemies
                    ] if isinstance(enemies, list) else []
                    events.append({
                        "type": "combat_start",
                        "floor": floor,
                        "hp": hp,
                        "max_hp": max_hp,
                        "enemies": combat_enemies,
                    })

                elif t == "combat_turn":
                    entry = {
                        "type": "combat_turn",
                        "floor": floor,
                        "hp": hp,
                        "max_hp": max_hp,
                        "turn": ev.get("turn"),
                        "cards_played": ev.get("cards_played", []),
                    }
                    if "network_value" in ev:
                        entry["network_value"] = ev["network_value"]
                    events.append(entry)

                elif t == "combat_end":
                    hp_after = ev.get("hp_after", hp)
                    hp = hp_after
                    events.append({
                        "type": "combat_end",
                        "floor": floor,
                        "hp": hp_after,
                        "max_hp": max_hp,
                        "outcome": ev.get("outcome"),
                        "turns": ev.get("turns"),
                        "enemies": combat_enemies,
                    })

                elif t == "decision":
                    screen = ev.get("screen_type", "")
                    entry = {
                        "type": "decision",
                        "floor": floor,
                        "hp": hp,
                        "max_hp": max_hp,
                        "screen_type": screen,
                        "choice": _summarize_choice(ev.get("choice", {}), screen),
                    }
                    if "network_value" in ev:
                        entry["network_value"] = ev["network_value"]
                    events.append(entry)

                elif t == "run_end":
                    floor = ev.get("floor", floor)
                    hp = ev.get("hp", ev.get("final_hp", hp))
                    max_hp = ev.get("max_hp", max_hp)
                    events.append({
                        "type": "run_end",
                        "floor": floor,
                        "hp": hp,
                        "max_hp": max_hp,
                        "outcome": ev.get("outcome"),
                    })

    except Exception as e:
        print(f"  Error parsing {fp}: {e}", file=sys.stderr)
        return None

    if not meta:
        return None

    meta["events"] = events
    return meta


def _summarize_choice(choice: dict, screen_type: str) -> str:
    """Short human-readable summary of a decision."""
    action = choice.get("action", "?")
    reasoning = choice.get("reasoning", "")
    if "Network:" in reasoning:
        return reasoning.split("Network:")[-1].strip()
    if screen_type == "card_reward":
        return f"pick card" if "select" in action else "skip"
    return reasoning[:60] if reasoning else action


def main():
    files = find_run_files()

    # Parse CLI args
    run_id_filter = None
    latest_n = None
    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "--run-id" and i + 1 < len(args):
            run_id_filter = args[i + 1]
            i += 2
        elif args[i] == "--latest" and i + 1 < len(args):
            latest_n = int(args[i + 1])
            i += 2
        else:
            i += 1

    if latest_n:
        files = files[-latest_n:]

    runs = []
    for fp in files:
        run = extract_run(fp)
        if run is None:
            continue
        if run_id_filter and run["run_id"] != run_id_filter:
            continue
        runs.append(run)

    with open(DATA_OUT, "w", encoding="utf-8") as f:
        json.dump({"runs": runs}, f)

    print(f"Extracted {len(runs)} runs to {DATA_OUT}")


if __name__ == "__main__":
    main()
