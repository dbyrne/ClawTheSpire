#!/usr/bin/env python3
"""Encounter Report — scan gameplay logs and compare against training data.

Reads all run log files (JSONL) from the logs/ directory and produces a
report showing:
  1. Every unique enemy group the bot has faced in real games
  2. Which of those are in the current training encounter list
  3. Which are NOT trained on (candidates for expanding training)
  4. How often each encounter was seen, and win/loss outcomes
  5. Per-profile (A/B) performance comparison when profile data is available

This script is READ-ONLY — it never modifies logs, training code, or any
other files. Safe to run at any time.

Usage:
    python3 encounter-report.py              # full report
    python3 encounter-report.py --untrained  # only show encounters NOT in training
    python3 encounter-report.py --compare    # show A vs B profile comparison
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


# ---------------------------------------------------------------------------
# All Act 1 encounter IDs currently used for self-play training.
# Updated to match the real game data IDs in encounters.json.
# Keep in sync with TRAINING_ENCOUNTERS in self_play.py.
# ---------------------------------------------------------------------------
TRAINING_ENCOUNTER_IDS = [
    # Weak
    "NIBBITS_WEAK", "SHRINKER_BEETLE_WEAK", "FUZZY_WURM_CRAWLER_WEAK", "SLIMES_WEAK",
    # Normal
    "NIBBITS_NORMAL", "SLIMES_NORMAL", "RUBY_RAIDERS_NORMAL", "INKLETS_NORMAL",
    "MAWLER_NORMAL", "CUBEX_CONSTRUCT_NORMAL", "VINE_SHAMBLER_NORMAL",
    "FLYCONID_NORMAL", "SNAPPING_JAXFRUIT_NORMAL", "FOGMOG_NORMAL",
    "OVERGROWTH_CRAWLERS", "SLITHERING_STRANGLER_NORMAL",
    # Elites
    "BYRDONIS_ELITE", "BYGONE_EFFIGY_ELITE", "PHROG_PARASITE_ELITE",
    # Bosses
    "VANTOM_BOSS", "CEREMONIAL_BEAST_BOSS", "THE_KIN_BOSS",
]

GAME_DATA_PATH = Path(__file__).parent / "STS2-Agent" / "mcp_server" / "data" / "eng"
LOGS_DIR = Path(__file__).parent / "logs"


def load_encounter_signatures() -> dict[str, tuple[str, ...]]:
    """Build a mapping from encounter_id -> sorted tuple of monster names."""
    enc_path = GAME_DATA_PATH / "encounters.json"
    if not enc_path.exists():
        return {}

    encounters = json.loads(enc_path.read_text())
    sigs: dict[str, tuple[str, ...]] = {}
    for enc in encounters:
        names = tuple(sorted(m["name"] for m in enc.get("monsters", [])))
        sigs[enc["id"]] = names
    return sigs


def scan_logs(logs_dir: Path) -> tuple[list[dict], dict[str, dict]]:
    """Read all JSONL log files and extract encounter events + run metadata.

    Returns:
      - combats: list of combat records with signature, floor, outcome, profile
      - runs: dict of run_id -> {profile, character, floor, outcome}
    """
    combats = []
    runs: dict[str, dict] = {}

    for log_file in sorted(logs_dir.glob("*.jsonl")):
        events = []
        for line in log_file.open():
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue

        # Extract profile from run_start event (if available)
        run_profile = None
        run_id = None
        run_character = None
        run_floor = 0
        for evt in events:
            if evt.get("type") == "run_start":
                run_profile = evt.get("config_profile")  # "a", "b", or None (old logs)
                run_id = evt.get("run_id")
                run_character = evt.get("character")
            if evt.get("type") == "run_end":
                run_floor = evt.get("floor", 0)
                if run_id:
                    runs[run_id] = {
                        "profile": run_profile,
                        "character": run_character,
                        "floor": run_floor,
                        "outcome": evt.get("outcome", "unknown"),
                        "log_file": log_file.name,
                    }

        # Walk through events and pair combat_start with combat_end
        current_combat_enemies: tuple[str, ...] | None = None
        current_floor: int | None = None

        for evt in events:
            if evt.get("type") == "combat_start":
                enemies = evt.get("enemies", [])
                current_combat_enemies = tuple(sorted(e["name"] for e in enemies))
                current_floor = evt.get("floor")

            elif evt.get("type") == "combat_end" and current_combat_enemies is not None:
                combats.append({
                    "signature": current_combat_enemies,
                    "floor": current_floor,
                    "outcome": evt.get("outcome", "unknown"),
                    "hp_before": evt.get("hp_before"),
                    "hp_after": evt.get("hp_after"),
                    "turns": evt.get("turns"),
                    "profile": run_profile,
                    "log_file": log_file.name,
                })
                current_combat_enemies = None

            elif evt.get("type") == "run_end" and current_combat_enemies is not None:
                combats.append({
                    "signature": current_combat_enemies,
                    "floor": current_floor,
                    "outcome": evt.get("outcome", "loss"),
                    "hp_before": None,
                    "hp_after": None,
                    "turns": None,
                    "profile": run_profile,
                    "log_file": log_file.name,
                })
                current_combat_enemies = None

    return combats, runs


def classify_encounters(
    combats: list[dict],
    enc_signatures: dict[str, tuple[str, ...]],
) -> dict[tuple[str, ...], dict]:
    """Group combats by enemy signature and classify as trained/untrained."""
    sig_to_id: dict[tuple[str, ...], str] = {}
    for enc_id, names in enc_signatures.items():
        sig_to_id[names] = enc_id

    # All Act 1 encounters are now in training
    training_sigs: set[tuple[str, ...]] = set()
    for enc_id in TRAINING_ENCOUNTER_IDS:
        if enc_id in enc_signatures:
            training_sigs.add(enc_signatures[enc_id])

    groups: dict[tuple[str, ...], dict] = defaultdict(lambda: {
        "count": 0,
        "wins": 0,
        "losses": 0,
        "encounter_id": None,
        "is_trained": False,
        "floors_seen": [],
        "avg_turns": 0,
        "total_turns": 0,
    })

    for combat in combats:
        sig = combat["signature"]
        g = groups[sig]
        g["count"] += 1
        if combat["outcome"] in ("win",):
            g["wins"] += 1
        else:
            g["losses"] += 1
        if combat["floor"] is not None:
            g["floors_seen"].append(combat["floor"])
        if combat["turns"]:
            g["total_turns"] += combat["turns"]

        if g["encounter_id"] is None and sig in sig_to_id:
            g["encounter_id"] = sig_to_id[sig]

        if sig in training_sigs:
            g["is_trained"] = True

    for g in groups.values():
        if g["wins"] + g["losses"] > 0 and g["total_turns"] > 0:
            g["avg_turns"] = round(g["total_turns"] / g["count"], 1)

    return dict(groups)


def print_report(groups: dict[tuple[str, ...], dict], untrained_only: bool = False):
    """Print a human-readable encounter report to the console."""
    trained = {sig: g for sig, g in groups.items() if g["is_trained"]}
    untrained = {sig: g for sig, g in groups.items() if not g["is_trained"]}

    total_combats = sum(g["count"] for g in groups.values())
    total_encounters = len(groups)

    print("=" * 70)
    print("  ENCOUNTER REPORT — ClawTheSpire Training Gap Analysis")
    print("=" * 70)
    print(f"  Total combats logged:     {total_combats}")
    print(f"  Unique enemy groups:      {total_encounters}")
    print(f"  Trained encounters seen:  {len(trained)}")
    print(f"  UNTRAINED encounters:     {len(untrained)}")
    print()

    if not untrained_only:
        print("-" * 70)
        print("  TRAINED encounters (in self-play training)")
        print("-" * 70)
        if trained:
            for sig, g in sorted(trained.items(), key=lambda x: -x[1]["count"]):
                enemy_str = " + ".join(sig)
                enc_id = g["encounter_id"] or "?"
                win_rate = (g["wins"] / g["count"] * 100) if g["count"] > 0 else 0
                print(f"  [{g['count']:2d}x] {enemy_str}")
                print(f"         ID: {enc_id}  |  Win: {g['wins']}/{g['count']} ({win_rate:.0f}%)"
                      f"  |  Avg turns: {g['avg_turns']}")
        else:
            print("  (none seen in logs)")
        print()

    print("-" * 70)
    print("  UNTRAINED encounters (not in self-play training)")
    print("-" * 70)
    if untrained:
        for sig, g in sorted(untrained.items(), key=lambda x: -x[1]["count"]):
            enemy_str = " + ".join(sig)
            enc_id = g["encounter_id"] or "(no matching game data ID)"
            win_rate = (g["wins"] / g["count"] * 100) if g["count"] > 0 else 0
            print(f"  [{g['count']:2d}x] {enemy_str}")
            print(f"         ID: {enc_id}  |  Win: {g['wins']}/{g['count']} ({win_rate:.0f}%)"
                  f"  |  Avg turns: {g['avg_turns']}")
    else:
        print("  All observed encounters are already trained!")

    print()
    print("=" * 70)


def print_profile_comparison(runs: dict[str, dict], combats: list[dict]):
    """Print a side-by-side comparison of profile A vs profile B performance."""

    # Separate runs by profile
    a_runs = [r for r in runs.values() if r.get("profile") == "a"]
    b_runs = [r for r in runs.values() if r.get("profile") == "b"]
    unknown_runs = [r for r in runs.values() if r.get("profile") not in ("a", "b")]

    # Separate combats by profile
    a_combats = [c for c in combats if c.get("profile") == "a"]
    b_combats = [c for c in combats if c.get("profile") == "b"]

    print("=" * 70)
    print("  A/B PROFILE COMPARISON")
    print("=" * 70)

    if not a_runs and not b_runs:
        print("  No profile data found in logs.")
        print("  Profile tracking was added recently — new runs will include it.")
        if unknown_runs:
            print(f"  ({len(unknown_runs)} runs from before profile tracking)")
        print()
        print("=" * 70)
        return

    # Helper to compute stats for a set of runs
    def run_stats(run_list: list[dict]) -> dict:
        if not run_list:
            return {"count": 0, "avg_floor": 0, "best_floor": 0, "wins": 0}
        floors = [r["floor"] for r in run_list]
        wins = sum(1 for r in run_list if r.get("outcome") == "victory")
        return {
            "count": len(run_list),
            "avg_floor": round(sum(floors) / len(floors), 1) if floors else 0,
            "best_floor": max(floors) if floors else 0,
            "wins": wins,
        }

    def combat_stats(combat_list: list[dict]) -> dict:
        if not combat_list:
            return {"total": 0, "wins": 0, "win_rate": 0}
        wins = sum(1 for c in combat_list if c["outcome"] == "win")
        return {
            "total": len(combat_list),
            "wins": wins,
            "win_rate": round(wins / len(combat_list) * 100, 1) if combat_list else 0,
        }

    a_rs = run_stats(a_runs)
    b_rs = run_stats(b_runs)
    a_cs = combat_stats(a_combats)
    b_cs = combat_stats(b_combats)

    # Side-by-side display
    col_w = 28
    print()
    print(f"  {'':24s} {'A (Champion)':>{col_w}s}  {'B (Challenger)':>{col_w}s}")
    print(f"  {'─' * 24} {'─' * col_w}  {'─' * col_w}")
    print(f"  {'Runs':24s} {str(a_rs['count']):>{col_w}s}  {str(b_rs['count']):>{col_w}s}")
    print(f"  {'Avg floor reached':24s} {str(a_rs['avg_floor']):>{col_w}s}  {str(b_rs['avg_floor']):>{col_w}s}")
    print(f"  {'Best floor reached':24s} {str(a_rs['best_floor']):>{col_w}s}  {str(b_rs['best_floor']):>{col_w}s}")
    print(f"  {'Run victories':24s} {str(a_rs['wins']):>{col_w}s}  {str(b_rs['wins']):>{col_w}s}")
    print(f"  {'Combats fought':24s} {str(a_cs['total']):>{col_w}s}  {str(b_cs['total']):>{col_w}s}")
    print(f"  {'Combat win rate':24s} {str(a_cs['win_rate']) + '%':>{col_w}s}  {str(b_cs['win_rate']) + '%':>{col_w}s}")

    # Floor difference callout
    if a_rs["count"] >= 2 and b_rs["count"] >= 2:
        delta = b_rs["avg_floor"] - a_rs["avg_floor"]
        direction = "BETTER" if delta > 0 else "WORSE" if delta < 0 else "SAME"
        print()
        sign = "+" if delta >= 0 else ""
        print(f"  Challenger is {direction}: {sign}{delta:.1f} floors vs champion")

    if unknown_runs:
        print()
        print(f"  Note: {len(unknown_runs)} older run(s) have no profile data")

    # Per-profile boss/elite breakdown if enough data
    a_bosses = [c for c in a_combats if c["floor"] and c["floor"] >= 15]
    b_bosses = [c for c in b_combats if c["floor"] and c["floor"] >= 15]
    if a_bosses or b_bosses:
        a_boss_cs = combat_stats(a_bosses)
        b_boss_cs = combat_stats(b_bosses)
        print()
        print(f"  {'Boss/late fights (f15+)':24s} "
              f"{'Win ' + str(a_boss_cs['wins']) + '/' + str(a_boss_cs['total']):>{col_w}s}  "
              f"{'Win ' + str(b_boss_cs['wins']) + '/' + str(b_boss_cs['total']):>{col_w}s}")

    print()
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Scan gameplay logs and report encounter coverage vs training data."
    )
    parser.add_argument(
        "--untrained", action="store_true",
        help="Only show encounters NOT in the training set.",
    )
    parser.add_argument(
        "--compare", action="store_true",
        help="Show A/B profile performance comparison.",
    )
    parser.add_argument(
        "--logs-dir", type=Path, default=LOGS_DIR,
        help="Path to the logs directory (default: ./logs).",
    )
    args = parser.parse_args()

    if not args.logs_dir.exists():
        print(f"No logs directory found at {args.logs_dir}")
        print("Run some games first to generate log data!")
        return

    log_files = list(args.logs_dir.glob("*.jsonl"))
    if not log_files:
        print(f"No log files found in {args.logs_dir}")
        return

    print(f"Scanning {len(log_files)} log file(s)...")
    print()

    enc_signatures = load_encounter_signatures()
    combats, runs = scan_logs(args.logs_dir)

    if args.compare:
        # Show A/B comparison only
        print_profile_comparison(runs, combats)
    else:
        # Show encounter report (optionally filtered)
        groups = classify_encounters(combats, enc_signatures)
        print_report(groups, untrained_only=args.untrained)
        # Always append A/B comparison if profile data exists
        print_profile_comparison(runs, combats)


if __name__ == "__main__":
    main()
