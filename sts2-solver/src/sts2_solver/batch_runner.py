"""Continuous batch runner: play games forever, reloading config each run.

Reads sts2_config.json before every game so code and config changes
are absorbed without restarting. Logs go to logs/gen{N}/ automatically.
Updates the dashboard after each completed run.

Supports multiple characters via config:
    "character": "Ironclad"              — single character
    "characters": ["Ironclad", "Silent"] — rotate between characters

Usage:
    python -m sts2_solver.batch_runner
    python -m sts2_solver.batch_runner --once       # single game, then exit
    python -m sts2_solver.batch_runner --character Silent  # override config
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from .runner import Runner, _load_env_from_mcp_json, DEFAULT_CHARACTER

CONFIG_PATH = Path(__file__).resolve().parents[3] / "sts2_config.json"
LOGS_ROOT = Path(__file__).resolve().parents[3] / "logs"


def load_config() -> dict:
    """Load config from sts2_config.json, with defaults."""
    defaults = {
        "gen": 1,
        "character": DEFAULT_CHARACTER,
        "model": "qwen3:8b",
        "poll_interval": 1.0,
    }
    try:
        with open(CONFIG_PATH, encoding="utf-8") as f:
            cfg = json.load(f)
        defaults.update(cfg)
    except FileNotFoundError:
        print(f"[warn] Config not found at {CONFIG_PATH}, using defaults")
    except json.JSONDecodeError as e:
        print(f"[warn] Config parse error: {e}, using defaults")
    return defaults


def run_one_game(cfg: dict, game_num: int) -> dict | None:
    """Play a single game with the given config. Returns result dict or None."""
    gen = cfg["gen"]
    logs_dir = LOGS_ROOT / f"gen{gen}"
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Set model from config
    if cfg.get("model"):
        os.environ["STS2_ADVISOR_MODEL"] = cfg["model"]

    character = cfg["character"]
    print(f"--- Game {game_num} (gen{gen}, {character}, {cfg['model']}) ---")
    t0 = time.time()
    try:
        runner = Runner(
            step_mode=False,
            dry_run=False,
            poll_interval=cfg["poll_interval"],
            character=cfg["character"],
            logs_dir=logs_dir,
            gen=f"gen{gen}",
        )
        runner.run()
        if getattr(runner, "_checkpoint_name", None):
            print(f"  Checkpoint: {runner._checkpoint_name}")
    except KeyboardInterrupt:
        raise
    except Exception as e:
        print(f"  Game crashed: {e}")
        import traceback
        traceback.print_exc()
        return None

    elapsed = time.time() - t0

    # Find the most recent log to get outcome
    log_files = sorted(logs_dir.glob("run_*.jsonl"), key=lambda p: p.stat().st_mtime)
    if not log_files:
        print("  No log file found")
        return None

    last_log = log_files[-1]
    outcome = "unknown"
    floor = "?"
    log_character = character
    with open(last_log, encoding="utf-8") as f:
        for line in f:
            event = json.loads(line.strip())
            if event.get("type") == "run_start" and event.get("character"):
                log_character = event["character"]
            if event.get("type") == "run_end":
                outcome = event.get("outcome", "unknown")
                floor = event.get("floor", "?")

    result = {
        "game": game_num,
        "gen": gen,
        "character": log_character,
        "outcome": outcome,
        "floor": floor,
        "time": elapsed,
        "log": last_log.name,
    }
    print(f"  Result: {outcome} on floor {floor} ({elapsed:.0f}s)")
    return result


def main():
    _load_env_from_mcp_json()

    parser = argparse.ArgumentParser(description="STS2 Continuous Runner")
    parser.add_argument("--once", action="store_true", help="Play one game then exit")
    parser.add_argument("--character", type=str, default=None,
                        help="Override character (e.g. Ironclad, Silent)")
    args = parser.parse_args()

    print("=== STS2 Continuous Runner ===")
    print(f"Config: {CONFIG_PATH}")
    print(f"Logs:   {LOGS_ROOT}")
    print()

    game_num = 0
    results = []
    char_index = 0

    while True:
        # Reload config each game
        cfg = load_config()
        game_num += 1

        # Resolve character: CLI flag > characters list > character field
        if args.character:
            cfg["character"] = args.character
        elif cfg.get("characters"):
            char_list = cfg["characters"]
            cfg["character"] = char_list[char_index % len(char_list)]
            char_index += 1

        result = run_one_game(cfg, game_num)
        if result:
            results.append(result)

        if args.once:
            break

        # Brief pause between games
        time.sleep(2.0)

    # Summary
    if results:
        total_time = sum(r["time"] for r in results)
        print(f"\n=== Summary ({len(results)} games, {total_time:.0f}s / {total_time / 60:.1f}min) ===")

        # Group by character
        chars = sorted(set(r.get("character", "?") for r in results))
        for char in chars:
            char_results = [r for r in results if r.get("character", "?") == char]
            floors = [r["floor"] for r in char_results if isinstance(r["floor"], int)]
            wins = sum(1 for r in char_results if r["outcome"] == "victory")
            avg_floor = sum(floors) / len(floors) if floors else 0
            print(f"  {char}: {len(char_results)} games, {wins} wins, avg floor {avg_floor:.1f}")


if __name__ == "__main__":
    main()
