"""Continuous batch runner: play games forever, reloading config each run.

Reads sts2_config.json before every game so code and config changes
are absorbed without restarting. Logs go to logs/gen{N}/ automatically.
Updates the dashboard after each completed run.

Usage:
    python -m sts2_solver.batch_runner
    python -m sts2_solver.batch_runner --once   # single game, then exit
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from .runner import Runner, _load_env_from_mcp_json, DEFAULT_CHARACTER

CONFIG_PATH = Path(__file__).resolve().parents[3] / "sts2_config.json"
LOGS_ROOT = Path(__file__).resolve().parents[3] / "logs"
DASHBOARD_SCRIPT = Path(__file__).resolve().parents[3] / "dashboard" / "update_data.py"


def load_config() -> dict:
    """Load config from sts2_config.json, with defaults."""
    defaults = {
        "gen": 1,
        "character": DEFAULT_CHARACTER,
        "model": "gpt-4o-mini",
        "local": False,
        "poll_interval": 1.0,
        "deploy_dashboard": True,
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


def update_dashboard(deploy: bool = True) -> None:
    """Run the dashboard update script."""
    if not DASHBOARD_SCRIPT.exists():
        return
    try:
        args = [sys.executable, str(DASHBOARD_SCRIPT)]
        if deploy:
            args.append("--deploy")
        subprocess.run(args, timeout=60, capture_output=True)
    except Exception as e:
        print(f"  [warn] Dashboard update failed: {e}")


def run_one_game(cfg: dict, game_num: int) -> dict | None:
    """Play a single game with the given config. Returns result dict or None."""
    gen = cfg["gen"]
    logs_dir = LOGS_ROOT / f"gen{gen}"
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Set environment for model selection
    if cfg["local"]:
        os.environ["STS2_ADVISOR_BASE_URL"] = "http://localhost:11434/v1"
        os.environ.setdefault("STS2_ADVISOR_MODEL", "qwen3:8b")
    else:
        os.environ.pop("STS2_ADVISOR_BASE_URL", None)
    if cfg.get("model"):
        os.environ["STS2_ADVISOR_MODEL"] = cfg["model"]

    print(f"--- Game {game_num} (gen{gen}, {cfg['model']}) ---")
    t0 = time.time()
    try:
        runner = Runner(
            step_mode=False,
            dry_run=False,
            poll_interval=cfg["poll_interval"],
            character=cfg["character"],
            logs_dir=logs_dir,
        )
        runner.run()
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
    with open(last_log, encoding="utf-8") as f:
        for line in f:
            event = json.loads(line.strip())
            if event.get("type") == "run_end":
                outcome = event.get("outcome", "unknown")
                floor = event.get("floor", "?")

    result = {
        "game": game_num,
        "gen": gen,
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
    args = parser.parse_args()

    print("=== STS2 Continuous Runner ===")
    print(f"Config: {CONFIG_PATH}")
    print(f"Logs:   {LOGS_ROOT}")
    print()

    game_num = 0
    results = []

    while True:
        # Reload config each game
        cfg = load_config()
        game_num += 1

        result = run_one_game(cfg, game_num)
        if result:
            results.append(result)

            # Update dashboard
            deploy = cfg.get("deploy_dashboard", True)
            print("  Updating dashboard...", end=" ", flush=True)
            update_dashboard(deploy=deploy)
            print("done")

        if args.once:
            break

        # Brief pause between games
        time.sleep(2.0)

    # Summary
    if results:
        floors = [r["floor"] for r in results if isinstance(r["floor"], int)]
        avg_floor = sum(floors) / len(floors) if floors else 0
        total_time = sum(r["time"] for r in results)
        print(f"\n=== Summary ({len(results)} games) ===")
        print(f"Avg floor: {avg_floor:.1f}")
        print(f"Total time: {total_time:.0f}s ({total_time / 60:.1f}min)")


if __name__ == "__main__":
    main()
