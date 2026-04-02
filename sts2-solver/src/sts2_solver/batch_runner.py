"""Batch runner: play N games and collect logs for fine-tuning.

Usage:
    python -m sts2_solver.batch_runner --games 50
    python -m sts2_solver.batch_runner --games 10 --local
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

from .runner import Runner, _load_env_from_mcp_json, DEFAULT_CHARACTER


def main():
    _load_env_from_mcp_json()
    parser = argparse.ArgumentParser(description="STS2 Batch Runner — play N games")
    parser.add_argument(
        "--games", type=int, default=10,
        help="Number of games to play (default: 10)",
    )
    parser.add_argument(
        "--character", type=str, default=DEFAULT_CHARACTER,
        help=f"Character to play (default: {DEFAULT_CHARACTER})",
    )
    parser.add_argument(
        "--local", action="store_true",
        help="Use local Ollama model instead of OpenAI API",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Override advisor model",
    )
    parser.add_argument(
        "--poll", type=float, default=1.0,
        help="Seconds between state polls (default: 1.0)",
    )
    args = parser.parse_args()

    if args.local:
        os.environ.setdefault("STS2_ADVISOR_BASE_URL", "http://localhost:11434/v1")
        os.environ.setdefault("STS2_ADVISOR_MODEL", "qwen3:8b")
    if args.model:
        os.environ["STS2_ADVISOR_MODEL"] = args.model

    logs_dir = Path(os.environ.get(
        "STS2_LOGS_DIR",
        Path(__file__).resolve().parents[3] / "logs",
    ))

    print(f"=== STS2 Batch Runner ===")
    print(f"Games: {args.games}")
    print(f"Character: {args.character}")
    print(f"Model: {os.environ.get('STS2_ADVISOR_MODEL', 'gpt-4o-mini')}")
    print(f"Logs: {logs_dir}")
    print()

    results = []
    for i in range(args.games):
        print(f"--- Game {i + 1}/{args.games} ---")
        t0 = time.time()
        try:
            runner = Runner(
                step_mode=False,
                dry_run=False,
                poll_interval=args.poll,
                character=args.character,
            )
            runner.run()
        except KeyboardInterrupt:
            print("\nStopped by user.")
            break
        except Exception as e:
            print(f"Game {i + 1} crashed: {e}")

        elapsed = time.time() - t0

        # Find the most recent log to get the outcome
        log_files = sorted(logs_dir.glob("run_*.jsonl"), key=lambda p: p.stat().st_mtime)
        if log_files:
            import json
            last_log = log_files[-1]
            outcome = "unknown"
            floor = "?"
            with open(last_log) as f:
                for line in f:
                    event = json.loads(line.strip())
                    if event.get("type") == "run_end":
                        outcome = event.get("outcome", "unknown")
                        floor = event.get("floor", "?")

            results.append({"game": i + 1, "outcome": outcome, "floor": floor, "time": elapsed, "log": last_log.name})
            print(f"  Result: {outcome} on floor {floor} ({elapsed:.0f}s)")
        else:
            print(f"  No log file found")

        # Brief pause between games to let the game reset
        if i < args.games - 1:
            time.sleep(2.0)

    # Summary
    print(f"\n=== Summary ({len(results)} games) ===")
    wins = sum(1 for r in results if r["outcome"] == "victory")
    defeats = sum(1 for r in results if r["outcome"] == "defeat")
    floors = [r["floor"] for r in results if isinstance(r["floor"], int)]
    avg_floor = sum(floors) / len(floors) if floors else 0
    total_time = sum(r["time"] for r in results)

    print(f"Wins: {wins}/{len(results)} ({100 * wins / len(results):.0f}%)" if results else "No games completed")
    print(f"Avg floor: {avg_floor:.1f}")
    print(f"Total time: {total_time:.0f}s ({total_time / 60:.1f}min)")
    print(f"Avg time/game: {total_time / len(results):.0f}s" if results else "")

    # Write summary to file
    summary_path = logs_dir / "batch_summary.jsonl"
    import json
    with open(summary_path, "a", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, default=str) + "\n")
    print(f"\nResults appended to {summary_path}")


if __name__ == "__main__":
    main()
