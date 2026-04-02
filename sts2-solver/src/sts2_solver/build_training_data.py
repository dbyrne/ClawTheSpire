"""Build fine-tuning dataset from game logs.

Reads JSONL game logs, extracts advisor decisions with their game state
context, labels quality based on run outcome, and outputs chat-format
JSONL suitable for fine-tuning.

Usage:
    python -m sts2_solver.build_training_data
    python -m sts2_solver.build_training_data --min-floor 10 --out training.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .advisor_prompts import SYSTEM_PROMPT


LOGS_DIR = Path(os.environ.get(
    "STS2_LOGS_DIR",
    Path(__file__).resolve().parents[3] / "logs",
))


def load_run(path: Path) -> list[dict]:
    """Load all events from a JSONL log file."""
    events = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events


def score_run(events: list[dict]) -> float:
    """Score a run for training data quality.

    Higher = better decisions to learn from.
    Returns 0.0-1.0.
    """
    run_end = next((e for e in events if e["type"] == "run_end"), None)
    if not run_end:
        return 0.0

    outcome = run_end.get("outcome", "")
    floor = run_end.get("floor", 0) or 0

    if outcome == "victory":
        return 1.0

    # Defeat: score by how far we got
    # Floor 17 = act 1 boss, floor 34 = act 2 boss, etc.
    return min(floor / 50.0, 0.8)


def extract_decisions(events: list[dict]) -> list[dict]:
    """Extract advisor decisions with their context."""
    decisions = []
    for e in events:
        if e["type"] != "decision":
            continue
        if e.get("source") != "advisor":
            continue

        choice = e.get("choice", {})
        if not choice.get("action"):
            continue

        decisions.append({
            "screen_type": e.get("screen_type", ""),
            "action": choice["action"],
            "option_index": choice.get("option_index"),
            "reasoning": choice.get("reasoning", ""),
            "latency_ms": e.get("latency_ms"),
            "user_prompt": e.get("user_prompt"),
        })

    return decisions


def build_training_example(decision: dict) -> dict | None:
    """Convert a decision into a chat-format training example.

    Returns a dict with {"messages": [...]} or None if invalid.
    Uses user_prompt from log if available (new logs), otherwise
    falls back to screen_type placeholder (old logs).
    """
    response = {
        "action": decision["action"],
        "option_index": decision["option_index"],
        "reasoning": decision["reasoning"],
    }

    user_content = decision.get("user_prompt") or f"[screen: {decision['screen_type']}]"
    has_full_prompt = "user_prompt" in decision and decision["user_prompt"]

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": json.dumps(response)},
        ],
        "_meta": {
            "screen_type": decision["screen_type"],
            "latency_ms": decision.get("latency_ms"),
            "has_full_prompt": has_full_prompt,
        },
    }


def build_dataset_v2(log_path: Path) -> list[dict]:
    """Build training examples by replaying the log and reconstructing prompts.

    This is the full pipeline: for each advisor decision, we reconstruct
    the game state at that point and build the exact prompt the advisor
    would have seen.
    """
    from .advisor_prompts import build_user_message, detect_screen_type
    from .game_data import load_game_data

    game_data = load_game_data()
    events = load_run(log_path)
    examples = []

    # We need the game state at each decision point. The log has diffs,
    # not full states. For v2, we need to either:
    # 1. Store full game state in decision events (best, requires log change)
    # 2. Replay diffs to reconstruct state (complex)
    #
    # For now, use the decision event's limited context.
    # TODO: Add game_state snapshot to decision log events for v2.

    for e in events:
        if e["type"] != "decision" or e.get("source") != "advisor":
            continue

        choice = e.get("choice", {})
        if not choice.get("action"):
            continue

        response = json.dumps({
            "action": choice["action"],
            "option_index": choice.get("option_index"),
            "reasoning": choice.get("reasoning", ""),
        })

        examples.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"[screen: {e.get('screen_type', '?')}]"},
                {"role": "assistant", "content": response},
            ],
            "_meta": {
                "screen_type": e.get("screen_type"),
                "latency_ms": e.get("latency_ms"),
            },
        })

    return examples


def main():
    parser = argparse.ArgumentParser(description="Build fine-tuning dataset from game logs")
    parser.add_argument(
        "--logs-dir", type=Path, default=LOGS_DIR,
        help=f"Directory containing game logs (default: {LOGS_DIR})",
    )
    parser.add_argument(
        "--out", type=Path, default=None,
        help="Output JSONL file (default: <logs-dir>/training_data.jsonl)",
    )
    parser.add_argument(
        "--min-floor", type=int, default=0,
        help="Only include runs that reached at least this floor",
    )
    parser.add_argument(
        "--min-score", type=float, default=0.0,
        help="Only include runs with quality score >= this (0.0-1.0)",
    )
    parser.add_argument(
        "--snapshot", action="store_true",
        help="Enable game state snapshots in decision logs (for future runs)",
    )
    args = parser.parse_args()

    out_path = args.out or args.logs_dir / "training_data.jsonl"

    log_files = sorted(args.logs_dir.glob("run_*.jsonl"))
    # Exclude test runs
    log_files = [f for f in log_files if "TEST" not in f.name]

    print(f"Found {len(log_files)} log files in {args.logs_dir}")

    all_examples = []
    run_stats = []

    for log_path in log_files:
        events = load_run(log_path)
        if not events:
            continue

        run_end = next((e for e in events if e["type"] == "run_end"), None)
        if not run_end:
            continue

        floor = run_end.get("floor", 0) or 0
        if floor < args.min_floor:
            continue

        run_score = score_run(events)
        if run_score < args.min_score:
            continue

        decisions = extract_decisions(events)
        examples = []
        for d in decisions:
            ex = build_training_example(d)
            if ex:
                ex["_meta"]["run_score"] = run_score
                ex["_meta"]["floor"] = floor
                ex["_meta"]["outcome"] = run_end.get("outcome")
                ex["_meta"]["log_file"] = log_path.name
                examples.append(ex)

        all_examples.extend(examples)
        run_stats.append({
            "file": log_path.name,
            "floor": floor,
            "outcome": run_end.get("outcome"),
            "score": run_score,
            "decisions": len(decisions),
            "examples": len(examples),
        })

    # Write output
    with open(out_path, "w", encoding="utf-8") as f:
        for ex in all_examples:
            f.write(json.dumps(ex, separators=(",", ":")) + "\n")

    print(f"\nRuns processed: {len(run_stats)}")
    print(f"Training examples: {len(all_examples)}")
    print(f"Output: {out_path}")
    print()

    # Per-screen breakdown
    from collections import Counter
    screen_counts = Counter(ex["_meta"]["screen_type"] for ex in all_examples)
    print("Examples by screen type:")
    for screen, count in screen_counts.most_common():
        print(f"  {screen}: {count}")

    print(f"\nRun quality distribution:")
    for stat in run_stats:
        print(f"  {stat['file']}: {stat['outcome']} floor {stat['floor']} "
              f"(score={stat['score']:.2f}, {stat['examples']} examples)")

    if args.snapshot:
        print("\n[NOTE] To collect full game state snapshots for v2 training data,")
        print("add STS2_LOG_SNAPSHOTS=1 to your environment before running games.")


if __name__ == "__main__":
    main()
