"""Experiment dashboard TUI.

Shows all experiments with live status, benchmarks, and eval results.

Usage:
    python -m sts2_solver.betaone.tui
    sts2-experiment dashboard
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .paths import EXPERIMENTS_DIR


def _load_json(path: Path):
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _load_jsonl_last(path: Path):
    try:
        with open(path) as f:
            lines = [l for l in f if l.strip()]
            return json.loads(lines[-1]) if lines else None
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _load_jsonl_all(path: Path, tail: int = 10):
    try:
        with open(path) as f:
            lines = [l for l in f if l.strip()]
            return [json.loads(l) for l in lines[-tail:]]
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def _format_time_ago(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s ago"
    elif seconds < 3600:
        return f"{seconds/60:.0f}m ago"
    elif seconds < 86400:
        return f"{seconds/3600:.1f}h ago"
    else:
        return f"{seconds/86400:.1f}d ago"


def _resolve_ts_name(ts_id: str) -> str:
    """Resolve a training set ID to its friendly name."""
    from .paths import BENCHMARK_DIR
    ts_path = BENCHMARK_DIR / "training_sets" / f"{ts_id}.yaml"
    if ts_path.exists():
        try:
            import yaml
            with open(ts_path) as f:
                data = yaml.safe_load(f)
            return data.get("name", ts_id)
        except Exception:
            pass
    return ts_id


def _status_text(age: float) -> Text:
    if age < 120:
        return Text("RUNNING", style="bold green")
    elif age < 600:
        return Text("STALLED?", style="bold yellow")
    else:
        return Text("stopped", style="dim")


def _collect_experiments() -> list[dict]:
    """Gather data for all experiments."""
    if not EXPERIMENTS_DIR.exists():
        return []

    experiments = []
    for d in sorted(EXPERIMENTS_DIR.iterdir(), key=lambda p: p.stat().st_mtime):
        if d.name.startswith("_") or not d.is_dir():
            continue
        config_path = d / "config.yaml"
        if not config_path.exists():
            continue

        import yaml
        with open(config_path, encoding="utf-8") as f:
            config = yaml.safe_load(f)

        progress = _load_json(d / "betaone_progress.json")
        history = _load_jsonl_all(d / "betaone_history.jsonl", tail=20)
        eval_result = _load_jsonl_last(d / "benchmarks" / "eval.jsonl")
        benchmarks = _load_jsonl_all(d / "benchmarks" / "results.jsonl", tail=10)

        experiments.append({
            "name": config.get("name", d.name),
            "method": config.get("method", "?"),
            "description": config.get("description", ""),
            "parent": config.get("parent"),
            "arch": config.get("architecture", {}),
            "training": config.get("training", {}),
            "data": config.get("data", {}),
            "progress": progress,
            "history": history,
            "eval": eval_result,
            "benchmarks": benchmarks,
            "dir": d,
        })

    return experiments


def build_dashboard(experiments: list[dict]) -> Group:
    """Build the full dashboard layout."""
    parts = []

    # === Experiment overview table ===
    overview = Table(title="Experiments", expand=True, show_lines=False)
    overview.add_column("Name", style="cyan", max_width=28)
    overview.add_column("Method", max_width=10)
    overview.add_column("Params", justify="right", max_width=8)
    overview.add_column("Status", max_width=10)
    overview.add_column("Gen", justify="right", max_width=7)
    overview.add_column("Train WR", justify="right", max_width=8)
    overview.add_column("Best", justify="right", max_width=7)
    overview.add_column("Eval", justify="right", max_width=7)
    overview.add_column("ET Avg", justify="right", max_width=7)
    overview.add_column("Buffer", justify="right", max_width=10)
    overview.add_column("Training Set", max_width=18)

    for exp in experiments:
        p = exp["progress"]
        ev = exp["eval"]
        arch = exp["arch"]

        if p:
            age = time.time() - p.get("timestamp", 0)
            status = _status_text(age)
            gen = str(p.get("gen", 0))
            total = p.get("num_generations", "?")
            gen_str = f"{gen}/{total}" if total != "?" else gen
            wr = f"{p.get('win_rate', 0):.1%}"
            best = f"{p.get('best_win_rate', 0):.1%}"
        else:
            status = Text("new", style="dim")
            gen_str = "-"
            wr = "-"
            best = "-"

        params = arch.get("total_params")
        params_str = f"{params//1000}K" if isinstance(params, int) else "-"

        eval_str = f"{ev['score']:.0%}" if ev and "score" in ev else "-"
        et_str = f"{ev['end_turn_avg']:.0%}" if ev and ev.get("end_turn_avg") else "-"

        ts = exp["data"].get("training_set", "")
        # Resolve friendly name from training set YAML
        ts_str = "-"
        if ts:
            ts_str = _resolve_ts_name(ts)

        if exp["method"] == "ppo":
            method = "PPO"
        else:
            sims = exp.get("training", {}).get("mcts", {}).get("num_sims", "?")
            method = f"MCTS-{sims}"

        # Replay buffer (MCTS only)
        if exp["method"] != "ppo" and p:
            buf_size = p.get("buffer_size", 0)
            buf_cap = exp.get("training", {}).get("mcts", {}).get("replay_capacity", 0)
            if buf_cap:
                buf_str = f"{buf_size//1000}K/{buf_cap//1000}K"
            else:
                buf_str = f"{buf_size//1000}K" if buf_size else "-"
        else:
            buf_str = "-"

        overview.add_row(exp["name"], method, params_str, status,
                         gen_str, wr, best, eval_str, et_str, buf_str, ts_str)

    parts.append(overview)

    # === Benchmark results table ===
    bench_rows = []
    for exp in experiments:
        for b in exp.get("benchmarks", []):
            suite = b.get("suite", "?")
            # Shorten suite ID
            if "-" in suite:
                suite_short = suite.split("-")[0] + "-" + suite.split("-")[1][:6]
            else:
                suite_short = suite[:14]
            bench_rows.append({
                "name": exp["name"],
                "suite": suite_short,
                "mode": b.get("mode", "?"),
                "sims": b.get("mcts_sims", 0),
                "wr": b.get("win_rate", 0),
                "ci_lo": b.get("ci95_lo", 0),
                "ci_hi": b.get("ci95_hi", 0),
                "n": b.get("games", 0),
            })

    if bench_rows:
        bench_table = Table(title="Latest Benchmarks", expand=True, show_lines=False)
        bench_table.add_column("Experiment", style="cyan", max_width=24)
        bench_table.add_column("Suite", max_width=16)
        bench_table.add_column("Mode", max_width=8)
        bench_table.add_column("WR", justify="right", max_width=7)
        bench_table.add_column("95% CI", justify="right", max_width=16)
        bench_table.add_column("N", justify="right", max_width=6)

        # Show most recent per experiment (last 2-3 per exp)
        seen = {}
        for row in reversed(bench_rows):
            key = (row["name"], row["suite"], row["mode"])
            if key not in seen:
                seen[key] = row
        for key in sorted(seen.keys()):
            row = seen[key]
            mode_str = f"{row['mode']}" + (f"-{row['sims']}" if row['sims'] > 0 else "")
            ci = f"[{row['ci_lo']:.1%}, {row['ci_hi']:.1%}]"
            bench_table.add_row(
                row["name"], row["suite"], mode_str,
                f"{row['wr']:.1%}", ci, str(row["n"]),
            )
        parts.append(bench_table)

    # === Active training sparklines ===
    for exp in experiments:
        p = exp["progress"]
        if not p:
            continue
        age = time.time() - p.get("timestamp", 0)
        if age > 600:
            continue  # skip stopped experiments

        history = exp["history"]
        if len(history) < 3:
            continue

        # Mini sparkline of win rate
        wrs = [r.get("win_rate", 0) for r in history]
        pls = [r.get("policy_loss", 0) for r in history]
        vls = [r.get("value_loss", 0) for r in history]
        gen_times = [r.get("gen_time", 0) for r in history]

        spark = Text()
        spark.append(f"  {exp['name']}", style="cyan")
        spark.append(f"  WR: ", style="dim")
        for wr in wrs[-15:]:
            if wr >= 0.6:
                spark.append("█", style="green")
            elif wr >= 0.4:
                spark.append("▆", style="yellow")
            else:
                spark.append("▂", style="red")
        spark.append(f" {wrs[-1]:.0%}", style="bold")
        spark.append(f"  pi={pls[-1]:.3f}  v={vls[-1]:.3f}", style="dim")
        spark.append(f"  {gen_times[-1]:.0f}s/gen", style="dim")
        parts.append(spark)

    # === Footer ===
    # === Training sets section ===
    from .training_set import list_training_sets
    ts_list = list_training_sets()
    if ts_list:
        ts_table = Table(title="Training Sets", expand=True, show_lines=False)
        ts_table.add_column("Name", style="cyan", max_width=28)
        ts_table.add_column("Recorded", justify="right", max_width=8)
        ts_table.add_column("Packages", justify="right", max_width=8)
        ts_table.add_column("Avg HP", justify="right", max_width=7)
        ts_table.add_column("Calibrated From", max_width=20)
        ts_table.add_column("ID", style="dim", max_width=18)

        for ts in ts_list:
            cal = ts.get("calibrated_with", {})
            ts_table.add_row(
                ts.get("name", "?"),
                str(ts.get("recorded_count", 0)),
                str(ts.get("packages_count", 0)),
                f"{ts.get('recorded_avg_hp', 0):.0f}",
                cal.get("checkpoint", "?"),
                ts.get("training_set_id", "?")[:16],
            )
        parts.append(ts_table)

    footer = Text(f"  {len(experiments)} experiments | {len(ts_list)} training sets | refresh 2s | Ctrl+C to exit", style="dim")
    parts.append(footer)

    return Group(*parts)


def main():
    console = Console()

    with Live(console=console, refresh_per_second=2, screen=True) as live:
        while True:
            experiments = _collect_experiments()
            live.update(build_dashboard(experiments))
            time.sleep(0.5)


if __name__ == "__main__":
    main()
