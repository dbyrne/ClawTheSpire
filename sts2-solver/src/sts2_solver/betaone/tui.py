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
    """Resolve a training/encounter set ID to its friendly name."""
    from .paths import BENCHMARK_DIR
    import yaml as _yaml
    # Check encounter sets first
    for subdir in ["encounter_sets", "training_sets"]:
        ts_path = BENCHMARK_DIR / subdir / f"{ts_id}.yaml"
        if ts_path.exists():
            try:
                with open(ts_path) as f:
                    data = _yaml.safe_load(f)
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
        history = _load_jsonl_all(d / "betaone_history.jsonl", tail=60)
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


EXP_COLORS = [
    "cyan", "green", "yellow", "magenta", "blue",
    "red", "bright_cyan", "bright_green", "bright_yellow", "bright_magenta",
    "bright_blue", "bright_red", "dark_orange", "purple", "turquoise2",
]


def _exp_color(idx: int) -> str:
    return EXP_COLORS[idx % len(EXP_COLORS)]


def build_dashboard(experiments: list[dict]) -> Group:
    """Build the full dashboard layout."""
    parts = []

    # Assign colors to experiments
    exp_color_map = {exp["name"]: _exp_color(i) for i, exp in enumerate(experiments)}

    # === Experiment overview table ===
    overview = Table(title="Experiments", expand=True, show_lines=False)
    overview.add_column("Name", max_width=28)
    overview.add_column("Method", max_width=10)
    overview.add_column("Params", justify="right", max_width=8)
    overview.add_column("Status", max_width=10)
    overview.add_column("Gen", justify="right", max_width=7)
    overview.add_column("Train WR", justify="right", max_width=8)
    overview.add_column("Best", justify="right", max_width=7)
    overview.add_column("Eval", justify="right", max_width=7)
    overview.add_column("ET Avg", justify="right", max_width=7)
    overview.add_column("Buffer", justify="right", max_width=10)
    overview.add_column("Encounter Set", max_width=18)

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

        ts = exp["data"].get("encounter_set") or exp["data"].get("training_set", "")
        ts_str = _resolve_ts_name(ts) if ts else "-"

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

        color = exp_color_map.get(exp["name"], "white")
        name_text = Text(exp["name"], style=color)
        overview.add_row(name_text, method, params_str, status,
                         gen_str, wr, best, eval_str, et_str, buf_str, ts_str)

    parts.append(overview)

    # === Benchmark results table (sorted by suite, then best WR) ===
    bench_rows = []
    for exp in experiments:
        for b in exp.get("benchmarks", []):
            suite = b.get("suite", "?")
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
        bench_table = Table(title="Benchmarks (best first per encounter set)", expand=True, show_lines=False)
        bench_table.add_column("Encounter Set", max_width=16)
        bench_table.add_column("Mode", max_width=10)
        bench_table.add_column("Experiment", max_width=26)
        bench_table.add_column("WR", justify="right", max_width=7)
        bench_table.add_column("95% CI", justify="right", max_width=16)
        bench_table.add_column("N", justify="right", max_width=6)

        # Deduplicate: keep most recent per (experiment, suite, mode)
        seen = {}
        for row in reversed(bench_rows):
            key = (row["name"], row["suite"], row["mode"])
            if key not in seen:
                seen[key] = row

        # Sort by suite, then mode, then best WR descending
        sorted_rows = sorted(seen.values(),
                             key=lambda r: (r["suite"], r["mode"], -r["wr"]))

        prev_suite = None
        for row in sorted_rows:
            mode_str = f"{row['mode']}" + (f"-{row['sims']}" if row['sims'] > 0 else "")
            ci = f"[{row['ci_lo']:.1%}, {row['ci_hi']:.1%}]"
            # Visual separator between suites
            suite_display = row["suite"] if row["suite"] != prev_suite else ""
            prev_suite = row["suite"]
            color = exp_color_map.get(row["name"], "white")
            bench_table.add_row(
                suite_display, mode_str, Text(row["name"], style=color),
                f"{row['wr']:.1%}", ci, str(row["n"]),
            )
        parts.append(bench_table)

    # === Encounter sets section ===
    from .encounter_set import list_encounter_sets
    es_list = list_encounter_sets()

    # Also include legacy training sets
    try:
        from .training_set import list_training_sets
        legacy_ts = list_training_sets()
    except Exception:
        legacy_ts = []

    all_sets = es_list + legacy_ts
    if all_sets:
        es_table = Table(title="Encounter Sets", expand=True, show_lines=False)
        es_table.add_column("Name", style="cyan", max_width=28)
        es_table.add_column("Count", justify="right", max_width=6)
        es_table.add_column("Avg HP", justify="right", max_width=7)
        es_table.add_column("Calibrated From", max_width=20)
        es_table.add_column("ID", style="dim", max_width=18)

        for es in es_list:
            src = es.get("source", {})
            cal = src.get("calibrated_with", {}) if isinstance(src, dict) else {}
            es_table.add_row(
                es.get("name", "?"),
                str(es.get("encounter_count", 0)),
                f"{es.get('avg_hp', 0):.0f}",
                cal.get("checkpoint", "?"),
                es.get("encounter_set_id", "?")[:16],
            )
        for ts in legacy_ts:
            cal = ts.get("calibrated_with", {})
            count = ts.get("recorded_count", 0) + ts.get("packages_count", 0)
            es_table.add_row(
                ts.get("name", "?") + " (legacy)",
                str(count),
                f"{ts.get('recorded_avg_hp', 0):.0f}",
                cal.get("checkpoint", "?"),
                ts.get("training_set_id", "?")[:16],
            )
        parts.append(es_table)

    # === Active training candlesticks ===
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

        wrs = [r.get("win_rate", 0) for r in history]
        pls = [r.get("policy_loss", 0) for r in history]
        vls = [r.get("value_loss", 0) for r in history]
        gen_times = [r.get("gen_time", 0) for r in history]
        n_hist = len(history)

        # Window size = buffer turnover in gens (for MCTS), else 25
        mcts_cfg = exp.get("training", {}).get("mcts", {})
        replay_cap = mcts_cfg.get("replay_capacity", 0)
        avg_steps = sum(r.get("steps", 0) for r in history) / max(len(history), 1)
        if replay_cap and avg_steps > 0:
            win_n = max(5, min(50, round(replay_cap / avg_steps)))
        else:
            win_n = 25  # default for PPO

        def _window(vals, n=None):
            n = n or win_n
            w = vals[-n:]
            return min(w), max(w), sum(w) / len(w)

        def _avg(vals, n=None):
            n = n or win_n
            w = vals[-n:]
            return sum(w) / len(w) if w else 0.0

        has_delta = n_hist >= win_n * 2

        color = exp_color_map.get(exp["name"], "white")
        header = Text()
        header.append(f"  {exp['name']}", style=color)
        gen_now = p.get('gen', 0)
        gen_total = p.get('num_generations', 0)
        avg_gen_time = sum(gen_times) / len(gen_times)
        remaining_s = (gen_total - gen_now) * avg_gen_time if gen_total else 0
        if remaining_s >= 3600:
            eta_str = f"{remaining_s/3600:.1f}h left"
        elif remaining_s >= 60:
            eta_str = f"{remaining_s/60:.0f}m left"
        else:
            eta_str = f"{remaining_s:.0f}s left"
        header.append(f"  gen {gen_now}/{gen_total}  {gen_times[-1]:.0f}s/gen  ~{eta_str}  window={win_n}", style="dim")
        parts.append(header)

        # Render one candlestick line per metric
        def _candle_line(label, vals, scale_lo, scale_hi, fmt_val, fmt_delta,
                         higher_is_better=True, bar_width=80):
            lo, hi, mean = _window(vals)
            line = Text()
            line.append(f"    {label:>3s} ", style="dim")

            # Build bar: · = empty, ░ = range, █ = mean
            bar = [' '] * bar_width
            span = scale_hi - scale_lo
            if span <= 0:
                span = 1.0

            def pos(v):
                return max(0, min(bar_width - 1, int((v - scale_lo) / span * (bar_width - 1))))

            lo_p = pos(lo)
            hi_p = pos(hi)
            mean_p = pos(mean)

            for i in range(bar_width):
                if lo_p <= i <= hi_p:
                    bar[i] = '░'
                else:
                    bar[i] = '·'
            bar[mean_p] = '█'

            # Render bar with colors
            for i, ch in enumerate(bar):
                if ch == '█':
                    line.append(ch, style="bold white")
                elif ch == '░':
                    line.append(ch, style="bright_black")
                else:
                    line.append(ch, style="dim")

            # Numeric summary
            line.append(f"  {fmt_val(mean)}", style="bold")
            line.append(f"  [{fmt_val(lo)}, {fmt_val(hi)}]", style="dim")

            # Delta from previous window
            if has_delta:
                prev_mean = _avg(vals[-win_n * 2:-win_n])
                d = mean - prev_mean
                if higher_is_better:
                    style = "green" if d > 0 else "red" if d < 0 else "dim"
                else:
                    style = "green" if d < 0 else "red" if d > 0 else "dim"
                sign = "+" if d >= 0 else ""
                line.append(f"  {sign}{fmt_delta(d)}", style=style)

            return line

        parts.append(_candle_line(
            "WR", wrs, 0.0, 0.80,
            fmt_val=lambda v: f"{v:.1%}",
            fmt_delta=lambda d: f"{d*100:.1f}%",
            higher_is_better=True,
        ))
        parts.append(_candle_line(
            "pi", pls, 0.5, 2.0,
            fmt_val=lambda v: f"{v:.3f}",
            fmt_delta=lambda d: f"{d:.3f}",
            higher_is_better=False,
        ))
        parts.append(_candle_line(
            "v", vls, 0.0, 1.0,
            fmt_val=lambda v: f"{v:.3f}",
            fmt_delta=lambda d: f"{d:.3f}",
            higher_is_better=False,
        ))

    # === Footer ===
    footer = Text(f"  {len(experiments)} experiments | {len(all_sets)} encounter sets | refresh 2s | Ctrl+C to exit", style="dim")
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
