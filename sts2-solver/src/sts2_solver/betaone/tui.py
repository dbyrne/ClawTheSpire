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

from .experiment import MCTS_DEFAULTS
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


_PHASE_COLORS = {
    "SELFPLAY": "bold cyan",
    "TRAINING": "bold magenta",
    "REANALYSING": "bold yellow",
    "EVALUATING": "bold blue",
}


def _status_text(age: float, phase: str | None = None) -> Text:
    # Fresh phase marker beats the timestamp-based fallback: gen steps at
    # 1000 POMCP sims routinely exceed 2 min, so without this every
    # healthy training run would show STALLED? for most of each gen.
    # Honor phase within 10 min of its timestamp — plenty for any single
    # phase to run (selfplay 5-8 min, reanalyse 4+ min).
    if phase and age < 600:
        return Text(phase, style=_PHASE_COLORS.get(phase, "bold green"))
    if age < 120:
        return Text("RUNNING", style="bold green")
    elif age < 600:
        return Text("STALLED?", style="bold yellow")
    else:
        return Text("stopped", style="dim")


MAX_EXPERIMENTS = 15


def _collect_experiments() -> list[dict]:
    """Gather data for the most recent MAX_EXPERIMENTS experiments by mtime.

    Aggregates via _all_experiment_sources — one authoritative dir per
    experiment name. Worktree experiments contribute their own dir only
    (not inherited copies from when the worktree was branched), and the
    worktree's copy wins over main's if both exist for the same name.
    """
    from .experiment import _all_experiment_sources

    import yaml as _yaml
    def _is_concluded(d: Path) -> bool:
        try:
            with open(d / "config.yaml", encoding="utf-8") as f:
                return _yaml.safe_load(f).get("concluded_gen") is not None
        except Exception:
            return False

    def _is_live(d: Path) -> bool:
        """Live = running or stalled (progress timestamp < 10 min old)."""
        cfg_path = d / "config.yaml"
        try:
            with open(cfg_path, encoding="utf-8") as f:
                net_type = _yaml.safe_load(f).get("network_type", "betaone")
        except Exception:
            return False
        prog_name = ("decknet_progress.json" if net_type == "decknet"
                     else "betaone_progress.json")
        prog = _load_json(d / prog_name)
        if not prog:
            return False
        return (time.time() - prog.get("timestamp", 0)) < 600

    all_dirs = [d for _name, d in _all_experiment_sources()]
    # Three-tier render order (top -> bottom):
    #   1. stopped  (inactive, fills remaining slots with most-recent-first)
    #   2. done     (finalized reference; always shown)
    #   3. live     (running or stalled in last 10 min; always shown at bottom)
    # Rationale: the rows you most want visible while monitoring are the ones
    # actively producing data. Scrolling's natural eye-landing is the bottom
    # row, so the live ones go there. Done experiments sit above as frozen
    # reference; stopped experiments fill the rest if there's room.
    live = sorted((d for d in all_dirs if _is_live(d) and not _is_concluded(d)),
                  key=lambda p: p.stat().st_mtime)
    done = sorted((d for d in all_dirs if _is_concluded(d)),
                  key=lambda p: p.stat().st_mtime)
    stopped = sorted((d for d in all_dirs
                      if not _is_live(d) and not _is_concluded(d)),
                     key=lambda p: p.stat().st_mtime)
    # Always include all live and all done; fill remaining with most-recent stopped.
    remaining = max(MAX_EXPERIMENTS - len(live) - len(done), 0)
    dirs = stopped[-remaining:] + done + live

    experiments = []
    for d in dirs:
        import yaml
        with open(d / "config.yaml", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        net_type = config.get("network_type", "betaone")
        prog_name = "decknet_progress.json" if net_type == "decknet" else "betaone_progress.json"
        hist_name = "decknet_history.jsonl" if net_type == "decknet" else "betaone_history.jsonl"
        progress = _load_json(d / prog_name)
        history = _load_jsonl_all(d / hist_name, tail=60)
        eval_all = _load_jsonl_all(d / "benchmarks" / "eval.jsonl", tail=200)
        value_eval_all = _load_jsonl_all(d / "benchmarks" / "value_eval.jsonl", tail=200)
        mcts_eval_all = _load_jsonl_all(d / "benchmarks" / "mcts_eval.jsonl", tail=200)

        concluded_gen = config.get("concluded_gen")
        if concluded_gen is not None:
            # Pin the surfaced eval to the concluded gen so finalized
            # experiments report their canonical scores, not whatever happened
            # to be the latest eval run.
            pinned = [r for r in eval_all if r.get("gen") == concluded_gen]
            eval_result = pinned[-1] if pinned else None
            pinned_v = [r for r in value_eval_all if r.get("gen") == concluded_gen]
            value_eval_result = pinned_v[-1] if pinned_v else None
            pinned_m = [r for r in mcts_eval_all if r.get("gen") == concluded_gen]
            mcts_eval_result = pinned_m[-1] if pinned_m else None
        else:
            eval_result = eval_all[-1] if eval_all else None
            value_eval_result = value_eval_all[-1] if value_eval_all else None
            mcts_eval_result = mcts_eval_all[-1] if mcts_eval_all else None

        eval_history = eval_all[-60:]
        value_eval_history = value_eval_all[-60:]
        mcts_eval_history = mcts_eval_all[-60:]
        benchmarks = _load_jsonl_all(d / "benchmarks" / "results.jsonl", tail=10)

        experiments.append({
            "name": config.get("name", d.name),
            "method": config.get("method", "?"),
            "network_type": config.get("network_type", "betaone"),
            "description": config.get("description", ""),
            "parent": config.get("parent"),
            "cold_start": config.get("checkpoints", {}).get("cold_start", False),
            "arch": config.get("architecture", {}),
            "training": config.get("training", {}),
            "data": config.get("data", {}),
            "progress": progress,
            "history": history,
            "eval": eval_result,
            "value_eval": value_eval_result,
            "mcts_eval": mcts_eval_result,
            "eval_history": eval_history,
            "value_eval_history": value_eval_history,
            "mcts_eval_history": mcts_eval_history,
            "benchmarks": benchmarks,
            "concluded_gen": concluded_gen,
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


def _combat_net_ref(exp: dict) -> str:
    """Pull the BetaOne checkpoint a DeckNet experiment inherits from.

    Renders the basename minus extension so rows stay readable; e.g.
    "experiments/mcts-bootstrap-pwfix1000/betaone_latest.pt" becomes
    "mcts-bootstrap-pwfix1000".
    """
    dn = exp.get("training", {}).get("decknet", {})
    ckpt = dn.get("betaone_checkpoint", "")
    if not ckpt:
        return "-"
    # Expect paths like ".../experiments/<exp-name>/betaone_latest.pt"
    parts = Path(ckpt).parts
    try:
        idx = parts.index("experiments")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    except ValueError:
        pass
    return Path(ckpt).stem


def build_dashboard(experiments: list[dict]) -> Group:
    """Build the full dashboard layout.

    Two-column top: training state (exps + encounter sets) on the left,
    benchmarks pane on the right at fixed width. Sparklines/candlesticks
    and the footer span the full width below. Keeps benchmarks visible
    without scrolling past the training state.

    BetaOne and DeckNet experiments render in separate tables because
    their metrics don't cleanly overlap (combat WR vs run WR, combat
    eval vs deck eval, etc.). Mixing them in one table required
    too many blank columns and made the view hard to scan.
    """
    # Left column accumulates training-state tables + candlesticks. Right
    # column is the benchmarks pane. Footer stays full-width below.
    left_parts: list = []
    right_parts: list = []
    full_parts: list = []
    # Back-compat alias — the per-experiment candlestick block below
    # appends to `parts`; we redirect it to left_parts so candlesticks
    # render in the left column alongside the exps tables.
    parts = left_parts

    # Assign colors to experiments (global, so same color across tables)
    exp_color_map = {exp["name"]: _exp_color(i) for i, exp in enumerate(experiments)}

    # Partition by network type
    betaone_exps = [e for e in experiments if e.get("network_type", "betaone") != "decknet"]
    decknet_exps = [e for e in experiments if e.get("network_type") == "decknet"]

    # === BetaOne experiments table ===
    if betaone_exps:
        overview = Table(title="BetaOne — Combat Experiments", expand=True, show_lines=False)
        overview.add_column("Name", max_width=28)
        overview.add_column("Method", max_width=10)
        overview.add_column("Params", justify="right", max_width=8)
        overview.add_column("VHL", justify="right", max_width=4)
        overview.add_column("Base", justify="right", max_width=5)
        overview.add_column("Status", max_width=10)
        overview.add_column("Start", max_width=20)
        overview.add_column("Gen", justify="right", max_width=7)
        overview.add_column("Train WR", justify="right", max_width=8)
        overview.add_column("Best", justify="right", max_width=7)
        overview.add_column("P-Eval", justify="right", max_width=7)
        overview.add_column("V-Eval", justify="right", max_width=7)
        overview.add_column("Search", justify="right", max_width=7)
        overview.add_column("Suite", justify="right", max_width=12)
        overview.add_column("ET Avg", justify="right", max_width=7)
        overview.add_column("Buffer", justify="right", max_width=10)
        overview.add_column("C", justify="right", max_width=4)
        overview.add_column("Noise", justify="right", max_width=5)
        overview.add_column("Encounter Set", max_width=18)

        for exp in betaone_exps:
            p = exp["progress"]
            ev = exp["eval"]
            vev = exp.get("value_eval")
            mev = exp.get("mcts_eval")
            arch = exp["arch"]
            concluded = exp.get("concluded_gen")

            if concluded is not None:
                status = Text("done", style="bold cyan")
                gen_str = f"{concluded}*"
                wr = "-"
                best = "-"
            elif p:
                age = time.time() - p.get("timestamp", 0)
                status = _status_text(age, phase=p.get("phase"))
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
            # Additional complexity/arch signals: value_head_layers (varies 1 vs 3
            # across current experiments — genuine head capacity difference) and
            # base_state_dim (encoder generation lineage: 137=legacy, 140=current,
            # 144=enemy-intent). Both from arch_meta; shown compactly.
            vhl = arch.get("value_head_layers")
            vhl_str = str(vhl) if vhl is not None else "-"
            base_dim = arch.get("base_state_dim")
            base_str = str(base_dim) if base_dim is not None else "-"

            # Start: "cold" if no parent OR cold_start=true (even after a fork,
            # the checkpoints.cold_start flag intends fresh weights — the fork's
            # copied parent checkpoint is discarded). Otherwise "<-parent-name"
            # for warm-starts. Parent name carries the lineage; cold-from-fork
            # tells the reader "forked for code inheritance, not weight inheritance."
            parent = exp.get("parent")
            cold = exp.get("cold_start", False)
            if not parent:
                start_str = Text("cold", style="dim")
            elif cold:
                abbrev = parent if len(parent) <= 12 else "…" + parent[-11:]
                start_str = Text(f"cold ({abbrev})", style="dim")
            else:
                abbrev = parent if len(parent) <= 18 else "…" + parent[-17:]
                start_str = Text(f"<-{abbrev}", style="green")

            eval_str = f"{ev['score']:.0%}" if ev and "score" in ev else "-"
            vev_str = f"{vev['score']:.0%}" if vev and "score" in vev else "-"
            # Net search contribution: (FIXED-BROKE)/(FIXED+ECHO+BROKE). Range
            # [-1, 1]. Positive = MCTS helps policy; negative = MCTS hurts.
            # Replaces prior rescue-rate metric that maxed at 100% when ECHO=0
            # even though BROKE cases made net contribution negative.
            mev_str = f"{mev['rescue_rate']*100:+.0f}%" if mev and "rescue_rate" in mev else "-"
            # Suite signature = "<P-total>/<V-total>" scenario counts. Same
            # signature across rows means scores were computed against the same
            # number of scenarios — apples-to-apples check.
            p_total = ev.get("total") if ev else None
            v_total = vev.get("total") if vev else None
            if p_total or v_total:
                suite_str = f"{p_total or '-'}/{v_total or '-'}"
            else:
                suite_str = "-"
            et_str = f"{ev['end_turn_avg']:.0%}" if ev and ev.get("end_turn_avg") else "-"

            ts = exp["data"].get("encounter_set") or exp["data"].get("training_set", "")
            ts_str = _resolve_ts_name(ts) if ts else "-"

            if exp["method"] == "ppo":
                method = "PPO"
                buf_str = "-"
                cpuct_str = "-"
                noise_str = "-"
            else:
                mcts = exp.get("training", {}).get("mcts", {})
                sims = mcts.get("num_sims", "?")
                prefix = "POMCP" if mcts.get("pomcp", False) else "MCTS"
                method = f"{prefix}-{sims}"
                # Replay buffer
                if p:
                    buf_size = p.get("buffer_size", 0)
                    buf_cap = mcts.get("replay_capacity", 0)
                    buf_str = f"{buf_size//1000}K/{buf_cap//1000}K" if buf_cap else (
                        f"{buf_size//1000}K" if buf_size else "-"
                    )
                else:
                    buf_str = "-"
                cpuct_str = f"{mcts.get('c_puct', MCTS_DEFAULTS['c_puct'])}"
                noise_str = f"{mcts.get('noise_frac', MCTS_DEFAULTS['noise_frac'])}"

            color = exp_color_map.get(exp["name"], "white")
            name_text = Text(exp["name"], style=color)
            # Finalized rows render dim so active experiments visually dominate;
            # the pinned scores are still legible, just deprioritized.
            row_style = "dim" if concluded is not None else None
            overview.add_row(name_text, method, params_str, vhl_str, base_str,
                             status, start_str,
                             gen_str, wr, best, eval_str, vev_str, mev_str,
                             suite_str,
                             et_str, buf_str,
                             cpuct_str, noise_str, ts_str,
                             style=row_style)

        left_parts.append(overview)

    # === DeckNet experiments table ===
    if decknet_exps:
        dn_table = Table(title="DeckNet — Deck-Building Experiments", expand=True, show_lines=False)
        dn_table.add_column("Name", max_width=28)
        dn_table.add_column("Method", max_width=12)
        dn_table.add_column("Params", justify="right", max_width=8)
        dn_table.add_column("Status", max_width=10)
        dn_table.add_column("Gen", justify="right", max_width=7)
        dn_table.add_column("Run WR", justify="right", max_width=8)
        dn_table.add_column("Avg Floor", justify="right", max_width=9)
        dn_table.add_column("Eval", justify="right", max_width=7)
        dn_table.add_column("Buffer", justify="right", max_width=10)
        dn_table.add_column("Combat Net", max_width=22)

        for exp in decknet_exps:
            p = exp["progress"]
            ev = exp["eval"]
            arch = exp["arch"]

            if p:
                age = time.time() - p.get("timestamp", 0)
                status = _status_text(age, phase=p.get("phase"))
                gen = str(p.get("gen", 0))
                total = p.get("num_generations", "?")
                gen_str = f"{gen}/{total}" if total != "?" else gen
                run_wr = f"{p.get('win_rate', 0):.1%}"
                avg_floor = f"{p.get('avg_floor', 0):.1f}"
                buf_size = p.get("buffer_size", 0)
                buf_cap = exp.get("training", {}).get("decknet", {}).get("replay_capacity", 0)
                buf_str = f"{buf_size//1000}K/{buf_cap//1000}K" if buf_cap else (
                    f"{buf_size//1000}K" if buf_size else "-"
                )
            else:
                status = Text("new", style="dim")
                gen_str = "-"
                run_wr = "-"
                avg_floor = "-"
                buf_str = "-"

            params = arch.get("total_params")
            params_str = f"{params//1000}K" if isinstance(params, int) else "-"

            dn = exp.get("training", {}).get("decknet", {})
            sims = dn.get("mcts_sims", "?")
            method = f"DeckNet-{sims}"

            eval_str = f"{ev['score']:.0%}" if ev and "score" in ev else "-"
            combat_net = _combat_net_ref(exp)

            color = exp_color_map.get(exp["name"], "white")
            name_text = Text(exp["name"], style=color)
            dn_table.add_row(name_text, method, params_str, status,
                             gen_str, run_wr, avg_floor, eval_str, buf_str,
                             combat_net)

        left_parts.append(dn_table)

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
                "gen": b.get("gen"),
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
        bench_table.add_column("Mode", max_width=14)
        bench_table.add_column("Experiment", max_width=26)
        bench_table.add_column("Gen", justify="right", max_width=5)
        bench_table.add_column("WR", justify="right", max_width=7)
        bench_table.add_column("95% CI", justify="right", max_width=16)
        bench_table.add_column("N", justify="right", max_width=6)

        # Deduplicate: keep most recent per (experiment, suite, mode, sims, gen).
        # Include gen so per-checkpoint benchmarks (gen-61 vs gen-70 on the
        # same experiment) stay as separate rows. Matches the save_benchmark
        # dedup key fix from commit 99a4d7e.
        seen = {}
        for row in reversed(bench_rows):
            key = (row["name"], row["suite"], row["mode"], row["sims"], row["gen"])
            if key not in seen:
                seen[key] = row

        # Sort by suite, then mode, then best WR descending (per-gen rows
        # land in WR order within the same suite/mode group).
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
            gen_str = str(row["gen"]) if row["gen"] is not None else "-"
            bench_table.add_row(
                suite_display, mode_str, Text(row["name"], style=color),
                gen_str,
                f"{row['wr']:.1%}", ci, str(row["n"]),
            )
        right_parts.append(bench_table)

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
        left_parts.append(es_table)

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
        # hp_loss is only logged for HP-head variants (hploss-aux-v1+).
        # None indicates "no HP head in this experiment" — we suppress the row
        # entirely rather than plot a zero line that'd look like a flat metric.
        hls = [r.get("hp_loss") for r in history]
        has_hp = any(v is not None for v in hls)
        hls = [v if v is not None else 0.0 for v in hls]
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
                         higher_is_better=True, bar_width=70, delta_mode="window"):
            # delta_mode:
            #   "window" — current-window mean vs prior-window mean. Smooths
            #              per-gen noise; right for dense metrics like losses.
            #   "last"   — latest point minus previous point. Right for sparse
            #              metrics like eval scores (one datum per eval_every
            #              gens) where "window mean" mixes multiple evals into
            #              one number and hides the step-to-step movement.
            lo, hi, mean = _window(vals)
            line = Text()
            line.append(f"    {label:>6s} ", style="dim")

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

            # Numeric summary — primary value matches delta semantics:
            # "window" delta-mode shows window mean + delta between windows;
            # "last" shows the latest point + delta from prior point so the
            # arithmetic (shown_value = previous + delta) adds up visually.
            primary = vals[-1] if delta_mode == "last" else mean
            line.append(f"  {fmt_val(primary)}", style="bold")
            line.append(f"  [{fmt_val(lo)}, {fmt_val(hi)}]", style="dim")

            # Delta — either window-vs-window (dense) or last-vs-prev (sparse)
            if delta_mode == "last":
                if len(vals) >= 2:
                    d = vals[-1] - vals[-2]
                    if higher_is_better:
                        style = "green" if d > 0 else "red" if d < 0 else "dim"
                    else:
                        style = "green" if d < 0 else "red" if d > 0 else "dim"
                    sign = "+" if d >= 0 else ""
                    line.append(f"  {sign}{fmt_delta(d)}", style=style)
            elif has_delta:
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
            "p-loss", pls, 0.5, 2.0,
            fmt_val=lambda v: f"{v:.3f}",
            fmt_delta=lambda d: f"{d:.3f}",
            higher_is_better=False,
        ))
        parts.append(_candle_line(
            "v-loss", vls, 0.0, 1.0,
            fmt_val=lambda v: f"{v:.3f}",
            fmt_delta=lambda d: f"{d:.3f}",
            higher_is_better=False,
        ))
        if has_hp:
            parts.append(_candle_line(
                "h-loss", hls, 0.0, 0.15,
                fmt_val=lambda v: f"{v:.4f}",
                fmt_delta=lambda d: f"{d:.4f}",
                higher_is_better=False,
            ))

        # Gradient telemetry — only rendered when sampling was enabled for
        # this experiment (grad_conflict_sample_every > 0). Shown together
        # as a block since they're interpreted together.
        cos_pv = [r.get("grad_cos_pv_mean") for r in history]
        np_vals = [r.get("grad_norm_p_mean") for r in history]
        nv_vals = [r.get("grad_norm_v_mean") for r in history]
        has_grad = any(v is not None for v in cos_pv)
        if has_grad:
            cos_pv_vals = [v if v is not None else 0.0 for v in cos_pv]
            # |g_V| / |g_P| ratio — magnitude imbalance between heads.
            # Ideal ≈ 1 (balanced). Observed ≈ 4-5 across all MCTS experiments.
            ratio_vp = []
            for p, v in zip(np_vals, nv_vals):
                if p and v and p > 0:
                    ratio_vp.append(v / p)
                else:
                    ratio_vp.append(0.0)
            parts.append(_candle_line(
                "cos-pv", cos_pv_vals, -0.5, 0.5,
                fmt_val=lambda v: f"{v:+.3f}",
                fmt_delta=lambda d: f"{d:.3f}",
                higher_is_better=True,
            ))
            parts.append(_candle_line(
                "|V/P|", ratio_vp, 0.0, 8.0,
                fmt_val=lambda v: f"{v:.2f}",
                fmt_delta=lambda d: f"{d:.2f}",
                higher_is_better=False,
            ))
        # HP-head variant: value/HP gradient cosine.
        cos_vh = [r.get("grad_cos_vh_mean") for r in history]
        if any(v is not None for v in cos_vh):
            cos_vh_vals = [v if v is not None else 0.0 for v in cos_vh]
            parts.append(_candle_line(
                "cos-vh", cos_vh_vals, 0.0, 1.0,
                fmt_val=lambda v: f"{v:+.3f}",
                fmt_delta=lambda d: f"{d:.3f}",
                higher_is_better=True,
            ))

        # Search/network agreement telemetry — echo-chamber diagnostic.
        # Rendered only when it's present in history (post-telemetry-v1).
        kl_vals = [r.get("kl_mcts_net_mean") for r in history]
        agree_vals = [r.get("top1_agree_mean") for r in history]
        vcorr_vals = [r.get("value_corr_mean") for r in history]
        if any(v is not None for v in kl_vals):
            kls = [v if v is not None else 0.0 for v in kl_vals]
            agrees = [v if v is not None else 0.0 for v in agree_vals]
            vcorrs = [v if v is not None else 0.0 for v in vcorr_vals]
            # kl toward 0: net matches search. Lower is better AS LONG AS evals
            # are also climbing — flat kl + flat eval = echo chamber.
            parts.append(_candle_line(
                "kl", kls, 0.0, 2.0,
                fmt_val=lambda v: f"{v:.3f}",
                fmt_delta=lambda d: f"{d:.3f}",
                higher_is_better=False,
            ))
            parts.append(_candle_line(
                "agree", agrees, 0.0, 1.0,
                fmt_val=lambda v: f"{v:.1%}",
                fmt_delta=lambda d: f"{d*100:+.1f}%",
                higher_is_better=True,
            ))
            parts.append(_candle_line(
                "v-corr", vcorrs, 0.0, 1.0,
                fmt_val=lambda v: f"{v:+.3f}",
                fmt_delta=lambda d: f"{d:.3f}",
                higher_is_better=True,
            ))

        # Eval pass-rate trajectory — only when eval_every is set and has
        # accumulated enough points. WR is compressed near the top of the
        # skill curve; eval scores move earlier and at higher resolution so
        # the trajectory tells us whether the model is still learning.
        # Delta is latest-vs-previous-eval (not a rolling mean): evals are
        # sparse (every eval_every gens) and step-to-step movement is what
        # actually tells us whether the last 10 gens helped.
        eh = exp.get("eval_history") or []
        # Eval sparklines now use window-mean deltas (not point-to-point) since
        # eval_every=1 has been the default since 2026-04-19 — per-gen eval
        # data is dense enough that single-point movement is dominated by
        # inherent gen-to-gen oscillation (V-Eval ±8pp, P-Eval ±3-5pp).
        # Window mean smooths that and surfaces real trends.
        if len(eh) >= 3:
            scores = [r.get("score", 0.0) for r in eh]
            parts.append(_candle_line(
                "p-eval", scores, 0.0, 1.0,
                fmt_val=lambda v: f"{v:.1%}",
                fmt_delta=lambda d: f"{d*100:.1f}%",
                higher_is_better=True,
            ))
        vh = exp.get("value_eval_history") or []
        if len(vh) >= 3:
            scores = [r.get("score", 0.0) for r in vh]
            parts.append(_candle_line(
                "v-eval", scores, 0.0, 1.0,
                fmt_val=lambda v: f"{v:.1%}",
                fmt_delta=lambda d: f"{d*100:.1f}%",
                higher_is_better=True,
            ))
        mh = exp.get("mcts_eval_history") or []
        if len(mh) >= 3:
            # Net search contribution: (FIXED-BROKE)/(FIXED+ECHO+BROKE).
            # Range [-1, 1]. Positive = value head provides useful corrective
            # signal at MCTS leaves; negative = MCTS breaks more correct picks
            # than it rescues wrong ones. Scale -0.5..+0.5 covers the
            # typical observed range cleanly.
            scores = [r.get("rescue_rate", 0.0) for r in mh]
            parts.append(_candle_line(
                "search", scores, -0.5, 0.5,
                fmt_val=lambda v: f"{v*100:+.0f}%",
                fmt_delta=lambda d: f"{d*100:+.0f}%",
                higher_is_better=True,
            ))
        # Confident-BAD rate: fraction of wrong picks where policy was
        # decisive (top1>=0.60). Rising = policy locking in wrong answers
        # (structural concern). Falling = close-calls resolving toward
        # correct (healthy maturation). Only shows on eval entries with
        # the confidence metrics (post 2026-04-21, commit adding conf_*).
        eh_conf = [r for r in (exp.get("eval_history") or [])
                   if r.get("conf_bad") is not None and r.get("bad_count")]
        if len(eh_conf) >= 3:
            rates = [r["conf_bad"] / max(r["bad_count"], 1) for r in eh_conf]
            parts.append(_candle_line(
                "conf_bad", rates, 0.0, 1.0,
                fmt_val=lambda v: f"{v:.0%}",
                fmt_delta=lambda d: f"{d*100:+.0f}%",
                higher_is_better=False,
            ))

    # === Footer ===
    footer = Text(f"  showing last {len(experiments)} experiments (max {MAX_EXPERIMENTS}) | {len(all_sets)} encounter sets | refresh 2s | Ctrl+C to exit", style="dim")
    full_parts.append(footer)

    # Two-column top: training state on the left, benchmarks pane on the
    # right. Both columns scale with terminal width (3:2 ratio) so wider
    # terminals give benchmarks more room. Footer stays full-width below.
    if left_parts or right_parts:
        top = Table.grid(expand=True, padding=(0, 1))
        top.add_column(ratio=3)
        top.add_column(ratio=2)
        top.add_row(
            Group(*left_parts) if left_parts else Text(""),
            Group(*right_parts) if right_parts else Text(""),
        )
        return Group(top, *full_parts)
    return Group(*full_parts)


def main():
    console = Console()

    with Live(console=console, refresh_per_second=2, screen=True) as live:
        while True:
            experiments = _collect_experiments()
            live.update(build_dashboard(experiments))
            time.sleep(0.5)


if __name__ == "__main__":
    main()
