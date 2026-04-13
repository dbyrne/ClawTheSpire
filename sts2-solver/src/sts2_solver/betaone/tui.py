"""BetaOne live training TUI.

Usage:
    python -m sts2_solver.betaone.tui [--dir betaone_checkpoints]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .curriculum import TIER_CONFIGS


def _load(path: Path, tail: int | None = 30):
    try:
        with open(path) as f:
            if path.suffix == ".json":
                return json.load(f)
            lines = f.readlines()
            if tail:
                lines = lines[-tail:]
            return [json.loads(l) for l in lines if l.strip()]
    except (FileNotFoundError, json.JSONDecodeError):
        return None



def _load_promotions(path: Path) -> dict[int, int]:
    """Scan full history for tier promotion gens. Returns {tier: gen}."""
    try:
        with open(path) as f:
            promoted_at: dict[int, int] = {}
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                if r.get("tier_change") == "promoted":
                    t = r["tier"]  # always the tier promoted FROM
                    if t not in promoted_at:
                        promoted_at[t] = "Eval" if r.get("eval_only") else r.get("gen", 0)
            return promoted_at
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _wr_color(wr: float) -> str:
    if wr >= 0.75: return "green"
    if wr >= 0.50: return "yellow"
    return "red"


def _ent_color(ent: float) -> str:
    if ent < 0.3: return "red"
    if ent < 0.5: return "yellow"
    if ent > 1.5: return "yellow"
    return "green"


def _sparkline(values: list[float], width: int = 20) -> str:
    """Tiny sparkline from recent values."""
    if not values:
        return ""
    blocks = " _.-=^"
    lo, hi = min(values), max(values)
    span = hi - lo if hi > lo else 1
    recent = values[-width:]
    return "".join(blocks[min(int((v - lo) / span * (len(blocks) - 1)), len(blocks) - 1)] for v in recent)


def build(progress: dict, history: list[dict], age: float,
          promoted_at: dict[int, int] | None = None) -> Group:
    promoted_at = promoted_at or {}
    # Per-tier cumulative from progress.json (persisted across runs)
    tier_cumulative: dict[str, list[int]] = progress.get("tier_cumulative", {})
    gen = progress.get("gen", 0)
    total = progress.get("num_generations", 5000)
    tier = progress.get("tier", 0)
    wr = progress.get("tier_wr") or progress.get("win_rate", 0)
    ent = progress.get("entropy", 0)
    gen_time = progress.get("gen_time", 0)

    # Status line
    if age < 10:
        status = "[green]RUNNING[/green]"
    elif age < 120:
        status = f"[green]RUNNING[/green] [dim]({age:.0f}s ago)[/dim]"
    else:
        status = f"[red]STOPPED[/red] [dim]({age / 60:.0f}m ago)[/dim]"

    eta_s = (total - gen) * gen_time
    eta = f"{eta_s / 60:.0f}m" if eta_s < 3600 else f"{eta_s / 3600:.1f}h"

    header = Text.from_markup(
        f"  [bold cyan]BetaOne[/bold cyan]  Gen [bold]{gen}[/bold]/{total}  "
        f"{status}  ETA {eta}  [dim]{gen_time:.1f}s/gen[/dim]"
    )

    # --- Curriculum ---
    # Count consecutive gens at {PROMOTE_THRESHOLD:.0%}+ for current tier
    cur_threshold = TIER_CONFIGS[min(tier, len(TIER_CONFIGS) - 1)].promote_threshold
    consecutive = 0
    for r in reversed(history):
        r_wr = r.get("tier_wr") or r.get("win_rate", 0)
        if r.get("tier") == tier and r_wr >= cur_threshold:
            consecutive += 1
        else:
            break

    ct = Table(box=None, padding=(0, 1), expand=True, show_header=False)
    ct.add_column("", width=3)
    ct.add_column("", width=3)
    ct.add_column("", width=25)
    ct.add_column("", width=11)
    ct.add_column("", ratio=2, no_wrap=True)
    ct.add_column("", ratio=1, no_wrap=True)

    # Compute per-tier recent win rates from tail history
    tier_wrs: dict[int, list[float]] = {}
    for r in history:
        t = r.get("tier", 0)
        tier_wrs.setdefault(t, []).append(r.get("tier_wr") or r.get("win_rate", 0))

    for i, cfg in enumerate(TIER_CONFIGS):
        p = 1 if i <= 3 else 2
        if cfg.custom_deck:
            deck = "fixed deck"
        elif cfg.deck_mode == "starter":
            deck = "starter"
        elif cfg.deck_mode == "review_all":
            deck = "all tiers"
        elif cfg.deck_archetypes:
            deck = "+".join(cfg.deck_archetypes)
        else:
            deck = f"random {cfg.deck_min_size}-{cfg.deck_max_size}"

        # Criteria: show recent win rate + cumulative stats
        tier_thresh = cfg.promote_threshold
        recent_at_tier = tier_wrs.get(i, [])
        cum = tier_cumulative.get(str(i), [0, 0])
        cum_w, cum_g = cum[0], cum[1]
        if recent_at_tier:
            avg_wr = sum(recent_at_tier[-10:]) / len(recent_at_tier[-10:])
            color = "green" if avg_wr >= tier_thresh else _wr_color(avg_wr)
            cum_s = f" [dim]{cum_w}/{cum_g}={cum_w/max(cum_g,1):.0%}[/dim]" if cum_g > 0 else ""
            criteria = f"[{color}]{avg_wr:.0%}[/{color}] [dim]({tier_thresh:.0%})[/dim]{cum_s}"
        elif cum_g > 0:
            cum_wr = cum_w / max(cum_g, 1)
            color = "green" if cum_wr >= tier_thresh else _wr_color(cum_wr)
            criteria = f"[dim]({tier_thresh:.0%})[/dim] [{color}]{cum_w}/{cum_g}={cum_wr:.0%}[/{color}]"
        else:
            criteria = f"[dim](need {tier_thresh:.0%})[/dim]"

        if i < tier:
            marker = "[green]OK[/green] "
            label = f"[dim]{cfg.name}[/dim]"
            deck_s = f"[dim]{deck}[/dim]"
            pgen = promoted_at.get(i)  # promoted FROM this tier
            if pgen == "Eval":
                prg = "[green]Eval passed[/green]"
            elif pgen:
                prg = f"[green]Gen {pgen}[/green]"
            else:
                prg = "[green]Passed[/green]"
        elif i == tier:
            marker = "[bold yellow]>>[/bold yellow] "
            label = f"[bold]{cfg.name}[/bold]"
            deck_s = f"[bold]{deck}[/bold]"
            dots = "[green]=[/green]" * min(consecutive, 3) + "[dim].[/dim]" * max(0, 3 - consecutive)
            if consecutive >= 3:
                prg = f"[green][{dots}] READY[/green]"
            else:
                prg = f"[yellow][{dots}] {consecutive}/3 at {tier_thresh:.0%}[/yellow]"
        else:
            marker = "[dim]..[/dim] "
            label = f"[dim]{cfg.name}[/dim]"
            deck_s = f"[dim]{deck}[/dim]"
            criteria = "[dim]--[/dim]"
            prg = "[dim]--[/dim]"

        ts = "green" if i < tier else "bold yellow" if i == tier else "dim"
        ct.add_row(marker, f"[{ts}]T{i}[/{ts}]", label, deck_s, criteria, prg)

    # --- Metrics ---
    mt = Table(box=None, padding=(0, 3), expand=True, show_header=False)
    mt.add_column("", width=12)
    mt.add_column("", width=10)
    mt.add_column("", width=12)
    mt.add_column("", width=10)
    mt.add_column("", width=12)
    mt.add_column("", width=10)

    wr_s = f"[{_wr_color(wr)}]{wr:.1%}[/{_wr_color(wr)}]"
    ent_s = f"[{_ent_color(ent)}]{ent:.3f}[/{_ent_color(ent)}]"
    mt.add_row(
        "[dim]Win Rate[/dim]", wr_s,
        "[dim]Entropy[/dim]", ent_s,
        "[dim]Avg HP[/dim]", f"{progress.get('avg_hp', 0):.0f}",
    )
    mt.add_row(
        "[dim]Policy Loss[/dim]", f"{progress.get('policy_loss', 0):+.4f}",
        "[dim]Value Loss[/dim]", f"{progress.get('value_loss', 0):.4f}",
        "[dim]Avg Reward[/dim]", f"{progress.get('avg_reward', 0):+.4f}",
    )
    mt.add_row(
        "[dim]Temperature[/dim]", f"{progress.get('temperature', 0):.2f}",
        "[dim]Steps/gen[/dim]", f"{progress.get('steps', 0):,}",
        "[dim]Best WR[/dim]", f"{progress.get('best_win_rate', 0):.1%}",
    )

    # --- History ---
    ht = Table(box=None, padding=(0, 1), expand=True)
    ht.add_column("Gen", style="dim", width=5, justify="right")
    ht.add_column("Tier", width=3, justify="center")
    ht.add_column("Win Rate", width=8, justify="right")
    ht.add_column("HP", width=4, justify="right")
    ht.add_column("Entropy", width=7, justify="right")
    ht.add_column("Pi Loss", width=8, justify="right")
    ht.add_column("V Loss", width=7, justify="right")
    ht.add_column("Reward", width=8, justify="right")
    ht.add_column("", width=3)

    for r in history[-14:]:
        tc = r.get("tier_change", "")
        tc_s = "[green]UP[/green]" if tc == "promoted" else ""
        rwr = r.get("tier_wr") or r.get("win_rate", 0)
        rent = r.get("entropy", 0)
        ht.add_row(
            str(r.get("gen", "")),
            f"T{r.get('tier', 0)}",
            f"[{_wr_color(rwr)}]{rwr:.1%}[/{_wr_color(rwr)}]",
            f"{r.get('avg_hp', 0):.0f}",
            f"[{_ent_color(rent)}]{rent:.3f}[/{_ent_color(rent)}]",
            f"{r.get('policy_loss', 0):+.4f}",
            f"{r.get('value_loss', 0):.4f}",
            f"{r.get('avg_reward', 0):+.4f}",
            tc_s,
        )

    # Win rate sparkline
    wr_vals = [r.get("tier_wr") or r.get("win_rate", 0) for r in history]
    spark = _sparkline(wr_vals, 30)

    # --- Flags ---
    flags = []
    if ent < 0.5:
        flags.append("[yellow]Entropy low (<0.5)[/yellow]")
    if ent > 2.0:
        flags.append("[yellow]Entropy high (>2.0)[/yellow]")
    if progress.get("value_loss", 0) > 5.0:
        flags.append("[red]Value loss diverging[/red]")
    regressed = progress.get("regressed")
    regressed_detail = progress.get("regressed_detail")
    if regressed:
        gens_until_recheck = 50 - (gen % 50) if gen % 50 != 0 else 50
        if regressed_detail:
            parts = [f"T{t} {regressed_detail[str(t)]:.0%}" for t in regressed if str(t) in regressed_detail]
        else:
            parts = [f"T{t}" for t in regressed]
        flags.append(f"[red]Regression: {', '.join(parts)} — reviewing ({gens_until_recheck} gens to recheck)[/red]")
    flag_str = "  ".join(flags) if flags else "[green]Healthy[/green]"
    gens_at = progress.get("gens_at_tier", 0)
    promo_str = f"T{tier} Promote: 3 consecutive >= {cur_threshold:.0%} ({consecutive}/3)"

    return Group(
        Panel(header, style="cyan"),
        Panel(ct, title="[bold]Curriculum[/bold]", border_style="blue"),
        Panel(mt, title="[bold]Metrics[/bold]", border_style="blue"),
        Panel(ht, title=f"[bold]Recent Generations[/bold]  [dim]WR trend: {spark}[/dim]", border_style="blue"),
        Text.from_markup(f"  {flag_str}\n  [dim]{promo_str}[/dim]"),
    )


def main():
    parser = argparse.ArgumentParser(description="BetaOne live TUI")
    parser.add_argument("--dir", default="betaone_checkpoints")
    args = parser.parse_args()

    d = Path(args.dir)
    console = Console()

    empty_progress = {
        "gen": 0, "num_generations": 0, "tier": 0,
        "win_rate": 0, "entropy": 0, "gen_time": 0, "avg_hp": 0,
        "policy_loss": 0, "value_loss": 0, "avg_reward": 0,
        "temperature": 0, "steps": 0, "best_win_rate": 0,
        "gens_at_tier": 0,
    }

    history_path = d / "betaone_history.jsonl"
    with Live(console=console, refresh_per_second=2, screen=True) as live:
        while True:
            p = _load(d / "betaone_progress.json")
            h = _load(history_path) or []
            promos = _load_promotions(history_path)
            if p:
                age = time.time() - p.get("timestamp", time.time())
                live.update(build(p, h, age, promos))
                if p.get("gen", 0) >= p.get("num_generations", 0) and age > 30:
                    break
            else:
                live.update(build(empty_progress, [], 999, {}))
            time.sleep(0.5)

    console.print("\n[green]Training complete.[/green]")


if __name__ == "__main__":
    main()
