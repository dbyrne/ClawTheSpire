#!/usr/bin/env python3
"""Promote an experiment checkpoint to the production frontier.

Copies `betaone_gen<N>.pt` + `card_vocab.json` from an experiment into
`betaone_checkpoints/` (where the runner looks) and writes a FRONTIER.md
pointer at the repo root so the runner can log which experiment it's
actually using.

Example:
    python scripts/promote_to_frontier.py reanalyse-v3 88
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import shutil
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
CKPT_DIR = REPO_ROOT / "betaone_checkpoints"
FRONTIER_PATH = REPO_ROOT / "FRONTIER.md"

# Places to look for an experiment's data dir, in preference order.
# Each returns a directory that should contain betaone_gen<N>.pt,
# card_vocab.json, and a benchmarks/ subdir.
_SEARCH_PATHS = [
    lambda name: REPO_ROOT / "sts2-solver" / "experiments" / name,
    lambda name: REPO_ROOT / "sts2-solver" / "experiments" / "_archive" / name,
    lambda name: REPO_ROOT.parent / f"sts2-{name}" / "sts2-solver" / "experiments" / name,
]


def find_experiment_dir(name: str, gen: int) -> Path:
    """Return the dir containing betaone_gen<gen>.pt for this experiment.

    Main's experiments/<name>/ sometimes holds only a PLAN.md + config stub
    when a worktree-based experiment was shipped but not archived — the
    actual checkpoints live in the sibling worktree. So we check for the
    specific .pt we need, not just dir existence.
    """
    candidates = []
    for resolver in _SEARCH_PATHS:
        d = resolver(name)
        candidates.append(d)
        if (d / f"betaone_gen{gen}.pt").exists():
            return d
    raise FileNotFoundError(
        f"betaone_gen{gen}.pt for '{name}' not found in any known location:\n  "
        + "\n  ".join(str(c) for c in candidates)
    )


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def pick_eval(rows: list[dict], gen: int, total: int) -> dict | None:
    """Pick the most recent eval row for `gen` on the given scenario count."""
    matches = [r for r in rows if r.get("gen") == gen and r.get("total") == total]
    if not matches:
        return None
    return max(matches, key=lambda r: r.get("timestamp", 0))


def pick_benchmarks(rows: list[dict], gen: int) -> list[dict]:
    """All MCTS benchmark rows for `gen`, sorted by suite."""
    matches = [r for r in rows if r.get("gen") == gen and r.get("mode") == "mcts"]
    return sorted(matches, key=lambda r: (r.get("suite", ""), -(r.get("mcts_sims") or 0)))


def verify_checkpoint_loads(ckpt_path: Path, vocab_path: Path) -> dict:
    """Sanity-load via the runner's path. Raises on failure."""
    import torch  # local import — script is runnable without torch in PATH only if asked to

    sys.path.insert(0, str(REPO_ROOT / "sts2-solver" / "src"))
    from sts2_solver.betaone.network import BetaOneNetwork, network_kwargs_from_meta

    with open(vocab_path) as f:
        vocab = json.load(f)
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    net = BetaOneNetwork(
        num_cards=len(vocab),
        **network_kwargs_from_meta(ck.get("arch_meta")),
    )
    net.load_state_dict(ck["model_state_dict"])
    net.eval()
    return {
        "gen": ck.get("gen"),
        "params": sum(p.numel() for p in net.parameters()),
        "vocab_size": len(vocab),
    }


def write_frontier_md(experiment: str, gen: int, info: dict,
                      p_eval: dict | None, v_eval: dict | None,
                      benchmarks: list[dict], prev_summary: str | None) -> None:
    now = _dt.datetime.now().replace(microsecond=0).isoformat()

    lines = []
    lines.append("---")
    lines.append(f"experiment: {experiment}")
    lines.append(f"gen: {gen}")
    lines.append(f"promoted_at: {now}")
    lines.append(f"params: {info['params']}")
    lines.append(f"vocab_size: {info['vocab_size']}")
    lines.append("---")
    lines.append("")
    lines.append(f"# Frontier combat checkpoint: {experiment} gen {gen}")
    lines.append("")
    lines.append(f"Promoted {now.split('T')[0]} via `scripts/promote_to_frontier.py`.")
    lines.append("")
    lines.append("## Scores at promotion")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|---|---|")
    if p_eval:
        lines.append(f"| P-Eval | {p_eval['passed']}/{p_eval['total']} ({p_eval.get('score', 0)*100:.1f}%) |")
    else:
        lines.append("| P-Eval | (no data) |")
    if v_eval:
        lines.append(f"| V-Eval | {v_eval['passed']}/{v_eval['total']} ({v_eval.get('score', 0)*100:.1f}%) |")
    else:
        lines.append("| V-Eval | (no data) |")
    for b in benchmarks:
        suite = b.get("suite", "?")
        sims = b.get("mcts_sims", 0)
        wr = b.get("win_rate", 0) * 100
        lo = b.get("ci95_lo", 0) * 100
        hi = b.get("ci95_hi", 0) * 100
        n = b.get("games", 0)
        lines.append(
            f"| Combat WR — {suite} (MCTS-{sims}) | {wr:.2f}% "
            f"(CI {lo:.2f}–{hi:.2f}%, N={n}) |"
        )
    lines.append(f"| Params | {info['params']:,} |")
    lines.append("")
    if prev_summary:
        lines.append("## Previous frontier")
        lines.append("")
        lines.append(prev_summary)
        lines.append("")
    lines.append("## How this file is used")
    lines.append("")
    lines.append(
        "The runner reads this file's YAML frontmatter at startup to log which "
        "experiment/gen it's actually loading. The `.pt` itself lives at "
        "`betaone_checkpoints/betaone_latest.pt`; this file is a human-readable "
        "pointer + history."
    )
    lines.append("")

    FRONTIER_PATH.write_text("\n".join(lines), encoding="utf-8")


def promote(experiment: str, gen: int, dry_run: bool = False) -> int:
    """Library entry point — callable from the experiment CLI."""
    exp_dir = find_experiment_dir(experiment, gen)
    src_ckpt = exp_dir / f"betaone_gen{gen}.pt"
    src_vocab = exp_dir / "card_vocab.json"
    if not src_vocab.exists():
        print(f"Error: {src_vocab} not found (needed to stay byte-compatible).", file=sys.stderr)
        return 2

    bench_dir = exp_dir / "benchmarks"
    p_eval = pick_eval(load_jsonl(bench_dir / "eval.jsonl"), gen, 127)
    v_eval = pick_eval(load_jsonl(bench_dir / "value_eval.jsonl"), gen, 121)
    benchmarks = pick_benchmarks(load_jsonl(bench_dir / "results.jsonl"), gen)

    prev_summary = None
    if FRONTIER_PATH.exists():
        head = []
        for line in FRONTIER_PATH.read_text(encoding="utf-8").splitlines():
            if line.startswith("experiment:") or line.startswith("gen:") or line.startswith("promoted_at:"):
                head.append(line)
            if len(head) >= 3:
                break
        if head:
            prev_summary = "Replaced: " + " · ".join(h.strip() for h in head)

    dst_ckpt = CKPT_DIR / "betaone_latest.pt"
    dst_vocab = CKPT_DIR / "card_vocab.json"
    backup_ckpt = CKPT_DIR / f"betaone_latest.pre-{experiment}-g{gen}.bak"

    print(f"Source: {src_ckpt}")
    print(f"  P-Eval: {p_eval['passed']}/127" if p_eval else "  P-Eval: (none)")
    print(f"  V-Eval: {v_eval['passed']}/121" if v_eval else "  V-Eval: (none)")
    print(f"  Benchmarks: {len(benchmarks)} MCTS rows")

    if dry_run:
        print("(dry-run) would back up, copy, and write FRONTIER.md")
        return 0

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    if dst_ckpt.exists():
        shutil.copy2(dst_ckpt, backup_ckpt)
        print(f"  backed up existing -> {backup_ckpt.name}")

    shutil.copy2(src_ckpt, dst_ckpt)
    if not dst_vocab.exists() or dst_vocab.read_bytes() != src_vocab.read_bytes():
        shutil.copy2(src_vocab, dst_vocab)
        print(f"  updated card_vocab.json")

    info = verify_checkpoint_loads(dst_ckpt, dst_vocab)
    if info["gen"] != gen:
        print(
            f"Warning: checkpoint claims gen {info['gen']} but you asked for "
            f"gen {gen}. Proceeding with the value from the file.",
            file=sys.stderr,
        )

    write_frontier_md(experiment, gen, info, p_eval, v_eval, benchmarks, prev_summary)
    print(f"  FRONTIER.md -> {FRONTIER_PATH}")
    print(f"Done. Runner will load {experiment} gen {gen} on next start.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("experiment", help="e.g. reanalyse-v3")
    ap.add_argument("gen", type=int, help="e.g. 88")
    ap.add_argument("--dry-run", action="store_true", help="print what would happen, don't modify anything")
    args = ap.parse_args()
    return promote(args.experiment, args.gen, args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
