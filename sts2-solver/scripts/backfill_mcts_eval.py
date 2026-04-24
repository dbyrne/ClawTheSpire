"""Backfill mcts_eval.jsonl for experiments that predate per-gen MCTS eval.

Calls run_mcts_eval on every betaone_gen*.pt in the experiment dir and writes
the same jsonl entry format that the training loop does. Safe to re-run —
dedupes by (suite, gen).

Usage:
    PYTHONIOENCODING=utf-8 python scripts/backfill_mcts_eval.py \\
        --experiment-dir C:/coding-projects/sts2-reanalyse-v4/sts2-solver/experiments/reanalyse-v4
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

from sts2_solver.betaone.eval import run_mcts_eval
from sts2_solver.betaone.suite import compute_eval_suite, suite_id as _suite_id


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--experiment-dir", required=True, type=Path)
    p.add_argument("--num-sims", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    exp = args.experiment_dir
    bench_dir = exp / "benchmarks"
    bench_dir.mkdir(exist_ok=True)
    out_path = bench_dir / "mcts_eval.jsonl"

    # Load existing entries to dedupe
    existing = set()
    if out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    r = json.loads(line)
                    existing.add((r.get("suite"), r.get("gen")))
                except Exception:
                    pass

    # Find all gen checkpoints
    gens = []
    for f in sorted(exp.glob("betaone_gen*.pt")):
        m = re.search(r"gen(\d+)", f.name)
        if m:
            gens.append((int(m.group(1)), f))
    gens.sort()
    print(f"found {len(gens)} checkpoints")

    suite = _suite_id(compute_eval_suite())
    print(f"suite: {suite}")

    written = 0
    with open(out_path, "a", encoding="utf-8") as out:
        for gen, path in gens:
            if (suite, gen) in existing:
                print(f"  gen {gen}: already in jsonl, skip")
                continue
            t0 = time.time()
            # Per-gen unique onnx dir avoids Windows file-lock contention from
            # retained ort::Session handles when re-exporting to the same path
            # in a tight loop.
            with tempfile.TemporaryDirectory(prefix="mcts_eval_backfill_") as td:
                res = run_mcts_eval(
                    str(path), num_sims=args.num_sims, seed=args.seed,
                    onnx_dir=td,
                )
            entry = {
                "suite": suite, "timestamp": time.time(), "gen": gen,
                "total": res["total"],
                "clean": res["clean"], "echo": res["echo"],
                "fixed": res["fixed"], "broke": res["broke"],
                "mixed": res["mixed"], "nomatch": res["nomatch"],
                "rescue_rate": round(res["rescue_rate"], 4),
            }
            out.write(json.dumps(entry) + "\n")
            out.flush()
            written += 1
            print(f"  gen {gen:3d}: CLEAN={res['clean']:3d} ECHO={res['echo']:2d} "
                  f"FIXED={res['fixed']:2d} BROKE={res['broke']:2d} MIXED={res['mixed']:2d} "
                  f"rescue={res['rescue_rate']:.0%} ({time.time()-t0:.1f}s)",
                  flush=True)

    print(f"\nwrote {written} new entries to {out_path}")


if __name__ == "__main__":
    main()
