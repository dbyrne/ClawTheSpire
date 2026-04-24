"""Register a distillation experiment so it shows up in the TUI.

For each distill experiment directory, creates:
  - config.yaml (minimal, with method: distill)
  - betaone_progress.json (so TUI reads status/progress)
  - benchmarks/eval.jsonl (P-Eval rows keyed by final epoch)
  - benchmarks/value_eval.jsonl (V-Eval rows)
  - benchmarks/mcts_eval.jsonl (CLEAN/ECHO/FIXED/BROKE/rescue)

Optionally runs policy-only benchmark and writes benchmarks/results.jsonl.

Usage:
    python -m scripts.register_distill_experiment --dir experiments/distill-c51-v1
    python -m scripts.register_distill_experiment --dir experiments/distill-c51-v1 --with-benchmark
    python -m scripts.register_distill_experiment --all  # register all distill-*
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import sts2_solver.betaone.eval as eval_mod
import sts2_solver.betaone.benchmark as bench_mod
import sts2_solver.betaone.network as net_mod
from distill_c51 import DistStudentNetwork
from distill_transformer import TransformerStudentNetwork


def _inspect_arch(ckpt_path: str):
    import torch
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    meta = ckpt.get("arch_meta") or {}
    return {
        "is_distributional": bool(meta.get("distributional_value")),
        "is_transformer": bool(meta.get("transformer_trunk")),
        "params": sum(
            p.numel() for p in torch.load(ckpt_path, map_location="cpu", weights_only=False)["model_state_dict"].values()
            if hasattr(p, "numel")
        ),
        "num_cards": ckpt.get("num_cards"),
        "arch_meta": meta,
        "epoch": ckpt.get("epoch") or ckpt.get("gen"),
    }


def _pick_class(info):
    if info["is_transformer"]:
        return TransformerStudentNetwork
    if info["is_distributional"]:
        return DistStudentNetwork
    return None


_ORIG_BN = eval_mod.BetaOneNetwork


def _patch(cls):
    """Patch BetaOneNetwork to cls (or restore original if cls is None)."""
    target = cls if cls is not None else _ORIG_BN
    eval_mod.BetaOneNetwork = target
    bench_mod.BetaOneNetwork = target
    net_mod.BetaOneNetwork = target


def _make_config(exp_dir: Path, info: dict):
    """Write a config.yaml that marks this as a finalized distill experiment."""
    name = exp_dir.name
    method = "distill_transformer" if info["is_transformer"] else ("distill_c51" if info["is_distributional"] else "distill")
    meta = info["arch_meta"]
    config = {
        "name": name,
        "method": method,
        "description": (
            f"Supervised distillation from v3 g88 + MCTS-2000. "
            f"Finalized at epoch {info['epoch']}."
        ),
        "network_type": "betaone",
        "architecture": {
            **meta,
            "num_cards": info.get("num_cards", 578),
            "total_params": info["params"],
        },
        "training": {
            "epochs": info["epoch"],
            "lr": 3e-4,
            "batch_size": 512,
        },
        "data": {
            "mode": "distillation",
            "teacher": "reanalyse-v3 g88",
        },
        "checkpoints": {
            "save_every": 10,
            "cold_start": True,
        },
        "concluded_gen": info["epoch"],  # repurpose gen as epoch so TUI pins display
        "concluded_reason": f"Supervised distillation finalized at epoch {info['epoch']}",
        "concluded_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    with open(exp_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def _make_progress(exp_dir: Path, info: dict):
    """Write betaone_progress.json in a TUI-compatible shape."""
    # Approximate final epoch as the 'gen' so TUI displays progress correctly.
    prog = {
        "gen": info["epoch"],
        "win_rate": 0.0,  # not applicable for distillation
        "avg_hp": 0.0,
        "steps": 0,
        "buffer_size": 0,
        "episodes": 0,
        "policy_loss": 0.0,
        "value_loss": 0.0,
        "num_sims": 0,
        "gen_time": 0,
        "timestamp": os.path.getmtime(exp_dir / "betaone_latest.pt"),
        "num_generations": info["epoch"],
        "best_win_rate": 0.0,
        "phase": "DONE",
    }
    with open(exp_dir / "betaone_progress.json", "w", encoding="utf-8") as f:
        json.dump(prog, f, indent=2)


def _run_evals_and_save(exp_dir: Path, ckpt_path: str, epoch: int, info: dict):
    """Run P-Eval + V-Eval + MCTS-eval and save to benchmarks/*.jsonl."""
    from sts2_solver.betaone.eval import run_eval, run_value_eval, run_mcts_eval
    from sts2_solver.betaone.suite import compute_eval_suite, suite_id as _suite_id

    bench_dir = exp_dir / "benchmarks"
    bench_dir.mkdir(parents=True, exist_ok=True)
    sid = _suite_id(compute_eval_suite())
    ts = time.time()

    pol = run_eval(ckpt_path)
    pol_entry = {
        "suite": sid, "timestamp": ts, "gen": epoch,
        "passed": pol["passed"], "total": pol["total"],
        "score": round(pol["passed"] / max(pol["total"], 1), 4),
        "end_turn_avg": pol.get("end_turn_avg"),
        "end_turn_high": pol.get("end_turn_high", 0),
        "bad_count": pol.get("bad_count"),
        "conf_bad": pol.get("conf_bad"),
        "close_bad": pol.get("close_bad"),
        "conf_clean": pol.get("conf_clean"),
        "by_category": pol.get("by_category", {}),
    }
    with open(bench_dir / "eval.jsonl", "w", encoding="utf-8") as f:
        f.write(json.dumps(pol_entry) + "\n")

    val = run_value_eval(ckpt_path)
    val_entry = {
        "suite": sid, "timestamp": ts, "gen": epoch,
        "passed": val["passed"], "total": val["total"],
        "score": round(val["passed"] / max(val["total"], 1), 4),
        "by_category": val.get("by_category", {}),
    }
    with open(bench_dir / "value_eval.jsonl", "w", encoding="utf-8") as f:
        f.write(json.dumps(val_entry) + "\n")

    mev = run_mcts_eval(ckpt_path)
    mev_entry = {
        "suite": sid, "timestamp": ts, "gen": epoch,
        "total": mev["total"],
        "clean": mev["clean"], "echo": mev["echo"],
        "fixed": mev["fixed"], "broke": mev["broke"],
        "rescue_rate": mev["rescue_rate"],
    }
    with open(bench_dir / "mcts_eval.jsonl", "w", encoding="utf-8") as f:
        f.write(json.dumps(mev_entry) + "\n")

    return pol_entry, val_entry, mev_entry


def register_one(exp_dir: Path, run_benchmark: bool = False):
    exp_dir = Path(exp_dir)
    ckpt_path = exp_dir / "betaone_latest.pt"
    if not ckpt_path.exists():
        print(f"  SKIP {exp_dir.name}: no betaone_latest.pt")
        return
    print(f"=== Registering {exp_dir.name} ===")
    info = _inspect_arch(str(ckpt_path))
    _patch(_pick_class(info))

    _make_config(exp_dir, info)
    _make_progress(exp_dir, info)
    pol, val, mev = _run_evals_and_save(exp_dir, str(ckpt_path), info["epoch"] or 100, info)
    print(f"  P-Eval: {pol['passed']}/{pol['total']}  V-Eval: {val['passed']}/{val['total']}  "
          f"MCTS-eval: C{mev['clean']} E{mev['echo']} F{mev['fixed']} B{mev['broke']} rescue={mev['rescue_rate']*100:+.0f}%")

    if run_benchmark:
        from sts2_solver.betaone.benchmark import benchmark_checkpoint
        from sts2_solver.betaone.encounter_set import load_encounter_set, load_encounter_set_meta
        enc_name = "lean-decks-v1"
        enc_set = load_encounter_set(enc_name)
        enc_meta = load_encounter_set_meta(enc_name) or {}
        suite_id = enc_meta.get("encounter_set_id", enc_name)
        res = benchmark_checkpoint(
            checkpoint_path=str(ckpt_path),
            encounter_set=enc_set,
            mode="policy",
            repeats=50,
            num_sims=0,
            c_puct=1.5, pomcp=True, turn_boundary_eval=True, pw_k=2.0,
        )
        r = res[0] if res else None
        if r:
            import math
            wins, games = r["wins"], r["games"]
            p = wins / max(games, 1)
            z = 1.96
            denom = 1 + z * z / games if games else 1
            center = (p + z * z / (2 * games)) / denom if games else 0
            margin = z * math.sqrt((p * (1 - p) + z * z / (4 * games)) / games) / denom if games else 0
            # Canonical benchmark row format (matches Experiment.save_benchmark).
            # Policy-only mode → null MCTS knobs so it doesn't merge with MCTS rows.
            entry = {
                "suite": suite_id, "mode": "policy", "mcts_sims": 0,
                "pw_k": None, "c_puct": None, "pomcp": None, "turn_boundary_eval": None,
                "timestamp": time.time(), "checkpoint": "gen_latest",
                "gen": info["epoch"],
                "win_rate": round(p, 4), "wins": wins, "games": games,
                "ci95_lo": round(max(0, center - margin), 4),
                "ci95_hi": round(min(1, center + margin), 4),
            }
            bench_dir = exp_dir / "benchmarks"
            with open(bench_dir / "results.jsonl", "w", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
            print(f"  policy-only WR: {p:.2%} [95%: {entry['ci95_lo']:.2%}, {entry['ci95_hi']:.2%}] (n={games})")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", help="Single experiment dir to register")
    p.add_argument("--all", action="store_true", help="Register all experiments/distill-* dirs")
    p.add_argument("--with-benchmark", action="store_true", help="Also run policy-only benchmark (slower)")
    args = p.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    if args.all:
        exp_root = Path("experiments")
        dirs = sorted([d for d in exp_root.iterdir() if d.is_dir() and d.name.startswith("distill-")])
        for d in dirs:
            register_one(d, run_benchmark=args.with_benchmark)
    elif args.dir:
        register_one(Path(args.dir), run_benchmark=args.with_benchmark)
    else:
        p.error("specify --dir or --all")


if __name__ == "__main__":
    main()
