"""Run P-Eval, V-Eval, MCTS-eval, and policy-only benchmark on a C51 checkpoint.

The distributional value head's state_dict shape differs from the scalar
BetaOneNetwork, so the vanilla eval harness falls back to partial load
and ends up with a random value head. This script monkey-patches
BetaOneNetwork -> DistStudentNetwork before calling the harness, so
state_dict loads correctly.

The network's forward returns (logits, E[V]) scalar — identical signature
to the scalar student — so everything downstream (ONNX export, MCTS,
eval comparisons) works unmodified.

Usage:
    python -m scripts.distill_c51_eval run \\
        --checkpoint experiments/distill-c51-v1/betaone_latest.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import sts2_solver.betaone.eval as eval_mod
import sts2_solver.betaone.benchmark as bench_mod
import sts2_solver.betaone.network as net_mod

from distill_c51 import DistStudentNetwork
from distill_transformer import TransformerStudentNetwork
from sts2_solver.betaone.encounter_set import load_encounter_set


def _patch_network(cls=None):
    """Replace BetaOneNetwork with the given class (default DistStudent)."""
    if cls is None:
        cls = DistStudentNetwork
    eval_mod.BetaOneNetwork = cls
    bench_mod.BetaOneNetwork = cls
    net_mod.BetaOneNetwork = cls


def _unpatch_network(orig):
    """Restore original BetaOneNetwork references in downstream modules."""
    eval_mod.BetaOneNetwork = orig
    bench_mod.BetaOneNetwork = orig
    net_mod.BetaOneNetwork = orig


def _inspect_arch(checkpoint_path: str):
    """Return (is_distributional, is_transformer) from checkpoint arch_meta."""
    import torch
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    meta = ckpt.get("arch_meta") or {}
    return bool(meta.get("distributional_value")), bool(meta.get("transformer_trunk"))


def _pick_class(checkpoint_path: str):
    """Select the right network class for this checkpoint."""
    is_dist, is_tf = _inspect_arch(checkpoint_path)
    if is_tf:
        return TransformerStudentNetwork
    if is_dist:
        return DistStudentNetwork
    return None  # use default BetaOneNetwork


# Kept for backward compat with older calls
def _is_distributional(checkpoint_path: str) -> bool:
    return _inspect_arch(checkpoint_path)[0]


def run_cmd(args):
    sys.stdout.reconfigure(encoding="utf-8")
    cls = _pick_class(args.checkpoint)
    if cls is not None:
        _patch_network(cls)
    from sts2_solver.betaone.eval import run_eval, run_value_eval, run_mcts_eval
    import math

    print("=" * 70)
    print(f"Eval C51 student: {args.checkpoint}")
    print("=" * 70)

    # Pre-run a quick forward to confirm state_dict loaded cleanly
    import torch
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    arch = ckpt.get("arch_meta", {})
    print(f"arch distributional_value={arch.get('distributional_value')} atoms={arch.get('c51_atoms')}")
    print()

    pol = run_eval(args.checkpoint)
    print(f"\n>>> P-Eval: {pol['passed']}/{pol['total']} ({100*pol['passed']/pol['total']:.1f}%)")
    print(f"    conf_bad={pol.get('conf_bad')}, bad_count={pol.get('bad_count')}, conf_clean={pol.get('conf_clean')}")

    val = run_value_eval(args.checkpoint)
    print(f"\n>>> V-Eval: {val['passed']}/{val['total']} ({100*val['passed']/val['total']:.1f}%)")

    mce = run_mcts_eval(args.checkpoint)
    print(
        f"\n>>> MCTS-eval: CLEAN={mce['clean']} ECHO={mce['echo']} "
        f"FIXED={mce['fixed']} BROKE={mce['broke']} rescue={mce['rescue_rate']*100:+.0f}%"
    )


def benchmark_cmd(args):
    sys.stdout.reconfigure(encoding="utf-8")

    import math
    import time
    orig_bn = bench_mod.BetaOneNetwork  # save original

    def _ci95(wins, n):
        if n == 0: return 0.0, 0.0, 0.0
        p = wins / n
        z = 1.96
        denom = 1 + z * z / n
        center = (p + z * z / (2 * n)) / denom
        halfw = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
        return p, max(0.0, center - halfw), min(1.0, center + halfw)

    encounter_set = load_encounter_set(args.encounter_set)
    print(f"Encounter set: {args.encounter_set} ({len(encounter_set)} encounters)")
    print(f"Repeats: {args.repeats} → n = {args.repeats * len(encounter_set)}")
    print(f"Mode: {args.mode}, sims: {args.num_sims if args.mode == 'mcts' else 0}")
    print()

    results = []
    for ckpt in args.checkpoints:
        label = Path(ckpt).parent.name + "/" + Path(ckpt).stem
        # Select the appropriate network class for THIS checkpoint (scalar vs dist vs transformer).
        # Wrong class would partial-load state_dict leaving components randomly initialized
        # (especially the trunk/value_head) — contaminating MCTS WR.
        cls = _pick_class(ckpt)
        if cls is None:
            _unpatch_network(orig_bn)
            print(f"=== {label} [scalar] ===")
        elif cls is TransformerStudentNetwork:
            _patch_network(cls)
            print(f"=== {label} [transformer+distributional] ===")
        else:
            _patch_network(cls)
            print(f"=== {label} [distributional] ===")
        from sts2_solver.betaone.benchmark import benchmark_checkpoint
        from sts2_solver.betaone.experiment import Experiment
        from sts2_solver.betaone.suite import get_encounter_set_suite
        # Resolve experiment (for on_progress save) from checkpoint path.
        exp_name = Path(ckpt).parent.name
        exp = None
        try:
            exp = Experiment(exp_name)
            if not exp.exists:
                exp = None
        except Exception:
            exp = None
        _, sid = get_encounter_set_suite(args.encounter_set)
        resolved_label = Path(ckpt).stem.replace("betaone_", "")
        # Incremental-save callback: fires per HP-group batch so partial progress
        # shows up in benchmarks/results.jsonl continuously (matches `sts2-experiment
        # benchmark` behavior). TUI reads the file live, so benchmarks appear in
        # real time instead of waiting for the whole 30-repeat run to complete.
        def _save_partial(partial, exp=exp, sid=sid, label=resolved_label):
            if exp is not None:
                exp.save_benchmark(partial, suite_id=sid, checkpoint=label)
        t0 = time.time()
        res = benchmark_checkpoint(
            checkpoint_path=ckpt,
            encounter_set=encounter_set,
            mode=args.mode,
            repeats=args.repeats,
            num_sims=args.num_sims if args.mode == "mcts" else 0,
            c_puct=1.5, pomcp=True, turn_boundary_eval=True, pw_k=2.0,
            on_progress=_save_partial,
        )
        dt = time.time() - t0
        r = res[0] if res else None
        if r is None:
            print("  NO RESULT"); continue
        wins, games = r["wins"], r["games"]
        wr, lo, hi = _ci95(wins, games)
        print(f"  {wins}/{games} = {100*wr:.2f}% [95%: {100*lo:.2f}, {100*hi:.2f}]  ({dt/60:.1f}min)")
        results.append((label, wins, games, wr, lo, hi))

    if len(results) == 2:
        a, b = results
        delta = 100 * (b[3] - a[3])
        pa, pb = a[3], b[3]
        na, nb = a[2], b[2]
        se = 100 * math.sqrt(pa*(1-pa)/na + pb*(1-pb)/nb)
        print(f"\nDelta ({b[0]} − {a[0]}): {delta:+.2f}pp (SE {se:.2f}pp)")
        print("  → SIGNIFICANT" if abs(delta) > 1.96 * se else "  → within noise")


def main():
    p = argparse.ArgumentParser()
    sp = p.add_subparsers(dest="cmd", required=True)

    r = sp.add_parser("run", help="P-Eval + V-Eval + MCTS-eval")
    r.add_argument("--checkpoint", required=True)
    r.set_defaults(func=run_cmd)

    b = sp.add_parser("benchmark", help="policy or mcts WR benchmark")
    b.add_argument("--checkpoints", nargs="+", required=True)
    b.add_argument("--encounter-set", default="lean-decks-v1")
    b.add_argument("--mode", default="policy", choices=["policy", "mcts"])
    b.add_argument("--num-sims", type=int, default=1000)
    b.add_argument("--repeats", type=int, default=50)
    b.set_defaults(func=benchmark_cmd)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
