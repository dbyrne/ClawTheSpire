"""CLI for experiment management.

Usage:
    python -m sts2_solver.betaone.experiment_cli create <name> --template mcts_selfplay
    python -m sts2_solver.betaone.experiment_cli train <name>
    python -m sts2_solver.betaone.experiment_cli fork <new> --from <source> [--override key=val]
    python -m sts2_solver.betaone.experiment_cli benchmark <name> [--checkpoint latest]
    python -m sts2_solver.betaone.experiment_cli compare <name1> <name2>
    python -m sts2_solver.betaone.experiment_cli list
    python -m sts2_solver.betaone.experiment_cli info <name>
    python -m sts2_solver.betaone.experiment_cli archive <name>
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

from .experiment import Experiment, ExperimentConfig
from .paths import EXPERIMENTS_DIR, TEMPLATES_DIR


def _format_method(config) -> str:
    """Format method string for display. DeckNet gets its own label since
    it's a different network type; BetaOne distinguishes PPO vs MCTS/POMCP
    based on method and the pomcp flag.
    """
    if getattr(config, "network_type", "betaone") == "decknet":
        dn = config.training.get("decknet", {})
        sims = dn.get("mcts_sims", "?")
        return f"DeckNet-{sims}"
    if config.method == "ppo":
        return "PPO"
    mcts = config.training.get("mcts", {})
    sims = mcts.get("num_sims", "?")
    prefix = "POMCP" if mcts.get("pomcp", False) else "MCTS"
    return f"{prefix}-{sims}"


def _parse_overrides(override_strs: list[str] | None) -> dict:
    """Parse 'key=value' strings into a dict, coercing types."""
    if not override_strs:
        return {}
    result = {}
    for s in override_strs:
        if "=" not in s:
            print(f"Warning: ignoring malformed override '{s}' (expected key=value)")
            continue
        key, val_str = s.split("=", 1)
        # Coerce value types
        if val_str.lower() in ("true", "false"):
            val = val_str.lower() == "true"
        elif val_str.lower() == "none":
            val = None
        else:
            try:
                val = int(val_str)
            except ValueError:
                try:
                    val = float(val_str)
                except ValueError:
                    val = val_str
        result[key] = val
    return result


def cmd_create(args):
    """Create a new experiment in a sibling git worktree.

    --no-worktree skips the worktree step and writes the config in-place
    (the current working tree's experiments/<name>/). This is what the
    outer call uses when it subprocess-invokes this CLI from within the
    newly-created worktree's venv — at which point "in-place" means the
    worktree's dir, which is where we wanted it.
    """
    import sys as _sys
    overrides = _parse_overrides(args.override)

    if args.no_worktree:
        # Inner call (running inside worktree venv): just write the config.
        exp = Experiment.create(args.name, template=args.template, overrides=overrides)
        config = exp.config
        print(f"Created experiment: {config.name}")
        print(f"  Method: {config.method}")
        print(f"  Dir: {exp.dir}")
        if config.description:
            print(f"  Description: {config.description}")
        return

    # Outer call (running from main): set up the worktree, then delegate.
    from .experiment import (
        _create_worktree, _setup_worktree_venv, _activation_hint,
        _experiment_worktree_path, _is_our_worktree,
    )

    worktree_root = _experiment_worktree_path(args.name)
    worktree_solver = worktree_root / "sts2-solver"
    exp_config = worktree_solver / "experiments" / args.name / "config.yaml"

    if worktree_root.exists():
        # Existing worktree: either ours (resume partial setup) or a stray
        # dir we shouldn't touch.
        if not _is_our_worktree(worktree_root, args.name):
            print(f"Error: {worktree_root} exists but isn't a git worktree "
                  f"on branch 'experiment/{args.name}'. Either a stray dir "
                  f"or a worktree for a different experiment. Clean up "
                  f"manually or pick a different name.")
            _sys.exit(1)
        if exp_config.exists():
            print(f"Resuming: worktree + config exist. Verifying venv/engine...")
        else:
            print(f"Resuming: worktree exists, config missing. Completing setup...")
    else:
        print(f"Creating worktree for experiment '{args.name}'...")
        try:
            worktree_solver = _create_worktree(args.name, base_branch="main")
        except (FileExistsError, RuntimeError) as e:
            print(f"Error: {e}")
            _sys.exit(1)
        print(f"  worktree: {worktree_solver}")

    print("Setting up venv + sts2_engine (idempotent)...")
    try:
        _setup_worktree_venv(worktree_solver)
    except Exception as e:
        print(f"Venv setup failed: {e}")
        print(f"Re-run `sts2-experiment repair {args.name}` after fixing.")
        _sys.exit(1)

    # Delegate config creation to the worktree's own CLI (so paths resolve
    # to the worktree's experiments/<name>/). Skip if config already exists
    # (resume case).
    if exp_config.exists():
        print(f"Config already exists at {exp_config} (skip write)")
    else:
        import sys, subprocess
        if sys.platform == "win32":
            venv_python = worktree_solver / ".venv" / "Scripts" / "python.exe"
        else:
            venv_python = worktree_solver / ".venv" / "bin" / "python"
        inner_args = [
            str(venv_python), "-m", "sts2_solver.betaone.experiment_cli",
            "create", args.name, "--template", args.template,
            "--no-worktree",
        ]
        for ov in (args.override or []):
            inner_args.extend(["--override", ov])
        subprocess.run(inner_args, cwd=str(worktree_solver), check=True)

    print()
    print(_activation_hint(worktree_solver))


def cmd_train(args):
    exp = Experiment(args.name)
    if not exp.exists:
        print(f"Experiment '{args.name}' not found. Create it first with 'create'.")
        sys.exit(1)

    config = exp.config
    kwargs = config.to_train_kwargs()
    kwargs["output_dir"] = exp.output_dir()

    if args.cold_start:
        # Only BetaOne training respects a cold_start flag; DeckNet always
        # trains its current in-memory net, no checkpoint resume yet.
        if config.network_type != "decknet":
            kwargs["cold_start"] = True

    print(f"Training experiment: {config.name} "
          f"(network={config.network_type}, method={config.method})")
    print(f"  Output: {exp.output_dir()}")

    # Dispatch to the right training entry point
    if config.network_type == "decknet":
        from ..decknet.train import train
    elif config.method == "mcts_selfplay":
        from .selfplay_train import train
    else:
        from .train import train

    train(**kwargs)


def cmd_fork(args):
    """Fork a new experiment from an existing one, in a new worktree.

    Branches off the source's experiment branch (experiment/<source>) if
    the source lives in a worktree, else off main. The new worktree gets
    its own venv + sts2_engine wheel like `create` does.

    --no-worktree: skip worktree setup; legacy in-place fork (runs the
    classic Experiment.fork logic in the current cwd).
    """
    import sys as _sys
    overrides = _parse_overrides(args.override)

    if args.no_worktree:
        # Legacy in-place fork (or inner call from the outer worktree flow).
        exp = Experiment.fork(
            args.name,
            source_name=args.source,
            checkpoint=args.checkpoint or "auto",
            overrides=overrides,
        )
        config = exp.config
        print(f"Forked experiment: {config.name}")
        print(f"  From: {config.parent} (checkpoint: {config.parent_checkpoint})")
        print(f"  Dir: {exp.dir}")
        return

    # Outer call: worktree-aware fork.
    from .experiment import (
        _create_worktree, _setup_worktree_venv, _activation_hint,
        _experiment_worktree_path, _experiment_branch, _is_our_worktree,
    )
    import subprocess

    worktree_root = _experiment_worktree_path(args.name)
    worktree_solver = worktree_root / "sts2-solver"
    exp_config = worktree_solver / "experiments" / args.name / "config.yaml"

    if worktree_root.exists():
        if not _is_our_worktree(worktree_root, args.name):
            print(f"Error: {worktree_root} exists but isn't a git worktree "
                  f"on branch 'experiment/{args.name}'.")
            _sys.exit(1)
        print(f"Resuming partial fork for '{args.name}'...")
    else:
        # Pick the base branch: if source has an experiment/<source> branch,
        # fork off that (carries the source's code changes into the child).
        source_branch = _experiment_branch(args.source)
        from .experiment import REPO_ROOT, _run_git
        try:
            _run_git(["rev-parse", "--verify", source_branch], cwd=REPO_ROOT)
            base = source_branch
        except RuntimeError:
            base = "main"
        print(f"Forking '{args.name}' from '{args.source}' (base branch: {base})...")
        try:
            worktree_solver = _create_worktree(args.name, base_branch=base)
        except (FileExistsError, RuntimeError) as e:
            print(f"Error: {e}")
            _sys.exit(1)
        print(f"  worktree: {worktree_solver}")

    print("Setting up venv + sts2_engine (idempotent)...")
    try:
        _setup_worktree_venv(worktree_solver)
    except Exception as e:
        print(f"Venv setup failed: {e}")
        print(f"Re-run `sts2-experiment repair {args.name}` after fixing.")
        _sys.exit(1)

    # Delegate the actual fork (config + checkpoint copy) to the worktree's
    # CLI. Skip if the fork config already exists (resume case).
    if exp_config.exists():
        print(f"Config already exists at {exp_config} (skip fork)")
    else:
        import sys, subprocess
        if sys.platform == "win32":
            venv_python = worktree_solver / ".venv" / "Scripts" / "python.exe"
        else:
            venv_python = worktree_solver / ".venv" / "bin" / "python"
        inner_args = [
            str(venv_python), "-m", "sts2_solver.betaone.experiment_cli",
            "fork", args.name, "--from", args.source,
            "--no-worktree",
        ]
        if args.checkpoint:
            inner_args.extend(["--checkpoint", args.checkpoint])
        for ov in (args.override or []):
            inner_args.extend(["--override", ov])
        subprocess.run(inner_args, cwd=str(worktree_solver), check=True)

    print()
    print(_activation_hint(worktree_solver))


def cmd_repair(args):
    """Re-run venv + sts2_engine setup on an existing experiment worktree.

    Use after an interrupted `create` or `fork`, or when the installed
    sts2_engine wheel is out of sync with the worktree's Rust source (e.g.
    you pulled new commits on experiment/<name> that touched sts2-engine/).
    Idempotent: skips steps that are already done.
    """
    import sys as _sys
    from .experiment import (
        _setup_worktree_venv, _experiment_worktree_path, _is_our_worktree,
        _activation_hint,
    )

    worktree_root = _experiment_worktree_path(args.name)
    if not worktree_root.exists():
        print(f"Error: no worktree for '{args.name}' at {worktree_root}. "
              f"Use `sts2-experiment create {args.name}` instead.")
        _sys.exit(1)
    if not _is_our_worktree(worktree_root, args.name):
        print(f"Error: {worktree_root} exists but isn't a git worktree on "
              f"branch 'experiment/{args.name}'.")
        _sys.exit(1)

    worktree_solver = worktree_root / "sts2-solver"
    print(f"Repairing setup for '{args.name}' at {worktree_solver}...")
    try:
        _setup_worktree_venv(worktree_solver, rebuild_engine=args.rebuild_engine)
    except Exception as e:
        print(f"Repair failed: {e}")
        _sys.exit(1)
    print()
    print(_activation_hint(worktree_solver))


def cmd_benchmark(args):
    exp = Experiment(args.name)
    if not exp.exists:
        print(f"Experiment '{args.name}' not found.")
        sys.exit(1)

    from .benchmark import benchmark_checkpoint
    from .encounter_set import load_encounter_set
    from .suite import get_encounter_set_suite

    try:
        ckpt_path_p = exp.resolve_checkpoint(args.checkpoint)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    if not ckpt_path_p.exists():
        print(f"Error: checkpoint not found at {ckpt_path_p}")
        sys.exit(1)
    ckpt_path = str(ckpt_path_p)
    # For downstream save_benchmark logging: record resolved gen, not spec.
    resolved_label = ckpt_path_p.name.replace("betaone_", "").replace(".pt", "")

    es_name = (args.encounter_set
               or exp.config.data.get("encounter_set")
               or exp.config.data.get("training_set"))
    if not es_name:
        print("No encounter set specified. Use --encounter-set <name> or set in config.")
        sys.exit(1)

    encounters = load_encounter_set(es_name)
    _, sid = get_encounter_set_suite(es_name)

    # Pull MCTS inference config from the experiment's training config so the
    # model is benchmarked under the same search settings it was optimized for.
    # Defaulting to stock MCTS here would silently mismatch (c_puct=2.5,
    # pomcp=False, turn_boundary_eval=False) and corrupt every benchmark.
    mcts_cfg = exp.config.training.get("mcts", {}) if exp.config.method == "mcts_selfplay" else {}
    c_puct = float(mcts_cfg.get("c_puct", 2.5))
    pomcp = bool(mcts_cfg.get("pomcp", False))
    turn_boundary_eval = bool(mcts_cfg.get("turn_boundary_eval", False))
    pw_k = float(mcts_cfg.get("pw_k", 1.0))
    if args.pw_k is not None:
        pw_k = args.pw_k

    print(f"Benchmarking: {exp.config.name}")
    print(f"  Checkpoint: {ckpt_path_p.name} (resolved from '{args.checkpoint}')")
    print(f"  Encounter set: {es_name} ({len(encounters)} encounters)")
    print(f"  repeats: {args.repeats}, sims: {args.sims}")
    print(f"  MCTS config: c_puct={c_puct}, pomcp={pomcp}, "
          f"turn_boundary_eval={turn_boundary_eval}, pw_k={pw_k}")

    # Incremental save: on_progress fires after each HP batch with a delta
    # dict. exp.save_benchmark accumulates by dedup key, so partial progress
    # persists to results.jsonl continuously — halting mid-benchmark with
    # Ctrl+C keeps everything completed up to the most recent HP batch.
    def _save_partial(partial):
        exp.save_benchmark(partial, suite_id=sid, checkpoint=resolved_label)

    results = benchmark_checkpoint(
        ckpt_path,
        encounter_set=encounters,
        mode="mcts",
        repeats=args.repeats,
        num_sims=args.sims,
        c_puct=c_puct,
        pomcp=pomcp,
        turn_boundary_eval=turn_boundary_eval,
        pw_k=pw_k,
        on_progress=_save_partial,
    )

    # Don't save cumulative results again — incremental saves already
    # accumulated everything through on_progress callbacks.
    print(f"  {len(results)} result(s) saved incrementally -> {exp.benchmarks_dir / 'results.jsonl'}")


def cmd_eval(args):
    exp = Experiment(args.name)
    if not exp.exists:
        print(f"Experiment '{args.name}' not found.")
        sys.exit(1)

    config = exp.config

    # --- DeckNet eval: separate harness with its own scenarios ---
    if config.network_type == "decknet":
        import json as _json
        from ..decknet.eval import run_eval as decknet_run_eval
        from ..decknet.network import DeckNet
        import torch

        ckpt_path = str(exp.dir / "decknet_latest.pt")
        if args.checkpoint and args.checkpoint != "latest":
            ckpt_path = str(exp.dir / f"decknet_{args.checkpoint}.pt")

        from .paths import BENCHMARK_DIR
        card_vocab = _json.loads((BENCHMARK_DIR / "card_vocab.json").read_text(encoding="utf-8"))
        net = DeckNet(num_cards=len(card_vocab))
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            net.load_state_dict(ckpt["model_state_dict"])
            gen = ckpt.get("gen", "?")
        else:
            print(f"No checkpoint at {ckpt_path} — evaluating untrained network.")
            gen = 0

        print(f"Evaluating DeckNet: {config.name} (gen {gen})")
        result = decknet_run_eval(net, card_vocab, verbose=True)
        result["gen"] = gen
        # Save to same eval.jsonl location as BetaOne for TUI visibility
        exp.save_eval(result, suite_id="decknet-phase0")
        print(f"\nDeckNet eval saved: {result['passed']}/{result['total']} "
              f"({result['passed']/max(result['total'],1):.1%})")
        print(f"  -> {exp.benchmarks_dir / 'eval.jsonl'}")
        return

    # --- BetaOne eval: existing combat-scenario harness ---
    from .eval import run_eval, run_value_eval
    from .suite import get_current_eval_suite

    try:
        ckpt_path_p = exp.resolve_checkpoint(args.checkpoint)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    if not ckpt_path_p.exists():
        print(f"Error: checkpoint not found at {ckpt_path_p}")
        sys.exit(1)
    ckpt_path = str(ckpt_path_p)

    # Compute and save suite definition
    _, suite_id = get_current_eval_suite()

    print(f"Evaluating: {exp.config.name} (suite: {suite_id})")
    result = run_eval(ckpt_path)

    # Save to experiment with suite tag
    exp.save_eval(result, suite_id=suite_id)
    print(f"\nPolicy eval saved: {result['passed']}/{result['total']} "
          f"({result['passed']/max(result['total'],1):.1%})")
    print(f"  Suite: {suite_id}")
    print(f"  -> {exp.benchmarks_dir / 'eval.jsonl'}")

    # Value head eval
    value_result = run_value_eval(ckpt_path)
    exp.save_value_eval(value_result, suite_id=suite_id)
    print(f"\nValue eval saved: {value_result['passed']}/{value_result['total']} "
          f"({value_result['passed']/max(value_result['total'],1):.1%})")
    print(f"  -> {exp.benchmarks_dir / 'value_eval.jsonl'}")


def cmd_compare(args):
    names = args.names
    rows = []
    for name in names:
        exp = Experiment(name)
        if not exp.exists:
            print(f"Experiment '{name}' not found, skipping.")
            continue
        info = exp.info()
        config = info["config"]
        progress = info["progress"] or {}
        eval_result = info.get("latest_eval")
        value_eval_result = info.get("latest_value_eval")
        # For concluded experiments, show the concluded gen in the Gen column
        # instead of the latest training gen — readers care about the canonical
        # state, not where training happened to stop.
        if info["is_concluded"]:
            gen_display = f"{config.concluded_gen}*"
        else:
            gen_display = str(progress.get("gen", "?"))
        # Suite signature = "<P-total>/<V-total>". The stored suite_id hash
        # historically only covered policy scenarios; V-Eval totals could
        # drift silently. Showing raw totals makes drift instantly visible.
        p_total = eval_result.get("total") if eval_result else None
        v_total = value_eval_result.get("total") if value_eval_result else None
        if p_total or v_total:
            suite_short = f"{p_total or '-'}/{v_total or '-'}"
        else:
            suite_short = "-"
        rows.append({
            "name": name,
            "method": _format_method(config),
            "params": config.architecture.get("total_params", "?"),
            "gen": gen_display,
            "wr": progress.get("win_rate", "?"),
            "best": progress.get("best_win_rate", "?"),
            "p_eval": eval_result.get("score") if eval_result else None,
            "v_eval": (value_eval_result["passed"] / max(value_eval_result["total"], 1)
                       if value_eval_result else None),
            "suite": suite_short,
            "conf_bad": eval_result.get("conf_bad") if eval_result else None,
            "close_bad": eval_result.get("close_bad") if eval_result else None,
            "bad_count": eval_result.get("bad_count") if eval_result else None,
        })

    if not rows:
        print("No experiments found.")
        return

    # Print comparison table
    has_conf = any(r.get("conf_bad") is not None for r in rows)
    header = (f"\n{'Experiment':<28s} {'Method':<12s} {'Params':>8s} {'Gen':>5s} "
              f"{'WR':>7s} {'Best':>7s} {'P-Eval':>7s} {'V-Eval':>7s} {'Suite':>9s}")
    if has_conf:
        header += f" {'CBad':>5s} {'Close':>6s}"
    print(header)
    print("-" * (99 + (13 if has_conf else 0)))
    for r in rows:
        wr = f"{r['wr']:.1%}" if isinstance(r["wr"], float) else str(r["wr"])
        best = f"{r['best']:.1%}" if isinstance(r["best"], float) else str(r["best"])
        gen = str(r["gen"])
        params = f"{r['params']:,}" if isinstance(r["params"], int) else str(r["params"])
        pev = f"{r['p_eval']:.1%}" if r["p_eval"] is not None else "  -"
        vev = f"{r['v_eval']:.1%}" if r["v_eval"] is not None else "  -"
        line = (f"  {r['name']:<26s} {r['method']:<12s} {params:>8s} {gen:>5s} "
                f"{wr:>7s} {best:>7s} {pev:>7s} {vev:>7s} {r['suite']:>9s}")
        if has_conf:
            cb = r.get("conf_bad")
            xb = r.get("close_bad")
            bc = r.get("bad_count")
            cb_str = f"{cb}/{bc}" if cb is not None and bc is not None else "  -"
            xb_str = f"{xb}/{bc}" if xb is not None and bc is not None else "  -"
            line += f" {cb_str:>5s} {xb_str:>6s}"
        print(line)
    if any("*" in r["gen"] for r in rows):
        print("  (* = finalized at concluded_gen; scores shown are at that gen)")
    # The Suite column shows P-total/V-total. Distinct signatures mean the
    # scenario counts differ and scores are NOT apples-to-apples. Re-run
    # `sts2-experiment eval <name>` at the concluded gen to align them.
    distinct_suites = {r["suite"] for r in rows if r["suite"] != "-"}
    if len(distinct_suites) > 1:
        print(f"  (WARNING: {len(distinct_suites)} different scenario counts "
              f"— scores are NOT apples-to-apples; re-run eval to align)")


def cmd_list(args):
    experiments = Experiment.list_all()
    if not experiments:
        print("No experiments found.")
        return

    print(f"\n{'Experiment':<30s} {'Method':<15s} {'Gen':>6s} {'WR':>7s} {'Best':>7s}")
    print("-" * 72)
    for e in experiments:
        wr = f"{e['win_rate']:.1%}" if e["win_rate"] else "  -"
        best = f"{e['best_win_rate']:.1%}" if e["best_win_rate"] else "  -"
        if e.get("concluded_gen") is not None:
            gen_str = f"{e['concluded_gen']}*"
        else:
            gen_str = str(e["gen"])
        print(f"  {e['name']:<28s} {e['method']:<15s} {gen_str:>6s} {wr:>7s} {best:>7s}")
    if any(e.get("concluded_gen") is not None for e in experiments):
        print("  (* = finalized at concluded_gen)")


def cmd_info(args):
    exp = Experiment(args.name)
    if not exp.exists:
        print(f"Experiment '{args.name}' not found.")
        sys.exit(1)

    info = exp.info()
    config = info["config"]
    progress = info["progress"]
    bench = info["latest_benchmark"]

    if info["is_concluded"]:
        print(f"[CONCLUDED @ gen {config.concluded_gen}] {config.name}")
        if config.concluded_reason:
            print(f"  Reason: {config.concluded_reason}")
        if config.concluded_at:
            print(f"  Finalized: {config.concluded_at}")
    else:
        print(f"Experiment: {config.name}")
    print(f"  Method: {config.method}")
    print(f"  Created: {config.created}")
    if config.description:
        print(f"  Description: {config.description}")
    if config.parent:
        print(f"  Forked from: {config.parent} ({config.parent_checkpoint})")
    print(f"  Dir: {exp.dir}")

    arch = config.architecture
    params = arch.get("total_params", "?")
    params_str = f"{params:,}" if isinstance(params, int) else str(params)
    print(f"\nArchitecture (v{arch.get('arch_version', '?')}, {params_str} params):")
    print(f"  State: ({arch.get('state_dim', '?')}) -> trunk({arch.get('trunk_input', '?')}->{arch.get('hidden_dim', '?')}) -> policy({arch.get('action_hidden', '?')}), value(1)")
    print(f"  Hand: embed({arch.get('card_embed_dim', '?')}) + stats({arch.get('card_stats_dim', '?')}) -> attn({arch.get('hand_proj_dim', '?')})")
    print(f"  Cards: {arch.get('num_cards', '?')} vocab, relics: {arch.get('relic_dim', '?')} flags")

    print(f"\nTraining:")
    for k, v in config.training.items():
        if isinstance(v, dict):
            print(f"  {k}:")
            for kk, vv in v.items():
                print(f"    {kk}: {vv}")
        else:
            print(f"  {k}: {v}")

    if progress:
        print(f"\nProgress:")
        print(f"  Gen: {progress.get('gen', '?')}")
        print(f"  Win rate: {progress.get('win_rate', 0):.1%}")
        print(f"  Best: {progress.get('best_win_rate', 0):.1%}")
        if "tier_name" in progress:
            print(f"  Tier: {progress['tier_name']}")
    else:
        print(f"\nProgress: no training data yet")

    if bench:
        print(f"\nLatest benchmark:")
        for k, v in bench.items():
            if k != "timestamp" and k != "checkpoint":
                print(f"  {k}: {v:.1%}" if isinstance(v, float) else f"  {k}: {v}")

    eval_result = info.get("latest_eval")
    if eval_result:
        score = eval_result.get("score", 0)
        passed = eval_result.get("passed", 0)
        total = eval_result.get("total", 0)
        gen = eval_result.get("gen", "?")
        label = "Eval at concluded gen" if info["is_concluded"] else "Latest eval"
        print(f"\n{label} (gen {gen}): {passed}/{total} ({score:.1%})")
        # Confidence profile (present on eval.jsonl entries written after 2026-04-21).
        conf_clean = eval_result.get("conf_clean")
        conf_bad = eval_result.get("conf_bad")
        close_bad = eval_result.get("close_bad")
        bad_count = eval_result.get("bad_count")
        if conf_clean is not None or conf_bad is not None:
            parts = []
            if conf_clean is not None:
                parts.append(f"conf_clean {conf_clean}/{passed}")
            if conf_bad is not None and bad_count is not None:
                parts.append(f"conf_bad {conf_bad}/{bad_count}")
            if close_bad is not None and bad_count is not None:
                parts.append(f"close_bad {close_bad}/{bad_count}")
            print(f"  confidence: {'  '.join(parts)}")
        by_cat = eval_result.get("by_category", {})
        if by_cat:
            for cat, counts in sorted(by_cat.items()):
                p, t = counts["passed"], counts["total"]
                status = "ok" if p == t else f"{p}/{t}"
                print(f"  {cat}: {status}")

    value_result = info.get("latest_value_eval")
    if value_result:
        v_passed = value_result.get("passed", 0)
        v_total = value_result.get("total", 0)
        v_gen = value_result.get("gen", "?")
        v_score = v_passed / max(v_total, 1)
        v_label = "Value eval at concluded gen" if info["is_concluded"] else "Latest value eval"
        print(f"\n{v_label} (gen {v_gen}): {v_passed}/{v_total} ({v_score:.1%})")


def cmd_calibrate(args):
    """Run offline calibration to produce an immutable training set."""
    from .calibrate import calibrate_all
    from .packages import PACKAGES, calibrate_packages
    from .data_utils import build_monster_data_json, load_solver_json, build_card_vocab
    from .network import BetaOneNetwork, export_onnx, network_kwargs_from_meta
    from .training_set import save_training_set, TRAINING_SETS_DIR
    from .paths import BENCHMARK_DIR
    import torch

    # Load checkpoint
    if args.checkpoint:
        exp = Experiment(args.checkpoint)
        if exp.exists:
            ckpt_path = str(exp.dir / "betaone_latest.pt")
        else:
            ckpt_path = args.checkpoint
    else:
        print("Must specify --checkpoint (experiment name or path)")
        sys.exit(1)

    print(f"Calibrating from: {ckpt_path}")
    print(f"  Sims: {args.sims}, Combats/HP: {args.combats}")

    # Load model and export ONNX
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    vocab_path = BENCHMARK_DIR / "card_vocab.json"
    import json as _json
    card_vocab = _json.loads(vocab_path.read_text(encoding="utf-8"))
    card_vocab_json = _json.dumps(card_vocab)
    net = BetaOneNetwork(
        num_cards=len(card_vocab),
        **network_kwargs_from_meta(ckpt.get("arch_meta")),
    )
    net.load_state_dict(ckpt["model_state_dict"])
    onnx_dir = str(TRAINING_SETS_DIR / "_calibration_onnx")
    onnx_path = export_onnx(net, onnx_dir)

    monster_json = build_monster_data_json()
    profiles_json = load_solver_json("enemy_profiles.json")

    # Calibrate recorded encounters
    rec_path = BENCHMARK_DIR / "benchmark_recorded.jsonl"
    if rec_path.exists():
        import json
        with open(rec_path, encoding="utf-8") as f:
            records = [json.loads(l) for l in f if l.strip()]
        print(f"\nCalibrating {len(records)} recorded encounters...")
        calibrated_records, avg_hp = calibrate_all(
            records, monster_json, profiles_json,
            card_vocab_json, onnx_path,
            num_sims=args.sims, combats=args.combats,
        )
    else:
        calibrated_records = []

    # Calibrate packages
    print(f"\nCalibrating archetype packages...")
    calibrate_packages(monster_json, profiles_json, card_vocab_json, onnx_path,
                       num_sims=args.sims, combats=args.combats)

    # Collect package HPs
    package_hps = {}
    for pkg in PACKAGES:
        if pkg.calibrated_hps:
            package_hps[pkg.name] = dict(pkg.calibrated_hps)

    # Save training set
    name = args.name or f"calibrated-from-{args.checkpoint}"
    ts_id = save_training_set(
        name=name,
        recorded_encounters=calibrated_records,
        package_hps=package_hps,
        calibrated_with={
            "checkpoint": args.checkpoint,
            "gen": ckpt.get("gen", "?"),
            "method": "mcts",
            "sims": args.sims,
            "combats_per_hp": args.combats,
        },
    )
    print(f"\nTraining set saved: {ts_id}")
    print(f"  Recorded: {len(calibrated_records)} encounters")
    print(f"  Packages: {sum(len(v) for v in package_hps.values())} encounter-HP pairs")
    print(f"\nUse in experiment config:")
    print(f"  data:")
    print(f"    training_set: {ts_id}")


def cmd_encounter_sets(args):
    from .encounter_set import list_encounter_sets
    sets = list_encounter_sets()
    # Also show legacy training sets
    from .training_set import list_training_sets
    legacy = list_training_sets()

    if not sets and not legacy:
        print("No encounter sets. Create one with: sts2-experiment generate --checkpoint <name>")
        return

    if sets:
        print(f"{'Name':<28s} {'ID':<18s} {'Count':>6s} {'HP':>5s} {'Source'}")
        print("-" * 80)
        for es in sets:
            src = es.get("source", {})
            src_str = src.get("calibrated_with", {}).get("checkpoint", "?") if isinstance(src, dict) else "?"
            name = es.get("name", "")
            eid = es.get("encounter_set_id", "?")
            eid_short = eid[:16] if len(eid) > 16 else eid
            print(f"  {name:<26s} {eid_short:<18s} {es.get('encounter_count', 0):>6d} {es.get('avg_hp', 0):>5.1f}   {src_str}")

    if legacy:
        print(f"\nLegacy training sets (use encounter sets instead):")
        for ts in legacy:
            cal = ts.get("calibrated_with", {})
            src = cal.get("checkpoint", "?")
            name = ts.get("name", "")
            print(f"  {name:<26s} rec={ts.get('recorded_count', 0)} pkg={ts.get('packages_count', 0)}  from {src}")


def cmd_generate(args):
    """Generate an encounter set from packages + recorded encounters."""
    from .generate_encounters import generate_combined
    from .encounter_set import save_encounter_set
    from .data_utils import build_monster_data_json, load_solver_json
    from .network import BetaOneNetwork, export_onnx, network_kwargs_from_meta
    from .paths import BENCHMARK_DIR
    import torch

    # Load checkpoint
    if args.checkpoint:
        exp = Experiment(args.checkpoint)
        if exp.exists:
            ckpt_path = str(exp.dir / "betaone_latest.pt")
        else:
            ckpt_path = args.checkpoint
    else:
        print("Must specify --checkpoint (experiment name or path)")
        sys.exit(1)

    include_recorded = not args.packages_only
    include_packages = not args.recorded_only
    print(f"Generating encounter set from: {ckpt_path}")
    print(f"  Recorded: {'yes' if include_recorded else 'no'}")
    print(f"  Packages: {'yes' if include_packages else 'no'}" +
          (f" ({args.decks_per} decks/encounter)" if include_packages else ""))
    print(f"  Combats/HP: {args.combats} (policy calibration)")

    # Export ONNX
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    import json as _json
    vocab_path = BENCHMARK_DIR / "card_vocab.json"
    card_vocab = _json.loads(vocab_path.read_text(encoding="utf-8"))
    card_vocab_json = _json.dumps(card_vocab)
    net = BetaOneNetwork(
        num_cards=len(card_vocab),
        **network_kwargs_from_meta(ckpt.get("arch_meta")),
    )
    net.load_state_dict(ckpt["model_state_dict"])
    from .encounter_set import ENCOUNTER_SETS_DIR
    onnx_dir = str(ENCOUNTER_SETS_DIR / "_calibration_onnx")
    onnx_path = export_onnx(net, onnx_dir)

    monster_json = build_monster_data_json()
    profiles_json = load_solver_json("enemy_profiles.json")

    recorded_path = str(BENCHMARK_DIR / "benchmark_recorded.jsonl") if include_recorded else None

    from .generate_encounters import generate_from_packages, generate_from_recorded
    encounters = []

    if include_recorded and recorded_path:
        import os
        if os.path.exists(recorded_path):
            import json
            with open(recorded_path, encoding="utf-8") as f:
                records = [json.loads(l) for l in f if l.strip()]
            print(f"\nCalibrating {len(records)} recorded encounters...")
            encounters.extend(generate_from_recorded(
                records, monster_json, profiles_json, card_vocab_json, onnx_path,
                combats=args.combats,
                potion_rate=args.potion_rate,
                potion_max_count=args.potion_max,
            ))

    if include_packages:
        package_names = None
        if args.packages:
            package_names = [p.strip() for p in args.packages.split(",") if p.strip()]
            print(f"\nGenerating package encounters from {package_names} ({args.decks_per} decks/encounter)...")
        else:
            print(f"\nGenerating package encounters ({args.decks_per} decks/encounter)...")
        if args.potion_rate > 0:
            print(f"  potion_rate={args.potion_rate}, potion_max={args.potion_max}")
        encounters.extend(generate_from_packages(
            monster_json, profiles_json, card_vocab_json, onnx_path,
            decks_per=args.decks_per,
            combats=args.combats,
            package_names=package_names,
            potion_rate=args.potion_rate,
            potion_max_count=args.potion_max,
        ))

    es_id = save_encounter_set(
        name=args.name,
        encounters=encounters,
        source={
            "type": "combined",
            "calibrated_with": {
                "checkpoint": args.checkpoint,
                "gen": ckpt.get("gen", "?"),
                "method": "policy",
                "combats_per_hp": args.combats,
                "decks_per_encounter": args.decks_per,
            },
        },
    )

    print(f"\nEncounter set saved: {es_id}")
    print(f"  Name: {args.name}")
    print(f"  Encounters: {len(encounters)}")
    print(f"\nUse in experiment config:")
    print(f"  data:")
    print(f"    encounter_set: {args.name}")


def cmd_suites(args):
    from .suite import list_suites, get_current_final_exam_suite, get_current_recorded_suite, get_current_eval_suite

    if args.refresh:
        _, fe_id = get_current_final_exam_suite()
        _, rec_id = get_current_recorded_suite()
        _, eval_id = get_current_eval_suite()
        print(f"Current final-exam suite: {fe_id}")
        print(f"Current recorded suite:   {rec_id}")
        print(f"Current eval suite:       {eval_id}")
        print()

    suites = list_suites()
    if not suites:
        print("No suites registered. Run with --refresh to compute current suites.")
        return

    print(f"{'Suite ID':<30s} {'Type':<12s} {'Details'}")
    print("-" * 75)
    for s in suites:
        sid = s.get("suite_id", "?")
        stype = s.get("type", "?")
        if stype == "final-exam":
            details = f"seed={s.get('seed')}, combats={s.get('combats')}, pool={s.get('encounter_pool_hash', '?')}"
        elif stype == "recorded":
            details = f"{s.get('recorded_count', 0)} encounters, {s.get('combats_per', 32)} combats/each"
        elif stype == "eval":
            details = f"{s.get('scenario_count', '?')} scenarios, {len(s.get('categories', []))} categories"
        else:
            details = ""
        print(f"  {sid:<28s} {stype:<12s} {details}")


def cmd_dashboard(args):
    from .tui import main as tui_main
    tui_main()


def cmd_archive(args):
    """Archive an experiment.

    Keeps the record (config, PLAN, benchmarks, history, progress, vocab) plus
    the concluded-gen checkpoint in experiments/_archive/<name>/ on main.
    Drops non-concluded-gen checkpoints, .venv/, train.log, and onnx exports.

    For worktree experiments, also removes the worktree (branch stays in git
    so full history is recoverable via `git worktree add <path> experiment/<name>`).

    Requires the experiment to be finalized unless --force is passed.
    """
    from .experiment import _experiment_branch, _experiment_worktree_path

    exp = Experiment(args.name)
    if not exp.exists:
        print(f"Experiment '{args.name}' not found.")
        sys.exit(1)
    try:
        result = exp.archive(force=args.force)
    except (ValueError, FileExistsError, RuntimeError) as e:
        print(f"Error: {e}")
        sys.exit(1)

    mb = result["kept_bytes"] / (1024 * 1024)
    print(f"Archived {args.name}: {len(result['kept'])} files kept ({mb:.1f} MB)")
    print(f"  Dest: {result['dest']}")
    for f in result["kept"]:
        print(f"    {f}")
    if result["source_kind"] == "worktree":
        branch = _experiment_branch(args.name)
        worktree = _experiment_worktree_path(args.name)
        print(f"  Removed: worktree (branch '{branch}' retained)")
        print(f"  To restore: git worktree add {worktree} {branch}")
    else:
        print(f"  Removed: in-tree dir")


def cmd_finalize(args):
    exp = Experiment(args.name)
    if not exp.exists:
        print(f"Experiment '{args.name}' not found.")
        sys.exit(1)
    try:
        exp.finalize(args.gen, args.reason)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    cfg = exp.config
    print(f"Finalized {cfg.name} at gen {cfg.concluded_gen}")
    print(f"  Reason: {cfg.concluded_reason}")
    info = exp.info()
    e = info.get("latest_eval")
    v = info.get("latest_value_eval")
    if e:
        print(f"  Decision eval: {e['passed']}/{e['total']} ({e.get('score',0):.1%})")
    if v:
        print(f"  Value eval:    {v['passed']}/{v['total']} ({v['passed']/max(v['total'],1):.1%})")


def cmd_promote(args):
    """Promote an experiment checkpoint to the production frontier.

    Thin wrapper around scripts/promote_to_frontier.py so `sts2-experiment
    promote <name> <gen>` works the same as running the script directly.
    """
    from pathlib import Path as _Path
    import sys as _sys
    script_dir = _Path(__file__).resolve().parents[3] / "scripts"
    _sys.path.insert(0, str(script_dir))
    try:
        import promote_to_frontier
    except ImportError as e:
        print(f"Error: could not load promote_to_frontier module: {e}")
        _sys.exit(1)
    _sys.exit(promote_to_frontier.promote(args.name, args.gen, dry_run=args.dry_run))


def cmd_session(args):
    """Spawn a new Claude Code session in an experiment's worktree.

    Opens a fresh terminal window with cwd set to the worktree's sts2-solver
    dir, activates its .venv, and runs `claude --dangerously-skip-permissions`.
    User types /training <name> themselves in the new session to pull in
    status + framing context.

    --dangerously-skip-permissions default rationale: experiment worktrees are
    isolated (per-worktree venv, scoped branch). Fast iteration matters more
    than per-file permission prompts inside that bounded scope. Use
    --no-skip-permissions to opt back into prompts if you want them.
    """
    import sys as _sys
    import subprocess as _subprocess
    from .experiment import _experiment_worktree_path

    worktree_root = _experiment_worktree_path(args.name)
    worktree_solver = worktree_root / "sts2-solver"
    if not worktree_solver.exists():
        print(f"Error: no worktree for '{args.name}' at {worktree_root}.")
        print(f"Create it first: sts2-experiment create {args.name} --template <t>")
        print(f"Or fork:           sts2-experiment fork {args.name} --from <src>")
        _sys.exit(1)

    claude_flags = "" if args.no_skip_permissions else " --dangerously-skip-permissions"
    first_prompt_hint = f"/training {args.name}"
    plan_path = worktree_solver / "experiments" / args.name / "PLAN.md"
    if plan_path.exists():
        first_prompt_hint += "   # then: read experiments/{}/PLAN.md".format(args.name)

    if _sys.platform == "win32":
        # `start cmd /k "..."` opens a new cmd window that stays open after
        # the chained command finishes. /k keeps the shell so if claude
        # exits, the user can relaunch it without losing their terminal.
        batch = (
            f'cd /d "{worktree_solver}" && '
            f'call .venv\\Scripts\\activate.bat && '
            f'claude{claude_flags}'
        )
        print(f"Spawning new Claude session in worktree '{args.name}'...")
        print(f"  worktree: {worktree_solver}")
        print(f"  flags: claude{claude_flags}")
        print(f"  suggested first prompt: {first_prompt_hint}")
        _subprocess.Popen(
            ["cmd", "/c", "start", "Claude: " + args.name, "cmd", "/k", batch],
            creationflags=_subprocess.CREATE_NEW_CONSOLE,
        )
        # Also print the exact shell sequence as a fallback. Useful if the
        # spawn doesn't visibly open a window (sandboxed tool contexts,
        # remote SSH, etc.) — user can paste this into any terminal.
        print()
        print("If no terminal window appears, paste these into one manually:")
        print(f"  cd /d {worktree_solver}")
        print(f"  call .venv\\Scripts\\activate.bat")
        print(f"  claude{claude_flags}")
    else:
        # macOS/Linux terminal spawning varies by environment. Print the
        # commands for copy-paste rather than trying to detect the right
        # terminal emulator and getting it wrong.
        print(f"# Run these in a new terminal to start a Claude session in '{args.name}':")
        print(f"cd {worktree_solver}")
        print(f"source .venv/bin/activate")
        print(f"claude{claude_flags}")
        print(f"# then type: {first_prompt_hint}")


def _csv_args(values: list[str] | None) -> list[str]:
    out: list[str] = []
    for raw in values or []:
        out.extend(part.strip() for part in str(raw).split(",") if part.strip())
    return out


def _print_json(payload) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def cmd_worker_image_build(args):
    from . import worker_orchestration as workers

    try:
        plan = workers.build_worker_image(
            experiment=args.name,
            repository=args.repository,
            tag_prefix=args.tag_prefix,
            push=args.push,
            ecr_login_enabled=args.ecr_login,
            regions=_csv_args(args.region),
            ensure_repository=args.ensure_repository,
            gen=args.gen,
            allow_dirty=args.allow_dirty,
            dry_run=args.dry_run,
        )
    except (ValueError, subprocess.CalledProcessError) as e:
        print(f"Error: {e}")
        sys.exit(1)

    print(f"Worker image for {args.name}:")
    print(f"  git_sha: {plan.git_sha}")
    print(f"  source:  {plan.solver_root}")
    print(f"  image:   {plan.image}")
    if plan.images_by_region:
        print("  regions:")
        for region, image in plan.images_by_region.items():
            print(f"    {region}: {image}")
    if args.dry_run:
        print("  commands:")
        for cmd in plan.commands:
            print("    " + " ".join(str(x) for x in cmd))


def cmd_worker_image_list(args):
    from . import worker_orchestration as workers

    records = workers.load_image_records(args.name)
    if args.json:
        _print_json(records)
        return
    if not records:
        print(f"No worker images recorded for {args.name}.")
        return
    for record in records[-args.limit:]:
        pushed = "pushed" if record.get("pushed") else "local"
        print(f"{record.get('built_at', '?')}  {record.get('git_sha', '?')[:12]}  {pushed}")
        print(f"  {record.get('image')}")
        images = record.get("images_by_region") or {}
        if isinstance(images, dict) and images:
            for region, image in images.items():
                print(f"  {region}: {image}")


def _resolve_worker_image(args):
    from . import worker_orchestration as workers

    if args.image != "auto":
        return args.image, None
    record = workers.latest_image_record(args.name, gen=args.gen)
    if not record:
        hint = f" for gen {args.gen}" if args.gen is not None else ""
        raise ValueError(
            f"no recorded worker image{hint}; run "
            f"`sts2-experiment worker-image build {args.name} --push ...` first"
        )
    return str(record["image"]), record


def _make_workers_plan(args):
    from . import worker_orchestration as workers

    image, record = _resolve_worker_image(args)
    config = workers.load_capacity_config(args.config)
    return workers.make_launch_plan(
        experiment=args.name,
        max_workers=args.max_workers,
        config=config,
        regions=_csv_args(args.region),
        instance_types=_csv_args(args.instance_type),
        image=image,
        image_record=record,
        coordinator_url=args.coordinator,
        threads_per_worker=args.threads_per_worker,
        worker_count=args.worker_count,
        market=args.market,
    )


def _print_launch_plan(plan) -> None:
    print(f"Worker launch plan for {plan.experiment}:")
    print(f"  target workers: {plan.max_workers}")
    print(f"  planned workers: {plan.planned_workers}")
    print(f"  coordinator: {plan.coordinator_url}")
    print(f"  market: {plan.market}")
    for idx, unit in enumerate(plan.units, 1):
        print(
            f"  [{idx}] {unit.region} {unit.instance_type}: "
            f"{unit.workers} worker(s), threads={unit.threads_per_worker}"
        )
        print(f"      image: {unit.image}")
        print(f"      ami: {unit.ami}")
        if unit.subnet_id:
            print(f"      subnet: {unit.subnet_id}")


def cmd_workers_plan(args):
    try:
        plan = _make_workers_plan(args)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    _print_launch_plan(plan)


def cmd_workers_launch(args):
    from . import worker_orchestration as workers

    try:
        plan = _make_workers_plan(args)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    _print_launch_plan(plan)

    tailscale_key = args.tailscale_auth_key or os.environ.get(args.tailscale_auth_key_env)
    if not tailscale_key and not args.dry_run:
        print(
            f"Error: Tailscale auth key required. Set {args.tailscale_auth_key_env} "
            "or pass --tailscale-auth-key."
        )
        sys.exit(1)
    if not tailscale_key:
        tailscale_key = "tskey-placeholder"

    try:
        responses = workers.launch_ec2_workers(
            plan,
            tailscale_auth_key=tailscale_key,
            worker_group=args.worker_group,
            lease_s=args.lease_s,
            idle_sleep_s=args.idle_sleep_s,
            dry_run=args.dry_run,
        )
    except subprocess.CalledProcessError as e:
        print(f"Error launching EC2 workers: {e}")
        sys.exit(1)

    if args.json:
        _print_json(responses)
        return
    if args.dry_run:
        print("Dry run only; no instances launched.")
        return
    launched = []
    for response in responses:
        for inst in response.get("Instances", []):
            launched.append((response.get("_region"), inst.get("InstanceId"), inst.get("InstanceType")))
    print(f"Launched {len(launched)} EC2 instance(s).")
    for region, instance_id, instance_type in launched:
        print(f"  {region}: {instance_id} ({instance_type})")


def cmd_workers_user_data(args):
    from . import worker_orchestration as workers

    image, record = _resolve_worker_image(args)
    git_sha = str((record or {}).get("git_sha") or workers.experiment_git_sha(args.name))
    worker_image = image
    if args.region:
        worker_image = workers._image_for_region(image, record, args.region)
    tailscale_key = args.tailscale_auth_key or os.environ.get(args.tailscale_auth_key_env) or "tskey-placeholder"
    script = workers.render_worker_user_data(
        experiment=args.name,
        coordinator_url=args.coordinator,
        tailscale_auth_key=tailscale_key,
        worker_image=worker_image,
        worker_git_sha=git_sha,
        aws_region=args.region,
        threads_per_worker=args.threads_per_worker,
        worker_count=int(args.worker_count),
        worker_group=args.worker_group,
        lease_s=args.lease_s,
        idle_sleep_s=args.idle_sleep_s,
    )
    if args.output:
        from pathlib import Path as _Path
        _Path(args.output).write_text(script, encoding="utf-8", newline="\n")
        print(f"Wrote user-data script: {args.output}")
    else:
        print(script)


def cmd_workers_cost(args):
    from . import worker_orchestration as workers

    try:
        summary = workers.estimate_ec2_cost(
            experiment=args.name,
            config=workers.load_capacity_config(args.config),
            regions=_csv_args(args.region),
            default_hourly_price=args.default_hourly_price,
            include_terminated=not args.active_only,
            include_recorded=not args.no_recorded,
        )
    except (ValueError, subprocess.CalledProcessError) as e:
        print(f"Error: {e}")
        sys.exit(1)

    if not args.no_record:
        workers.record_cost_snapshot(args.name, summary)

    if args.json:
        _print_json(summary)
        return

    print(f"Estimated EC2 worker cost for {args.name}:")
    print(f"  total:       ${summary['estimated_total_cost']:.4f}")
    print(f"  hourly burn: ${summary['estimated_hourly_burn']:.4f}/hr")
    print(f"  instances:   {summary['instance_count']} ({summary['active_instance_count']} active)")
    if summary["unknown_price_count"]:
        print(f"  unknown prices: {summary['unknown_price_count']} instance(s)")
    print()
    print("  Instance                 Region      Type          State        Market     Hours    $/hr       Cost")
    print("  -----------------------------------------------------------------------------------------------")
    for item in summary["instances"]:
        price = "?" if item["hourly_price"] is None else f"{item['hourly_price']:.4f}"
        cost = "?" if item["cost"] is None else f"{item['cost']:.4f}"
        print(
            f"  {item['instance_id']:<24} "
            f"{item['region']:<11} "
            f"{item['instance_type']:<13} "
            f"{item['state']:<12} "
            f"{item['market']:<9} "
            f"{item['hours']:>7.2f} "
            f"{price:>8} "
            f"{cost:>9}"
        )
    if not args.no_record:
        print()
        print("  Snapshot recorded in the experiment cost ledger.")


def cmd_coordinator_start(args):
    from . import worker_orchestration as workers

    record = workers.start_coordinator(
        experiment=args.name,
        host=args.host,
        port=args.port,
        static=args.static,
    )
    print(f"Started companion coordinator for {args.name}:")
    print(f"  pid: {record['pid']}")
    print(f"  url: {record['url']}")
    print(f"  log: {record['log']}")


def cmd_coordinator_status(args):
    from . import worker_orchestration as workers

    status = workers.coordinator_status(args.name, url=args.url)
    if args.json:
        _print_json(status)
        return
    if status["healthy"]:
        print(f"Coordinator healthy: {status['url']}")
    else:
        print(f"Coordinator not healthy: {status['url']}")
        print(f"  {status.get('error')}")
    record = status.get("record") or {}
    if record:
        print(f"  pid: {record.get('pid')}")
        print(f"  log: {record.get('log')}")


def cmd_coordinator_stop(args):
    from . import worker_orchestration as workers

    try:
        record = workers.stop_coordinator(args.name)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    print(f"Stopped coordinator for {args.name} (pid {record.get('pid')}).")


def cmd_ship(args):
    """Sync a finalized worktree experiment's data back to main.

    Copies config.yaml + benchmark/eval jsonl rows from the worktree into
    main's experiments/<name>/. Does NOT merge code — that's a separate
    decision. Doesn't delete the branch or worktree either; `archive`
    cleans those up when you're ready.
    """
    import sys as _sys
    import shutil as _shutil
    from .experiment import (
        _experiment_worktree_path, _experiment_branch,
        EXPERIMENTS_DIR,
    )

    exp = Experiment(args.name)
    if not exp.exists:
        print(f"Experiment '{args.name}' not found.")
        _sys.exit(1)
    if exp.config.concluded_gen is None:
        print(f"Error: '{args.name}' is not finalized. Run "
              f"`sts2-experiment finalize {args.name} --gen <N> --reason \"...\"` first.")
        _sys.exit(1)

    worktree_path = _experiment_worktree_path(args.name)
    worktree_exp_dir = worktree_path / "sts2-solver" / "experiments" / args.name
    if not worktree_exp_dir.exists():
        print(f"Note: no worktree at {worktree_path} — experiment's data "
              f"already lives in main at {exp.dir}. Nothing to sync.")
        return

    dest_dir = EXPERIMENTS_DIR / args.name
    dest_bench = dest_dir / "benchmarks"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_bench.mkdir(exist_ok=True)

    # Copy config (contains concluded_gen + concluded_reason).
    _shutil.copy2(worktree_exp_dir / "config.yaml", dest_dir / "config.yaml")
    copied = ["config.yaml"]

    # Copy the three benchmark/eval jsonl files if present.
    for fname in ["results.jsonl", "eval.jsonl", "value_eval.jsonl"]:
        src = worktree_exp_dir / "benchmarks" / fname
        if src.exists():
            _shutil.copy2(src, dest_bench / fname)
            copied.append(f"benchmarks/{fname}")

    print(f"Synced finalized data for {args.name} to main:")
    print(f"  from: {worktree_exp_dir}")
    print(f"  to:   {dest_dir}")
    print(f"  files: {', '.join(copied)}")
    print()
    print("Not done: code on the experiment branch has NOT been merged.")
    print("If the code changes should ship to main, review + merge yourself:")
    print(f"  git diff main..{_experiment_branch(args.name)} -- sts2-solver/src sts2-solver/sts2-engine")
    print(f"  git merge {_experiment_branch(args.name)}    # or cherry-pick the commits you want")
    print()
    print("To reclaim worktree disk:")
    print(f"  sts2-experiment archive {args.name}")


def cmd_unfinalize(args):
    exp = Experiment(args.name)
    if not exp.exists:
        print(f"Experiment '{args.name}' not found.")
        sys.exit(1)
    if exp.config.concluded_gen is None:
        print(f"{args.name} is not finalized; nothing to clear.")
        return
    exp.unfinalize()
    print(f"Unfinalized: {args.name}")


def main():
    # Reconfigure stdout/stderr to UTF-8 so non-ASCII characters in scenario
    # labels / descriptions don't crash the eval printer on Windows cp1252.
    import sys as _sys
    for _stream in (_sys.stdout, _sys.stderr):
        if hasattr(_stream, "reconfigure"):
            try:
                _stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass

    parser = argparse.ArgumentParser(
        prog="sts2-experiment",
        description="BetaOne experiment management",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # create
    p = sub.add_parser("create",
                        help="Create a new experiment (in a sibling git worktree)")
    p.add_argument("name", help="Experiment name")
    p.add_argument("--template", "-t", required=True,
                    help="Template name (e.g., mcts_selfplay, ppo)")
    p.add_argument("--override", "-o", nargs="*", default=[],
                    help="Override config values (key=value)")
    p.add_argument("--no-worktree", action="store_true",
                    help="Skip worktree creation; write config in-place. "
                         "Internal use (the outer call delegates here from "
                         "inside the new worktree's venv).")
    p.set_defaults(func=cmd_create)

    # train
    p = sub.add_parser("train", help="Train an experiment (resumes from checkpoint)")
    p.add_argument("name", help="Experiment name")
    p.add_argument("--cold-start", action="store_true",
                    help="Ignore existing checkpoint, start fresh")
    p.set_defaults(func=cmd_train)

    # fork
    p = sub.add_parser("fork",
                        help="Fork a new experiment from an existing one (in a new worktree)")
    p.add_argument("name", help="New experiment name")
    p.add_argument("--from", dest="source", required=True,
                    help="Source experiment name")
    p.add_argument("--checkpoint", default="auto",
                    help="Checkpoint to fork from. 'auto' (default) = concluded "
                         "gen if source is finalized, else latest. Also accepts "
                         "'finalized'/'concluded' (requires source is finalized), "
                         "'latest', or a specific 'genN'.")
    p.add_argument("--override", "-o", nargs="*", default=[],
                    help="Override config values (key=value)")
    p.add_argument("--no-worktree", action="store_true",
                    help="Skip worktree creation; fork in-place in the current "
                         "working tree's experiments/. Internal use (the outer "
                         "call delegates here from inside the new worktree's venv).")
    p.set_defaults(func=cmd_fork)

    # repair
    p = sub.add_parser("repair",
                        help="Re-run venv + sts2_engine setup on an existing "
                             "experiment worktree (idempotent; use after an "
                             "interrupted create/fork or when Rust source changed).")
    p.add_argument("name", help="Experiment name")
    p.add_argument("--no-rebuild-engine", dest="rebuild_engine",
                    action="store_false", default=True,
                    help="Skip the maturin develop step if sts2_engine is "
                         "already importable. Default: always rebuild.")
    p.set_defaults(func=cmd_repair)

    # eval
    p = sub.add_parser("eval", help="Run eval harness and save results")
    p.add_argument("name", help="Experiment name")
    p.add_argument("--checkpoint", default="auto",
                    help="Checkpoint to evaluate. 'auto' (default) = finalized "
                         "gen if set, else latest. Also accepts 'finalized'/"
                         "'concluded', 'latest', or 'genN'.")
    p.set_defaults(func=cmd_eval)

    # benchmark
    p = sub.add_parser("benchmark", help="Benchmark against an encounter set")
    p.add_argument("name", help="Experiment name")
    p.add_argument("--encounter-set", default=None,
                    help="Encounter set name (default: from experiment config)")
    p.add_argument("--repeats", type=int, default=1,
                    help="Times to repeat each encounter (default: 1)")
    p.add_argument("--checkpoint", default="auto",
                    help="Checkpoint to benchmark. 'auto' (default) = finalized "
                         "gen if set, else latest. Also accepts 'finalized'/"
                         "'concluded', 'latest', or 'genN'.")
    p.add_argument("--sims", type=int, default=400,
                    help="MCTS simulations per decision (default: 400)")
    p.add_argument("--pw-k", type=float, default=None,
                    help="Progressive widening multiplier for POMCP chance "
                         "nodes (overrides config; higher = wider)")
    p.set_defaults(func=cmd_benchmark)

    # compare
    p = sub.add_parser("compare", help="Compare experiments")
    p.add_argument("names", nargs="+", help="Experiment names to compare")
    p.set_defaults(func=cmd_compare)

    # list
    p = sub.add_parser("list", help="List all experiments")
    p.set_defaults(func=cmd_list)

    # info
    p = sub.add_parser("info", help="Show detailed experiment info")
    p.add_argument("name", help="Experiment name")
    p.set_defaults(func=cmd_info)

    # calibrate
    p = sub.add_parser("calibrate", help="Run offline calibration to produce a training set")
    p.add_argument("--checkpoint", required=True,
                    help="Experiment name or checkpoint path to calibrate against")
    p.add_argument("--name", default=None,
                    help="Name for the training set")
    p.add_argument("--sims", type=int, default=400,
                    help="MCTS sims for calibration (default: 400)")
    p.add_argument("--combats", type=int, default=64,
                    help="Combats per HP level (default: 64)")
    p.set_defaults(func=cmd_calibrate)

    # generate
    p = sub.add_parser("generate", help="Generate an encounter set")
    p.add_argument("name", help="Friendly name for the encounter set")
    p.add_argument("--checkpoint", required=True,
                    help="Experiment name or checkpoint path to calibrate against")
    p.add_argument("--packages-only", action="store_true",
                    help="Only generate from archetype packages (no recorded)")
    p.add_argument("--recorded-only", action="store_true",
                    help="Only generate from recorded death encounters (no packages)")
    p.add_argument("--decks-per", type=int, default=3,
                    help="Random deck variants per package encounter (default: 3)")
    p.add_argument("--combats", type=int, default=64,
                    help="Combats per HP level in calibration (default: 64)")
    p.add_argument("--packages", default=None,
                    help="Comma-separated package names to restrict generation to "
                         "(e.g. synergy_grand_finale,synergy_finisher). Default: all.")
    p.add_argument("--potion-rate", type=float, default=0.0,
                    help="Fraction of encounters that get a non-empty potion "
                         "inventory (default 0 = no potions, preserves prior sets).")
    p.add_argument("--potion-max", type=int, default=2,
                    help="Max potions per encounter when sampled (default 2, game allows 3).")
    p.set_defaults(func=cmd_generate)

    # encounter-sets
    p = sub.add_parser("encounter-sets", help="List available encounter sets")
    p.set_defaults(func=cmd_encounter_sets)

    # training-sets (legacy alias)
    p = sub.add_parser("training-sets", help="List training sets (legacy, use encounter-sets)")
    p.set_defaults(func=cmd_encounter_sets)

    # suites
    p = sub.add_parser("suites", help="List benchmark suites")
    p.add_argument("--refresh", action="store_true",
                    help="Compute and register current suites")
    p.set_defaults(func=cmd_suites)

    # dashboard
    p = sub.add_parser("dashboard", help="Live experiment dashboard TUI")
    p.set_defaults(func=cmd_dashboard)

    # archive
    p = sub.add_parser("archive", help="Archive an experiment")
    p.add_argument("name", help="Experiment name")
    p.add_argument(
        "--force", action="store_true",
        help="Archive without a pinned concluded_gen (no checkpoint retained)",
    )
    p.set_defaults(func=cmd_archive)

    # finalize
    p = sub.add_parser("finalize",
                       help="Mark a gen as the experiment's canonical conclusion")
    p.add_argument("name", help="Experiment name")
    p.add_argument("--gen", type=int, required=True,
                    help="Generation to promote (must have a betaone_genN.pt)")
    p.add_argument("--reason", required=True,
                    help="Short note on why this gen is the conclusion")
    p.set_defaults(func=cmd_finalize)

    # unfinalize
    p = sub.add_parser("unfinalize",
                       help="Clear the finalized-gen marker on an experiment")
    p.add_argument("name", help="Experiment name")
    p.set_defaults(func=cmd_unfinalize)

    # promote
    p = sub.add_parser("promote",
                       help="Promote an experiment gen to the production frontier "
                            "(copies checkpoint into betaone_checkpoints/, writes FRONTIER.md)")
    p.add_argument("name", help="Experiment name")
    p.add_argument("gen", type=int, help="Generation to promote (must have betaone_genN.pt)")
    p.add_argument("--dry-run", action="store_true",
                   help="Print what would happen without modifying anything")
    p.set_defaults(func=cmd_promote)

    # ship
    p = sub.add_parser("ship",
                       help="Sync a finalized worktree experiment's data to main")
    p.add_argument("name", help="Experiment name (must be finalized)")
    p.set_defaults(func=cmd_ship)

    # session
    p = sub.add_parser("session",
                       help="Spawn a new Claude Code session in an experiment's worktree")
    p.add_argument("name", help="Experiment name (must have a worktree)")
    p.add_argument("--no-skip-permissions", action="store_true",
                    help="Launch claude without --dangerously-skip-permissions "
                         "(default is to skip permissions, since worktrees are "
                         "scoped and we want fast iteration).")
    p.set_defaults(func=cmd_session)

    # worker-image
    p = sub.add_parser("worker-image",
                       help="Build and record immutable distributed worker images")
    wimg = p.add_subparsers(dest="worker_image_command", required=True)

    b = wimg.add_parser("build", help="Build a worker image for an experiment worktree")
    b.add_argument("name", help="Experiment name")
    b.add_argument("--repository", required=True,
                   help="Image repository without tag, e.g. acct.dkr.ecr.us-east-1.amazonaws.com/sts2-worker")
    b.add_argument("--tag-prefix", default=None,
                   help="Image tag prefix (default: experiment name)")
    b.add_argument("--push", action="store_true", help="Push the built image")
    b.add_argument("--ecr-login", action="store_true",
                   help="Login to ECR before pushing")
    b.add_argument("--region", action="append", default=[],
                   help="Region to push to; repeat or comma-separate for region diversity")
    b.add_argument("--ensure-repository", action="store_true",
                   help="Create missing ECR repositories before pushing")
    b.add_argument("--gen", type=int, default=None,
                   help="Require compatibility with an already scheduled generation")
    b.add_argument("--allow-dirty", action="store_true",
                   help="Allow uncommitted source changes in fingerprint-affecting paths")
    b.add_argument("--dry-run", action="store_true",
                   help="Print docker commands without running them")
    b.set_defaults(func=cmd_worker_image_build)

    l = wimg.add_parser("list", help="List recorded worker images for an experiment")
    l.add_argument("name", help="Experiment name")
    l.add_argument("--limit", type=int, default=10)
    l.add_argument("--json", action="store_true")
    l.set_defaults(func=cmd_worker_image_list)

    # workers
    p = sub.add_parser("workers",
                       help="Plan or launch EC2 distributed workers for an experiment")
    wsp = p.add_subparsers(dest="workers_command", required=True)

    def _add_worker_capacity_args(parser):
        parser.add_argument("name", help="Experiment name")
        parser.add_argument("--max-workers", type=int, required=True,
                            help="Maximum distributed worker containers to launch")
        parser.add_argument("--config", default=None,
                            help="JSON EC2 capacity config (regions, AMIs, subnets, security groups)")
        parser.add_argument("--image", default="auto",
                            help="'auto' to use the latest recorded compatible image, or an explicit image URI")
        parser.add_argument("--gen", type=int, default=None,
                            help="When --image=auto, select the image matching this generation fingerprint")
        parser.add_argument("--coordinator", default=None,
                            help="Companion API URL reachable by workers")
        parser.add_argument("--region", action="append", default=[],
                            help="AWS region; repeat or comma-separate for region diversity")
        parser.add_argument("--instance-type", action="append", default=[],
                            help="Allowed EC2 instance type; repeat or comma-separate")
        parser.add_argument("--threads-per-worker", type=int, default=1)
        parser.add_argument("--worker-count", default="auto",
                            help="'auto' or an explicit worker count per instance")
        parser.add_argument("--market", choices=["spot", "on-demand"], default="spot")

    wp = wsp.add_parser("plan", help="Show an EC2 launch plan without launching")
    _add_worker_capacity_args(wp)
    wp.set_defaults(func=cmd_workers_plan)

    wl = wsp.add_parser("launch", help="Launch EC2 workers from a recorded image")
    _add_worker_capacity_args(wl)
    wl.add_argument("--tailscale-auth-key", default=None,
                    help="Tailscale auth key; prefer env via --tailscale-auth-key-env")
    wl.add_argument("--tailscale-auth-key-env", default="TAILSCALE_AUTH_KEY",
                    help="Environment variable containing the Tailscale auth key")
    wl.add_argument("--worker-group", default="ec2")
    wl.add_argument("--lease-s", type=float, default=240.0)
    wl.add_argument("--idle-sleep-s", type=float, default=5.0)
    wl.add_argument("--dry-run", action="store_true")
    wl.add_argument("--json", action="store_true")
    wl.set_defaults(func=cmd_workers_launch)

    wu = wsp.add_parser("user-data", help="Render one EC2 cloud-init user-data script")
    wu.add_argument("name", help="Experiment name")
    wu.add_argument("--image", default="auto")
    wu.add_argument("--gen", type=int, default=None)
    wu.add_argument("--coordinator", required=True)
    wu.add_argument("--region", default="us-east-1")
    wu.add_argument("--threads-per-worker", type=int, default=1)
    wu.add_argument("--worker-count", default="1")
    wu.add_argument("--worker-group", default="ec2")
    wu.add_argument("--lease-s", type=float, default=240.0)
    wu.add_argument("--idle-sleep-s", type=float, default=5.0)
    wu.add_argument("--tailscale-auth-key", default=None)
    wu.add_argument("--tailscale-auth-key-env", default="TAILSCALE_AUTH_KEY")
    wu.add_argument("--output", default=None)
    wu.set_defaults(func=cmd_workers_user_data)

    wc = wsp.add_parser("cost", help="Estimate running EC2 worker cost")
    wc.add_argument("name", help="Experiment name")
    wc.add_argument("--config", default=None,
                    help="JSON EC2 capacity config; may include hourly_prices overrides")
    wc.add_argument("--region", action="append", default=[],
                    help="AWS region to scan; repeat or comma-separate")
    wc.add_argument("--default-hourly-price", type=float, default=None,
                    help="Fallback hourly price when AWS/config pricing is unavailable")
    wc.add_argument("--active-only", action="store_true",
                    help="Only ask EC2 for active states, not terminated instances")
    wc.add_argument("--no-recorded", action="store_true",
                    help="Do not include prior ledger entries for instances no longer returned by EC2")
    wc.add_argument("--no-record", action="store_true",
                    help="Do not append this estimate to worker_costs.jsonl")
    wc.add_argument("--json", action="store_true")
    wc.set_defaults(func=cmd_workers_cost)

    # coordinator
    p = sub.add_parser("coordinator",
                       help="Start, stop, or check the companion coordinator")
    csp = p.add_subparsers(dest="coordinator_command", required=True)
    cs = csp.add_parser("start", help="Start the companion API in the background")
    cs.add_argument("name", help="Experiment name")
    cs.add_argument("--host", default="0.0.0.0")
    cs.add_argument("--port", type=int, default=8765)
    cs.add_argument("--static", default=None,
                    help="Optional companion-web/out static export path")
    cs.set_defaults(func=cmd_coordinator_start)

    ct = csp.add_parser("status", help="Check coordinator health")
    ct.add_argument("name", help="Experiment name")
    ct.add_argument("--url", default=None,
                    help="Override URL to check instead of the recorded one")
    ct.add_argument("--json", action="store_true")
    ct.set_defaults(func=cmd_coordinator_status)

    cx = csp.add_parser("stop", help="Stop the recorded coordinator process")
    cx.add_argument("name", help="Experiment name")
    cx.set_defaults(func=cmd_coordinator_stop)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
