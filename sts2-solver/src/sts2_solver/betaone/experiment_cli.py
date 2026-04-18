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
        _experiment_worktree_path,
    )

    if _experiment_worktree_path(args.name).exists():
        print(f"Error: worktree for '{args.name}' already exists at "
              f"{_experiment_worktree_path(args.name)}. "
              f"Use `sts2-experiment archive {args.name}` to remove it.")
        _sys.exit(1)

    print(f"Creating worktree for experiment '{args.name}'...")
    try:
        worktree_solver = _create_worktree(args.name, base_branch="main")
    except (FileExistsError, RuntimeError) as e:
        print(f"Error: {e}")
        _sys.exit(1)
    print(f"  worktree: {worktree_solver}")

    print("Setting up worktree venv + sts2_engine (takes ~30-60s)...")
    try:
        _setup_worktree_venv(worktree_solver)
    except Exception as e:
        print(f"Venv setup failed: {e}")
        print("Worktree is created but venv setup incomplete. Fix and rerun:")
        print(f"  cd {worktree_solver}")
        print(f"  python -m venv --system-site-packages .venv")
        print(f"  <activate> && pip install -e . && cd sts2-engine && maturin develop --release")
        _sys.exit(1)

    # Delegate config creation to the worktree's own CLI (so paths resolve
    # to the worktree's experiments/<name>/). The worktree's venv has the
    # newly-built sts2_engine wheel installed.
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
        _experiment_worktree_path, _experiment_branch,
    )
    import subprocess

    if _experiment_worktree_path(args.name).exists():
        print(f"Error: worktree for '{args.name}' already exists at "
              f"{_experiment_worktree_path(args.name)}. "
              f"Use `sts2-experiment archive {args.name}` first.")
        _sys.exit(1)

    # Pick the base branch: if source has an experiment/<source> branch,
    # fork off that (carries the source's code changes into the child).
    # Otherwise fork off main.
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

    print("Setting up worktree venv + sts2_engine (takes ~30-60s)...")
    try:
        _setup_worktree_venv(worktree_solver)
    except Exception as e:
        print(f"Venv setup failed: {e}")
        _sys.exit(1)

    # Delegate the actual fork (config + checkpoint copy) to the worktree's
    # CLI inside its venv, so paths resolve to the worktree's dirs.
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
        })

    if not rows:
        print("No experiments found.")
        return

    # Print comparison table
    print(f"\n{'Experiment':<28s} {'Method':<12s} {'Params':>8s} {'Gen':>5s} "
          f"{'WR':>7s} {'Best':>7s} {'P-Eval':>7s} {'V-Eval':>7s} {'Suite':>9s}")
    print("-" * 99)
    for r in rows:
        wr = f"{r['wr']:.1%}" if isinstance(r["wr"], float) else str(r["wr"])
        best = f"{r['best']:.1%}" if isinstance(r["best"], float) else str(r["best"])
        gen = str(r["gen"])
        params = f"{r['params']:,}" if isinstance(r["params"], int) else str(r["params"])
        pev = f"{r['p_eval']:.1%}" if r["p_eval"] is not None else "  -"
        vev = f"{r['v_eval']:.1%}" if r["v_eval"] is not None else "  -"
        print(f"  {r['name']:<26s} {r['method']:<12s} {params:>8s} {gen:>5s} "
              f"{wr:>7s} {best:>7s} {pev:>7s} {vev:>7s} {r['suite']:>9s}")
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
            ))

    if include_packages:
        package_names = None
        if args.packages:
            package_names = [p.strip() for p in args.packages.split(",") if p.strip()]
            print(f"\nGenerating package encounters from {package_names} ({args.decks_per} decks/encounter)...")
        else:
            print(f"\nGenerating package encounters ({args.decks_per} decks/encounter)...")
        encounters.extend(generate_from_packages(
            monster_json, profiles_json, card_vocab_json, onnx_path,
            decks_per=args.decks_per,
            combats=args.combats,
            package_names=package_names,
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
    """Remove an experiment's worktree + venv but keep the branch.

    Reclaims disk (the venv is ~300MB, checkpoints can be several GB).
    The branch stays in git so the full history is recoverable — to pick
    the experiment back up, `git worktree add <path> experiment/<name>`
    and run venv setup again.
    """
    import sys as _sys
    from .experiment import _experiment_worktree_path, _experiment_branch, _run_git

    worktree_path = _experiment_worktree_path(args.name)
    if not worktree_path.exists():
        # Fall back to legacy in-tree archive (pre-worktree experiments).
        exp = Experiment(args.name)
        if not exp.exists:
            print(f"Experiment '{args.name}' not found (no worktree, no in-tree dir).")
            _sys.exit(1)
        exp.archive()
        print(f"Archived legacy in-tree experiment: {args.name}")
        return

    # Worktree-based archive: remove the worktree (and its venv inside).
    # Git's `worktree remove` refuses if the worktree has uncommitted changes
    # or locked state — force unless the user passed --keep-uncommitted.
    from .experiment import REPO_ROOT
    print(f"Removing worktree at {worktree_path}...")
    try:
        _run_git(["worktree", "remove", "--force", str(worktree_path)], cwd=REPO_ROOT)
    except RuntimeError as e:
        print(f"Error: {e}")
        _sys.exit(1)
    print(f"Archived: worktree removed, branch {_experiment_branch(args.name)} retained.")
    print(f"To restore: git worktree add {worktree_path} {_experiment_branch(args.name)}")


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

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
