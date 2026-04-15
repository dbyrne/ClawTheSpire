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
import sys
import time

from .experiment import Experiment, ExperimentConfig
from .paths import EXPERIMENTS_DIR, TEMPLATES_DIR


def _format_method(config) -> str:
    """Format method string with sim count for MCTS."""
    if config.method == "ppo":
        return "PPO"
    sims = config.training.get("mcts", {}).get("num_sims", "?")
    return f"MCTS-{sims}"


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
    overrides = _parse_overrides(args.override)
    exp = Experiment.create(args.name, template=args.template, overrides=overrides)
    config = exp.config
    print(f"Created experiment: {config.name}")
    print(f"  Method: {config.method}")
    print(f"  Dir: {exp.dir}")
    if config.description:
        print(f"  Description: {config.description}")


def cmd_train(args):
    exp = Experiment(args.name)
    if not exp.exists:
        print(f"Experiment '{args.name}' not found. Create it first with 'create'.")
        sys.exit(1)

    config = exp.config
    kwargs = config.to_train_kwargs()
    kwargs["output_dir"] = exp.output_dir()

    if args.cold_start:
        kwargs["cold_start"] = True

    print(f"Training experiment: {config.name} ({config.method})")
    print(f"  Output: {exp.output_dir()}")

    if config.method == "mcts_selfplay":
        from .selfplay_train import train
    else:
        from .train import train

    train(**kwargs)


def cmd_fork(args):
    overrides = _parse_overrides(args.override)
    exp = Experiment.fork(
        args.name,
        source_name=args.source,
        checkpoint=args.checkpoint or "latest",
        overrides=overrides,
    )
    config = exp.config
    print(f"Forked experiment: {config.name}")
    print(f"  From: {config.parent} (checkpoint: {config.parent_checkpoint})")
    print(f"  Dir: {exp.dir}")


def cmd_benchmark(args):
    exp = Experiment(args.name)
    if not exp.exists:
        print(f"Experiment '{args.name}' not found.")
        sys.exit(1)

    from .benchmark import benchmark_checkpoint
    from .suite import get_current_final_exam_suite, get_current_recorded_suite

    ckpt_path = str(exp.dir / "betaone_latest.pt")
    if args.checkpoint and args.checkpoint != "latest":
        ckpt_path = str(exp.dir / f"betaone_{args.checkpoint}.pt")

    # Determine which suites to run
    suite_types = []
    if args.suite in ("final-exam", "all"):
        suite_types.append("final-exam")
    if args.suite in ("recorded", "all"):
        suite_types.append("recorded")
    if args.suite in ("training-set", "encounter-set"):
        suite_types.append("encounter-set")

    for suite_type in suite_types:
        if suite_type == "final-exam":
            _, sid = get_current_final_exam_suite()
        elif suite_type == "recorded":
            _, sid = get_current_recorded_suite()
        elif suite_type == "encounter-set":
            es_name = (args.encounter_set or args.training_set
                       or exp.config.data.get("encounter_set")
                       or exp.config.data.get("training_set"))
            if not es_name:
                print("  No encounter set specified (use --encounter-set or set in config)")
                continue
            from .suite import get_encounter_set_suite
            _, sid = get_encounter_set_suite(es_name)

        print(f"\nBenchmarking: {exp.config.name}")
        print(f"  Suite: {sid} ({suite_type})")
        print(f"  Mode: {args.mode}")

        ts_for_bench = None
        if suite_type == "encounter-set":
            ts_for_bench = (args.encounter_set or args.training_set
                            or exp.config.data.get("encounter_set")
                            or exp.config.data.get("training_set"))

        results = benchmark_checkpoint(
            ckpt_path,
            suite_type=suite_type,
            mode=args.mode,
            combats=args.combats,
            num_sims=args.sims,
            ts_id=ts_for_bench,
        )

        for result in results:
            exp.save_benchmark(result, suite_id=sid,
                               checkpoint=args.checkpoint or "latest")

        print(f"  {len(results)} result(s) saved -> {exp.benchmarks_dir / 'results.jsonl'}")


def cmd_eval(args):
    exp = Experiment(args.name)
    if not exp.exists:
        print(f"Experiment '{args.name}' not found.")
        sys.exit(1)

    from .eval import run_eval
    from .suite import get_current_eval_suite

    ckpt_path = str(exp.dir / "betaone_latest.pt")
    if args.checkpoint and args.checkpoint != "latest":
        ckpt_path = str(exp.dir / f"betaone_{args.checkpoint}.pt")

    # Compute and save suite definition
    _, suite_id = get_current_eval_suite()

    print(f"Evaluating: {exp.config.name} (suite: {suite_id})")
    result = run_eval(ckpt_path)

    # Save to experiment with suite tag
    exp.save_eval(result, suite_id=suite_id)
    print(f"\nEval saved: {result['passed']}/{result['total']} "
          f"({result['passed']/max(result['total'],1):.1%})")
    print(f"  Suite: {suite_id}")
    print(f"  -> {exp.benchmarks_dir / 'eval.jsonl'}")


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
        rows.append({
            "name": name,
            "method": _format_method(config),
            "params": config.architecture.get("total_params", "?"),
            "gen": progress.get("gen", "?"),
            "wr": progress.get("win_rate", "?"),
            "best": progress.get("best_win_rate", "?"),
            "eval": eval_result.get("score") if eval_result else None,
        })

    if not rows:
        print("No experiments found.")
        return

    # Print comparison table
    print(f"\n{'Experiment':<28s} {'Method':<12s} {'Params':>8s} {'Gen':>5s} {'WR':>7s} {'Best':>7s} {'Eval':>7s}")
    print("-" * 82)
    for r in rows:
        wr = f"{r['wr']:.1%}" if isinstance(r["wr"], float) else str(r["wr"])
        best = f"{r['best']:.1%}" if isinstance(r["best"], float) else str(r["best"])
        gen = str(r["gen"])
        params = f"{r['params']:,}" if isinstance(r["params"], int) else str(r["params"])
        ev = f"{r['eval']:.1%}" if r["eval"] is not None else "  -"
        print(f"  {r['name']:<26s} {r['method']:<12s} {params:>8s} {gen:>5s} {wr:>7s} {best:>7s} {ev:>7s}")


def cmd_list(args):
    experiments = Experiment.list_all()
    if not experiments:
        print("No experiments found.")
        return

    print(f"\n{'Experiment':<30s} {'Method':<15s} {'Gen':>5s} {'WR':>7s} {'Best':>7s}")
    print("-" * 70)
    for e in experiments:
        wr = f"{e['win_rate']:.1%}" if e["win_rate"] else "  -"
        best = f"{e['best_win_rate']:.1%}" if e["best_win_rate"] else "  -"
        print(f"  {e['name']:<28s} {e['method']:<15s} {e['gen']:>5d} {wr:>7s} {best:>7s}")


def cmd_info(args):
    exp = Experiment(args.name)
    if not exp.exists:
        print(f"Experiment '{args.name}' not found.")
        sys.exit(1)

    info = exp.info()
    config = info["config"]
    progress = info["progress"]
    bench = info["latest_benchmark"]

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
        print(f"\nLatest eval (gen {gen}): {passed}/{total} ({score:.1%})")
        by_cat = eval_result.get("by_category", {})
        if by_cat:
            for cat, counts in sorted(by_cat.items()):
                p, t = counts["passed"], counts["total"]
                status = "ok" if p == t else f"{p}/{t}"
                print(f"  {cat}: {status}")


def cmd_calibrate(args):
    """Run offline calibration to produce an immutable training set."""
    from .calibrate import calibrate_all
    from .packages import PACKAGES, calibrate_packages
    from .data_utils import build_monster_data_json, load_solver_json, build_card_vocab
    from .network import BetaOneNetwork, export_onnx
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
    net = BetaOneNetwork(num_cards=len(card_vocab))
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
    from .network import BetaOneNetwork, export_onnx
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

    print(f"Generating encounter set from: {ckpt_path}")
    print(f"  Sims: {args.sims}, Combats/HP: {args.combats}, Decks/encounter: {args.decks_per}")

    # Export ONNX
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    import json as _json
    vocab_path = BENCHMARK_DIR / "card_vocab.json"
    card_vocab = _json.loads(vocab_path.read_text(encoding="utf-8"))
    card_vocab_json = _json.dumps(card_vocab)
    net = BetaOneNetwork(num_cards=len(card_vocab))
    net.load_state_dict(ckpt["model_state_dict"])
    from .encounter_set import ENCOUNTER_SETS_DIR
    onnx_dir = str(ENCOUNTER_SETS_DIR / "_calibration_onnx")
    onnx_path = export_onnx(net, onnx_dir)

    monster_json = build_monster_data_json()
    profiles_json = load_solver_json("enemy_profiles.json")

    recorded_path = str(BENCHMARK_DIR / "benchmark_recorded.jsonl")

    encounters = generate_combined(
        monster_json, profiles_json, card_vocab_json, onnx_path,
        recorded_path=recorded_path,
        decks_per=args.decks_per,
        num_sims=args.sims,
        combats=args.combats,
    )

    es_id = save_encounter_set(
        name=args.name,
        encounters=encounters,
        source={
            "type": "combined",
            "calibrated_with": {
                "checkpoint": args.checkpoint,
                "gen": ckpt.get("gen", "?"),
                "method": "mcts",
                "sims": args.sims,
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
    exp = Experiment(args.name)
    if not exp.exists:
        print(f"Experiment '{args.name}' not found.")
        sys.exit(1)
    exp.archive()
    print(f"Archived: {args.name}")


def main():
    parser = argparse.ArgumentParser(
        prog="sts2-experiment",
        description="BetaOne experiment management",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # create
    p = sub.add_parser("create", help="Create a new experiment from a template")
    p.add_argument("name", help="Experiment name")
    p.add_argument("--template", "-t", required=True,
                    help="Template name (e.g., mcts_selfplay, ppo)")
    p.add_argument("--override", "-o", nargs="*", default=[],
                    help="Override config values (key=value)")
    p.set_defaults(func=cmd_create)

    # train
    p = sub.add_parser("train", help="Train an experiment (resumes from checkpoint)")
    p.add_argument("name", help="Experiment name")
    p.add_argument("--cold-start", action="store_true",
                    help="Ignore existing checkpoint, start fresh")
    p.set_defaults(func=cmd_train)

    # fork
    p = sub.add_parser("fork", help="Fork a new experiment from an existing one")
    p.add_argument("name", help="New experiment name")
    p.add_argument("--from", dest="source", required=True,
                    help="Source experiment name")
    p.add_argument("--checkpoint", default="latest",
                    help="Checkpoint to fork from (default: latest)")
    p.add_argument("--override", "-o", nargs="*", default=[],
                    help="Override config values (key=value)")
    p.set_defaults(func=cmd_fork)

    # eval
    p = sub.add_parser("eval", help="Run eval harness and save results")
    p.add_argument("name", help="Experiment name")
    p.add_argument("--checkpoint", default="latest",
                    help="Checkpoint to evaluate (default: latest)")
    p.set_defaults(func=cmd_eval)

    # benchmark
    p = sub.add_parser("benchmark", help="Run combat benchmark for an experiment")
    p.add_argument("name", help="Experiment name")
    p.add_argument("--suite", choices=["final-exam", "recorded", "training-set", "encounter-set", "all"], default="all",
                    help="Which suite: final-exam, recorded, encounter-set, training-set (legacy), or all")
    p.add_argument("--training-set", default=None,
                    help="Training set name/ID (legacy, use --encounter-set)")
    p.add_argument("--encounter-set", default=None,
                    help="Encounter set name/ID for --suite encounter-set")
    p.add_argument("--mode", choices=["policy", "mcts", "both"], default="both",
                    help="Inference mode: policy (network only), mcts (with search), both (default)")
    p.add_argument("--checkpoint", default="latest",
                    help="Checkpoint to benchmark (default: latest)")
    p.add_argument("--combats", type=int, default=256,
                    help="Number of combats for final exam")
    p.add_argument("--sims", type=int, default=400,
                    help="MCTS simulations per decision (default: 400)")
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
    p = sub.add_parser("generate", help="Generate an encounter set from packages + recorded")
    p.add_argument("name", help="Friendly name for the encounter set")
    p.add_argument("--checkpoint", required=True,
                    help="Experiment name or checkpoint path to calibrate against")
    p.add_argument("--decks-per", type=int, default=3,
                    help="Random deck variants per package encounter (default: 3)")
    p.add_argument("--sims", type=int, default=400,
                    help="MCTS sims for calibration (default: 400)")
    p.add_argument("--combats", type=int, default=64,
                    help="Combats per HP level in calibration (default: 64)")
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

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
