"""Data access: read JSONL / progress files across main + worktrees.

Reuses sts2_solver.betaone.experiment helpers (_all_experiment_sources,
_read_progress) so the companion never duplicates discovery logic.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

from ..betaone import experiment as exp_mod
from . import status as status_mod


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _kind(cfg) -> str:
    """Classify experiment by surface area, mirroring the TUI's table split."""
    if (cfg.method or "").startswith("distill_"):
        return "distill"
    if getattr(cfg, "network_type", None) == "decknet":
        return "decknet"
    return "betaone"


def _method_string(cfg) -> str:
    if (cfg.method or "").startswith("distill_"):
        # e.g. "distill_c51" -> "d_c51", matching the TUI shorthand
        return cfg.method.replace("distill_", "d_")
    if getattr(cfg, "network_type", None) == "decknet":
        dn = cfg.training.get("decknet", {})
        return f"DeckNet-{dn.get('mcts_sims', '?')}"
    if cfg.method == "ppo":
        return "PPO"
    mcts = cfg.training.get("mcts", {})
    prefix = "POMCP" if mcts.get("pomcp", False) else "MCTS"
    return f"{prefix}-{mcts.get('num_sims', '?')}"


def _progress_path(d: Path, cfg) -> Path:
    if getattr(cfg, "network_type", None) == "decknet":
        return d / "decknet_progress.json"
    return d / "betaone_progress.json"


def _history_path(d: Path, cfg) -> Path:
    if getattr(cfg, "network_type", None) == "decknet":
        return d / "decknet_history.jsonl"
    return d / "betaone_history.jsonl"


def _latest_pinned(rows: list[dict], concluded_gen: int | None) -> dict | None:
    """Latest eval row, pinned to concluded_gen if the experiment's finalized.

    Matches Experiment.info()'s semantics: finalized runs should report
    their canonical scores, not whatever got evaluated last.
    """
    if not rows:
        return None
    if concluded_gen is not None:
        pinned = [r for r in rows if r.get("gen") == concluded_gen]
        if pinned:
            return pinned[-1]
    return rows[-1]


def _peak_eval(rows: list[dict]) -> dict | None:
    """Row with the highest pass rate across all eval runs.

    Suites expand over time (e.g. 127 -> 128 scenarios), so pass rate is
    the right axis rather than absolute passed count. Ties broken by
    higher total (denser eval), then later gen.
    """
    scored = [
        (r, r["passed"] / r["total"])
        for r in rows
        if r.get("total") and r.get("passed") is not None
    ]
    if not scored:
        return None
    scored.sort(
        key=lambda x: (x[1], x[0].get("total", 0), x[0].get("gen", 0)),
        reverse=True,
    )
    top = scored[0][0]
    return {
        "passed": top.get("passed"),
        "total": top.get("total"),
        "gen": top.get("gen"),
        "score": scored[0][1],
    }


def _arch_params(cfg) -> int | None:
    arch = getattr(cfg, "architecture", None) or {}
    tp = arch.get("total_params") if isinstance(arch, dict) else None
    return tp if isinstance(tp, int) else None


def list_experiments(*, include_kinds: tuple[str, ...] | None = None) -> list[dict]:
    """Every experiment with a summary + liveness state.

    If `include_kinds` is given, restrict to those kinds. Default returns
    everything except distill (distill is its own tab — see list_distill()).
    """
    if include_kinds is None:
        include_kinds = ("betaone", "decknet")
    out: list[dict] = []
    for name, d in exp_mod._all_experiment_sources():
        try:
            cfg = exp_mod.ExperimentConfig.from_yaml(d / "config.yaml")
        except Exception:
            continue
        if _kind(cfg) not in include_kinds:
            continue
        progress = exp_mod._read_progress(_progress_path(d, cfg))
        history = _read_jsonl(_history_path(d, cfg))
        # Use last 5 gens of history for a robust cadence baseline.
        recent_times = [
            float(r.get("gen_time") or 0.0)
            for r in history[-5:]
        ]
        st = status_mod.classify(progress, recent_times)

        # Recent-window metrics for the card: last 10 gens
        window = history[-10:] if history else []

        def _mean(key: str):
            vals = [r.get(key) for r in window if r.get(key) is not None]
            return statistics.mean(vals) if vals else None

        def _window_delta(key: str, n: int = 10) -> float | None:
            """Mean(last n) - mean(prior n). None until history >= 2n.

            Mirrors TUI's 'window' delta-mode — smooths per-gen noise and
            shows real trend direction.
            """
            vals = [
                r.get(key) for r in history if r.get(key) is not None
            ]
            if len(vals) < 2 * n:
                return None
            cur = statistics.mean(vals[-n:])
            prev = statistics.mean(vals[-2 * n : -n])
            return cur - prev

        wr_last10 = _mean("win_rate")
        v_loss_last10 = _mean("value_loss")
        p_loss_last10 = _mean("policy_loss")
        kl_last10 = _mean("kl_mcts_net_mean")
        top1_last10 = _mean("top1_agree_mean")
        vcorr_last10 = _mean("value_corr_mean")
        wr_delta = _window_delta("win_rate")
        p_loss_delta = _window_delta("policy_loss")
        v_loss_delta = _window_delta("value_loss")
        kl_delta = _window_delta("kl_mcts_net_mean")
        top1_delta = _window_delta("top1_agree_mean")
        vcorr_delta = _window_delta("value_corr_mean")

        # Peak WR + which gen hit it. Prefer the row with strict max
        # win_rate; ties broken by later gen (more mature checkpoint).
        best_wr_row = None
        for r in history:
            wr = r.get("win_rate")
            if wr is None:
                continue
            if best_wr_row is None or wr > best_wr_row.get("win_rate", 0):
                best_wr_row = r
        best_wr = best_wr_row.get("win_rate") if best_wr_row else None
        best_wr_gen = best_wr_row.get("gen") if best_wr_row else None
        # Fall back to progress.best_win_rate when history is sparse but
        # the training loop has tracked it — we'll have no gen in that
        # case, which the card shows as "peak X%".
        if best_wr is None and progress:
            best_wr = progress.get("best_win_rate")

        # Training elapsed = sum of per-gen durations (resume-aware).
        # ETA = remaining gens * recent median cadence. Cadence reused
        # from `recent_times` above (already the last 5 gen_times).
        gen_times_all = [
            float(r.get("gen_time") or 0.0)
            for r in history
            if r.get("gen_time")
        ]
        training_elapsed_s = sum(gen_times_all) if gen_times_all else None
        total_gens = cfg.training.get("generations")
        cur_gen = progress.get("gen") if progress else None
        cadence = st.get("cadence_s")
        if (
            total_gens
            and cur_gen is not None
            and cadence
            and cur_gen < total_gens
            and st.get("state") == "RUNNING"
        ):
            training_eta_s = (total_gens - cur_gen) * cadence
        else:
            training_eta_s = None

        evals = _read_jsonl(d / "benchmarks" / "eval.jsonl")
        value_evals = _read_jsonl(d / "benchmarks" / "value_eval.jsonl")
        mcts_evals = _read_jsonl(d / "benchmarks" / "mcts_eval.jsonl")
        latest_eval = _latest_pinned(evals, cfg.concluded_gen)
        latest_value_eval = _latest_pinned(value_evals, cfg.concluded_gen)
        latest_mcts_eval = _latest_pinned(mcts_evals, cfg.concluded_gen)
        peak_eval = _peak_eval(evals)
        peak_value_eval = _peak_eval(value_evals)

        def _eval_delta(rows: list[dict], key: str, n: int = 5) -> float | None:
            vals = [r.get(key) for r in rows if r.get(key) is not None]
            if len(vals) < 2 * n:
                return None
            return statistics.mean(vals[-n:]) - statistics.mean(vals[-2 * n : -n])

        eval_delta = _eval_delta(evals, "score")
        value_eval_delta = _eval_delta(value_evals, "score")
        rescue_delta = _eval_delta(mcts_evals, "rescue_rate")

        # Sparkline-ready time series. Each is [{gen, value}] with nulls
        # filtered out. Tails capped at 60 points — enough resolution for
        # a 80px-wide sparkline, small enough to keep the payload tight.
        def _series(rows: list[dict], key: str, n: int = 60) -> list[dict]:
            out = [
                {"gen": r.get("gen"), "value": r.get(key)}
                for r in rows
                if r.get(key) is not None
            ]
            return out[-n:]

        def _eval_score_series(rows: list[dict], n: int = 60) -> list[dict]:
            out: list[dict] = []
            for r in rows:
                t = r.get("total")
                p = r.get("passed")
                if not t or p is None:
                    continue
                out.append({
                    "gen": r.get("gen"),
                    "value": p / t,
                    "passed": p,
                    "total": t,
                })
            return out[-n:]

        wr_series = _series(history, "win_rate")
        p_loss_series = _series(history, "policy_loss")
        v_loss_series = _series(history, "value_loss")
        kl_series = _series(history, "kl_mcts_net_mean")
        top1_series = _series(history, "top1_agree_mean")
        vcorr_series = _series(history, "value_corr_mean")
        eval_series = _eval_score_series(evals)
        value_eval_series = _eval_score_series(value_evals)
        rescue_series = _series(mcts_evals, "rescue_rate")

        out.append({
            "name": name,
            "kind": _kind(cfg),
            "method": _method_string(cfg),
            "finalized": cfg.concluded_gen is not None,
            "concluded_gen": cfg.concluded_gen,
            "generations_total": cfg.training.get("generations"),
            "encounter_set": (
                cfg.training.get("encounter_set")
                or cfg.training.get("mcts", {}).get("encounter_set")
                or cfg.training.get("ppo", {}).get("encounter_set")
            ),
            "params": _arch_params(cfg),
            "description": cfg.description,
            "status": st,
            "progress": progress,
            "win_rate_last10": wr_last10,
            "policy_loss_last10": p_loss_last10,
            "value_loss_last10": v_loss_last10,
            "kl_mcts_net_last10": kl_last10,
            "top1_agree_last10": top1_last10,
            "value_corr_last10": vcorr_last10,
            "gen_time_last": (history[-1].get("gen_time") if history else None),
            "parent": getattr(cfg, "parent", None),
            "best_win_rate": best_wr,
            "best_win_rate_gen": best_wr_gen,
            "training_elapsed_s": training_elapsed_s,
            "training_eta_s": training_eta_s,
            "latest_eval": latest_eval,
            "latest_value_eval": latest_value_eval,
            "latest_mcts_eval": latest_mcts_eval,
            "peak_eval": peak_eval,
            "peak_value_eval": peak_value_eval,
            "win_rate_delta": wr_delta,
            "policy_loss_delta": p_loss_delta,
            "value_loss_delta": v_loss_delta,
            "kl_mcts_net_delta": kl_delta,
            "top1_agree_delta": top1_delta,
            "value_corr_delta": vcorr_delta,
            "eval_delta": eval_delta,
            "value_eval_delta": value_eval_delta,
            "rescue_delta": rescue_delta,
            "wr_series": wr_series,
            "p_loss_series": p_loss_series,
            "v_loss_series": v_loss_series,
            "kl_series": kl_series,
            "top1_series": top1_series,
            "vcorr_series": vcorr_series,
            "eval_series": eval_series,
            "value_eval_series": value_eval_series,
            "rescue_series": rescue_series,
        })

    # Sort: running first, then stalled, then stopped/finalized, then alpha
    state_order = {"RUNNING": 0, "STALLED": 1, "STOPPED": 2, "UNKNOWN": 3}
    out.sort(key=lambda e: (
        state_order.get(e["status"]["state"], 9),
        e["finalized"],
        e["name"],
    ))
    return out


def get_experiment(name: str) -> dict | None:
    """Detail view for a single experiment."""
    for n, d in exp_mod._all_experiment_sources():
        if n != name:
            continue
        try:
            cfg = exp_mod.ExperimentConfig.from_yaml(d / "config.yaml")
        except Exception:
            return None
        progress = exp_mod._read_progress(_progress_path(d, cfg))
        history = _read_jsonl(_history_path(d, cfg))
        evals = _read_jsonl(d / "benchmarks" / "eval.jsonl")
        value_evals = _read_jsonl(d / "benchmarks" / "value_eval.jsonl")
        mcts_evals = _read_jsonl(d / "benchmarks" / "mcts_eval.jsonl")
        bench = _read_jsonl(d / "benchmarks" / "results.jsonl")

        recent_times = [float(r.get("gen_time") or 0.0) for r in history[-5:]]
        st = status_mod.classify(progress, recent_times)

        return {
            "name": name,
            "dir": str(d),
            "method": _method_string(cfg),
            "description": cfg.description,
            "finalized": cfg.concluded_gen is not None,
            "concluded_gen": cfg.concluded_gen,
            "concluded_reason": getattr(cfg, "concluded_reason", None),
            "parent": getattr(cfg, "parent", None),
            "generations_total": cfg.training.get("generations"),
            "params": _arch_params(cfg),
            "encounter_set": (
                cfg.training.get("encounter_set")
                or cfg.training.get("mcts", {}).get("encounter_set")
                or cfg.training.get("ppo", {}).get("encounter_set")
            ),
            "config": {
                "method": cfg.method,
                "training": cfg.training,
            },
            "status": st,
            "progress": progress,
            "history_tail": history[-50:],
            "eval_history": evals,
            "value_eval_history": value_evals,
            "mcts_eval_history": mcts_evals,
            "benchmarks": bench,
        }
    return None


def all_benchmarks() -> list[dict]:
    """Flatten results.jsonl across every experiment for cross-comparison."""
    rows: list[dict] = []
    for name, d in exp_mod._all_experiment_sources():
        try:
            cfg = exp_mod.ExperimentConfig.from_yaml(d / "config.yaml")
        except Exception:
            continue
        for r in _read_jsonl(d / "benchmarks" / "results.jsonl"):
            rows.append({
                "experiment": name,
                "finalized": cfg.concluded_gen is not None,
                "concluded_gen": cfg.concluded_gen,
                **r,
            })
    return rows


def _collect_eval_rows(filename: str) -> list[dict]:
    rows: list[dict] = []
    for name, d in exp_mod._all_experiment_sources():
        for r in _read_jsonl(d / "benchmarks" / filename):
            rows.append({"experiment": name, **r})
    return rows


def leaderboard() -> dict:
    """Best experiment+gen for each eval category, separately for P-Eval and V-Eval.

    A "top score" is the row maximizing pass/total within a category; ties
    broken by higher total (prefers denser evaluations) then later gen.
    """
    def by_category(rows: list[dict]) -> dict[str, list[dict]]:
        cat_rows: dict[str, list[dict]] = {}
        for r in rows:
            for cat, v in (r.get("by_category") or {}).items():
                # New format: {passed: int, total: int}.
                # Legacy format (distill-*): list of per-scenario records,
                # each {name, passed: bool, ...}.
                if isinstance(v, dict):
                    passed = v.get("passed")
                    total = v.get("total") or 0
                elif isinstance(v, list):
                    total = len(v)
                    passed = sum(
                        1 for s in v if isinstance(s, dict) and s.get("passed")
                    )
                else:
                    continue
                if total <= 0 or passed is None:
                    continue
                cat_rows.setdefault(cat, []).append({
                    "experiment": r["experiment"],
                    "gen": r.get("gen"),
                    "passed": passed,
                    "total": total,
                    "score": passed / total,
                    "suite": r.get("suite"),
                    "timestamp": r.get("timestamp"),
                })
        # sort each category by score desc, then total desc, then gen desc
        for cat in cat_rows:
            cat_rows[cat].sort(
                key=lambda x: (x["score"], x["total"], x["gen"] or 0),
                reverse=True,
            )
        return cat_rows

    p_rows = _collect_eval_rows("eval.jsonl")
    v_rows = _collect_eval_rows("value_eval.jsonl")

    def totals(rows: list[dict]) -> list[dict]:
        # Overall best (total passed / total) per experiment+gen
        totals = [{
            "experiment": r["experiment"],
            "gen": r.get("gen"),
            "passed": r.get("passed"),
            "total": r.get("total"),
            "score": (r.get("passed") / r["total"]) if r.get("total") else None,
            "suite": r.get("suite"),
        } for r in rows if r.get("total")]
        totals.sort(key=lambda x: (x["score"] or 0, x["total"] or 0, x["gen"] or 0), reverse=True)
        return totals

    return {
        "policy_eval": {
            "overall_top": totals(p_rows)[:10],
            "by_category": by_category(p_rows),
        },
        "value_eval": {
            "overall_top": totals(v_rows)[:10],
            "by_category": by_category(v_rows),
        },
    }


def list_distill() -> list[dict]:
    """Distillation experiments with epoch-indexed metrics + dataset.

    Distillation is supervised: it iterates epochs (not gens) over a teacher
    dataset and writes distill_history.jsonl with {epoch, train/val pol+val
    losses, val_top1, time_s}. Liveness uses the same status.classify
    helper, fed the recent epoch durations from distill_history.
    """
    out: list[dict] = []
    for name, d in exp_mod._all_experiment_sources():
        try:
            cfg = exp_mod.ExperimentConfig.from_yaml(d / "config.yaml")
        except Exception:
            continue
        if _kind(cfg) != "distill":
            continue

        dhist = _read_jsonl(d / "distill_history.jsonl")
        last = dhist[-1] if dhist else {}
        recent_times = [float(r.get("time_s") or 0.0) for r in dhist[-5:]]

        # Synthesize a progress dict so status.classify can do its job —
        # uses the most recent epoch row's timestamp.
        synthetic_progress = {
            "timestamp": last.get("timestamp")
            or (d / "distill_history.jsonl").stat().st_mtime
            if (d / "distill_history.jsonl").exists()
            else None,
        } if last or (d / "distill_history.jsonl").exists() else None
        st = status_mod.classify(synthetic_progress, recent_times)

        # Recent metric averages over last 10 epochs
        window = dhist[-10:]

        def _mean(key: str):
            vals = [r.get(key) for r in window if r.get(key) is not None]
            return statistics.mean(vals) if vals else None

        epochs_total = (cfg.training or {}).get("epochs")
        data_cfg = getattr(cfg, "data", {}) or {}
        # Some configs nest 'data' inside training; fallback when needed.
        if not data_cfg and isinstance(cfg.training, dict):
            data_cfg = cfg.training.get("data", {}) or {}

        # Best val_top1 across all logged epochs (gives a "peak")
        top1_vals = [r.get("val_top1") for r in dhist if r.get("val_top1") is not None]
        best_top1 = max(top1_vals) if top1_vals else None

        # Policy-only WR benchmark, if present
        bench = _read_jsonl(d / "benchmarks" / "results.jsonl")
        pol_benches = [b for b in bench if b.get("mode") == "policy"]
        policy_wr = pol_benches[-1].get("win_rate") if pol_benches else None

        # Latest evals (mirror what the live tab surfaces)
        evals = _read_jsonl(d / "benchmarks" / "eval.jsonl")
        value_evals = _read_jsonl(d / "benchmarks" / "value_eval.jsonl")
        mcts_evals = _read_jsonl(d / "benchmarks" / "mcts_eval.jsonl")

        latest_eval = _latest_pinned(evals, cfg.concluded_gen)
        latest_value_eval = _latest_pinned(value_evals, cfg.concluded_gen)
        latest_mcts_eval = _latest_pinned(mcts_evals, cfg.concluded_gen)

        out.append({
            "name": name,
            "kind": "distill",
            "method": _method_string(cfg),
            "finalized": cfg.concluded_gen is not None,
            "concluded_epoch": cfg.concluded_gen,
            "epochs_total": epochs_total,
            "params": _arch_params(cfg),
            "description": cfg.description,
            "parent": getattr(cfg, "parent", None),
            "dataset": (
                data_cfg.get("dataset")
                or data_cfg.get("teacher")
                or data_cfg.get("mode")
            ),
            "status": st,
            "current_epoch": last.get("epoch"),
            "val_top1_last": last.get("val_top1"),
            "val_top1_best": best_top1,
            "val_top1_last10": _mean("val_top1"),
            "val_pol_loss_last10": _mean("val_pol_loss"),
            "val_val_loss_last10": _mean("val_val_loss"),
            "train_top1_last10": _mean("train_top1"),
            "epoch_time_last": last.get("time_s"),
            "policy_only_wr": policy_wr,
            "latest_eval": latest_eval,
            "latest_value_eval": latest_value_eval,
            "latest_mcts_eval": latest_mcts_eval,
        })

    state_order = {"RUNNING": 0, "STALLED": 1, "STOPPED": 2, "UNKNOWN": 3}
    out.sort(key=lambda e: (
        state_order.get(e["status"]["state"], 9),
        e["finalized"],
        e["name"],
    ))
    return out
