"""Background eval runner for BetaOne self-play experiments."""

from __future__ import annotations

import argparse
import json
import os
import time
from contextlib import contextmanager


@contextmanager
def _file_lock(path: str, timeout_s: float = 60.0, poll_s: float = 0.05):
    deadline = time.time() + timeout_s
    lock_fd: int | None = None
    while lock_fd is None:
        try:
            lock_fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(lock_fd, str(os.getpid()).encode("ascii"))
        except FileExistsError:
            if time.time() >= deadline:
                raise TimeoutError(f"Timed out waiting for lock: {path}")
            time.sleep(poll_s)
    try:
        yield
    finally:
        if lock_fd is not None:
            os.close(lock_fd)
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass


def append_history_record(history_path: str, record: dict) -> None:
    parent = os.path.dirname(history_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with _file_lock(history_path + ".lock"):
        with open(history_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")


def _patch_history_record(history_path: str, gen: int, updates: dict) -> None:
    if not os.path.exists(history_path):
        return
    with _file_lock(history_path + ".lock"):
        with open(history_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        patched = False
        out_lines: list[str] = []
        for line in lines:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                out_lines.append(line)
                continue
            if row.get("gen") == gen:
                row.update(updates)
                line = json.dumps(row) + "\n"
                patched = True
            out_lines.append(line)

        if not patched:
            return
        tmp_path = history_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            f.writelines(out_lines)
        os.replace(tmp_path, history_path)


def _append_jsonl(path: str, entry: dict) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with _file_lock(path + ".lock"):
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")


def run_async_eval(
    checkpoint_path: str,
    output_dir: str,
    gen: int,
    history_path: str,
    *,
    c_puct: float,
    pomcp: bool,
    turn_boundary_eval: bool,
    pw_k: float,
) -> int:
    eval_t0 = time.perf_counter()
    bench_dir = os.path.join(output_dir, "benchmarks")
    jobs_path = os.path.join(bench_dir, "eval_jobs.jsonl")
    _append_jsonl(jobs_path, {
        "timestamp": time.time(),
        "gen": gen,
        "checkpoint": checkpoint_path,
        "status": "started",
    })

    try:
        from .eval import run_eval, run_value_eval, run_mcts_eval
        from .suite import compute_eval_suite, suite_id as _suite_id

        sid = _suite_id(compute_eval_suite())
        pol = run_eval(checkpoint_path)
        val = run_value_eval(checkpoint_path)
        mce = run_mcts_eval(
            checkpoint_path,
            c_puct=c_puct,
            pomcp=pomcp,
            turn_boundary_eval=turn_boundary_eval,
            pw_k=pw_k,
        )

        pol_entry = {
            "suite": sid, "timestamp": time.time(), "gen": gen,
            "passed": pol["passed"], "total": pol["total"],
            "score": round(pol["passed"] / max(pol["total"], 1), 4),
            "end_turn_avg": pol.get("end_turn_avg"),
            "end_turn_high": pol.get("end_turn_high", 0),
            "bad_count": pol.get("bad_count"),
            "conf_bad": pol.get("conf_bad"),
            "close_bad": pol.get("close_bad"),
            "conf_clean": pol.get("conf_clean"),
            "by_category": {
                cat: {"passed": sum(1 for r in rs if r["passed"]), "total": len(rs)}
                for cat, rs in pol.get("by_category", {}).items()
            },
        }
        val_entry = {
            "suite": sid, "timestamp": time.time(), "gen": gen,
            "passed": val["passed"], "total": val["total"],
            "score": round(val["passed"] / max(val["total"], 1), 4),
            "by_category": val.get("by_category", {}),
        }
        mce_entry = {
            "suite": sid, "timestamp": time.time(), "gen": gen,
            "total": mce["total"],
            "clean": mce["clean"], "echo": mce["echo"],
            "fixed": mce["fixed"], "broke": mce["broke"],
            "mixed": mce["mixed"], "nomatch": mce["nomatch"],
            "metric": mce.get("metric", "real_rescue"),
            "real_rescue_rate": round(mce.get("real_rescue_rate", mce["rescue_rate"]), 4),
            "rescue_rate": round(mce["rescue_rate"], 4),
            "num_sims": mce.get("num_sims"),
            "c_puct": mce.get("c_puct"),
            "pomcp": mce.get("pomcp"),
            "turn_boundary_eval": mce.get("turn_boundary_eval"),
            "pw_k": mce.get("pw_k"),
        }
        _append_jsonl(os.path.join(bench_dir, "eval.jsonl"), pol_entry)
        _append_jsonl(os.path.join(bench_dir, "value_eval.jsonl"), val_entry)
        _append_jsonl(os.path.join(bench_dir, "mcts_eval.jsonl"), mce_entry)

        eval_sec = time.perf_counter() - eval_t0
        _patch_history_record(history_path, gen, {
            "eval_sec": round(eval_sec, 2),
            "eval_async_status": "done",
            "eval_async_completed_at": time.time(),
        })
        _append_jsonl(jobs_path, {
            "timestamp": time.time(),
            "gen": gen,
            "status": "done",
            "eval_sec": round(eval_sec, 2),
            "policy_score": pol_entry["score"],
            "value_score": val_entry["score"],
            "real_rescue_rate": mce_entry["real_rescue_rate"],
        })
        print(
            f"eval gen {gen}: {pol['passed']}/{pol['total']} "
            f"({pol_entry['score']:.0%}) | value: {val['passed']}/{val['total']} "
            f"({val_entry['score']:.0%}) | "
            f"mcts: CLEAN={mce['clean']} ECHO={mce['echo']} "
            f"FIXED={mce['fixed']} BROKE={mce['broke']} "
            f"(real rescue {mce['real_rescue_rate']:.0%})"
        )
        return 0
    except Exception as exc:
        eval_sec = time.perf_counter() - eval_t0
        _patch_history_record(history_path, gen, {
            "eval_sec": round(eval_sec, 2),
            "eval_async_status": "failed",
            "eval_async_error": str(exc),
            "eval_async_completed_at": time.time(),
        })
        _append_jsonl(jobs_path, {
            "timestamp": time.time(),
            "gen": gen,
            "status": "failed",
            "eval_sec": round(eval_sec, 2),
            "error": str(exc),
        })
        print(f"[eval_every] failed gen {gen}: {exc}")
        return 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a BetaOne eval job")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--history", required=True)
    parser.add_argument("--gen", type=int, required=True)
    parser.add_argument("--c-puct", type=float, default=2.5)
    parser.add_argument("--pomcp", action="store_true")
    parser.add_argument("--turn-boundary-eval", action="store_true")
    parser.add_argument("--pw-k", type=float, default=1.0)
    args = parser.parse_args()
    return run_async_eval(
        args.checkpoint,
        args.output_dir,
        args.gen,
        args.history,
        c_puct=args.c_puct,
        pomcp=args.pomcp,
        turn_boundary_eval=args.turn_boundary_eval,
        pw_k=args.pw_k,
    )


if __name__ == "__main__":
    raise SystemExit(main())
