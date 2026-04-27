"""Overnight experiment watchdog for the encoder-v2/cpuct3 runs.

Keeps the machine busy without blindly oversubscribing the CPU:
- monitors cpuct3 and cpuct3-outcome generation/eval progress;
- kills cpuct3-outcome if gen >= 6 confirms the policy-eval collapse;
- kills cpuct3 if the configured gen-20 policy-eval kill gate still fails;
- starts a q_target_mix=0.25 fallback fork if fewer than two experiments run.

This is intentionally conservative. It only makes decisions at completed
generation boundaries, based on persisted history/eval rows.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path


MAIN_SOLVER = Path(r"C:\coding-projects\STS2\sts2-solver")
MAIN_PY = MAIN_SOLVER / ".venv" / "Scripts" / "python.exe"
LOG_DIR = MAIN_SOLVER / "experiments" / "_watchdog"
LOG_PATH = LOG_DIR / "overnight_watchdog.log"

EXPERIMENTS = {
    "encoder-v2-cpuct3": {
        "solver": Path(r"C:\coding-projects\sts2-encoder-v2-cpuct3\sts2-solver"),
        "kill_gen": 20,
        "kill_policy_passed": 95,
    },
    "encoder-v2-cpuct3-outcome": {
        "solver": Path(r"C:\coding-projects\sts2-encoder-v2-cpuct3-outcome\sts2-solver"),
        "kill_gen": 6,
        "kill_policy_passed": 55,
        "kill_rescue_floor": -0.02,
    },
}

FALLBACK_NAME = "encoder-v2-cpuct3-qmix025"
FALLBACK_SOURCE = "encoder-v2-cpuct3"
FALLBACK_SOLVER = Path(rf"C:\coding-projects\sts2-{FALLBACK_NAME}\sts2-solver")

TELEMETRY_FILES = [
    Path("src/sts2_solver/betaone/eval.py"),
    Path("src/sts2_solver/betaone/selfplay_train.py"),
    Path("sts2-engine/src/mcts.rs"),
    Path("sts2-engine/src/betaone/selfplay.rs"),
]


def log(msg: str) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    line = f"{datetime.now().isoformat(timespec='seconds')} {msg}"
    print(line, flush=True)
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def run(cmd: list[str], *, cwd: Path, timeout: int | None = None) -> subprocess.CompletedProcess:
    log(f"RUN cwd={cwd} cmd={' '.join(cmd)}")
    return subprocess.run(
        cmd,
        cwd=str(cwd),
        text=True,
        capture_output=True,
        timeout=timeout,
    )


def read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    try:
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    except OSError:
        pass
    return rows


def latest_for_gen(path: Path, gen: int) -> dict | None:
    match = None
    for row in read_jsonl(path):
        if row.get("gen") == gen or row.get("generation") == gen:
            match = row
    return match


def exp_dir(solver: Path, name: str) -> Path:
    return solver / "experiments" / name


def completed_gen(solver: Path, name: str) -> int:
    rows = read_jsonl(exp_dir(solver, name) / "betaone_history.jsonl")
    if rows:
        return int(rows[-1].get("gen", 0) or 0)
    progress = read_json(exp_dir(solver, name) / "betaone_progress.json") or {}
    return int(progress.get("gen", 0) or 0) - 1


def running_experiment_names() -> set[str]:
    ps = (
        "Get-CimInstance Win32_Process | "
        "Where-Object { $_.CommandLine -match 'sts2_solver.betaone.experiment_cli train' } | "
        "ForEach-Object { $_.CommandLine }"
    )
    out = subprocess.run(
        ["powershell.exe", "-NoProfile", "-Command", ps],
        text=True,
        capture_output=True,
        timeout=20,
    )
    names: set[str] = set()
    for line in out.stdout.splitlines():
        for name in list(EXPERIMENTS) + [FALLBACK_NAME]:
            if f"train {name}" in line:
                names.add(name)
    return names


def pids_for(name: str) -> list[int]:
    ps = (
        "Get-CimInstance Win32_Process | "
        f"Where-Object {{ $_.CommandLine -match 'sts2_solver.betaone.experiment_cli train {name}' }} | "
        "Select-Object -ExpandProperty ProcessId"
    )
    out = subprocess.run(
        ["powershell.exe", "-NoProfile", "-Command", ps],
        text=True,
        capture_output=True,
        timeout=20,
    )
    pids: list[int] = []
    for line in out.stdout.splitlines():
        try:
            pids.append(int(line.strip()))
        except ValueError:
            pass
    return pids


def hardware_snapshot() -> str:
    cpu = "unknown"
    try:
        out = subprocess.run(
            [
                "powershell.exe",
                "-NoProfile",
                "-Command",
                "(Get-Counter '\\Processor(_Total)\\% Processor Time').CounterSamples[0].CookedValue",
            ],
            text=True,
            capture_output=True,
            timeout=10,
        )
        cpu = f"{float(out.stdout.strip()):.1f}%"
    except Exception:
        pass

    gpu = "nvidia-smi unavailable"
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            capture_output=True,
            timeout=10,
        )
        if out.returncode == 0 and out.stdout.strip():
            util, used, total = [part.strip() for part in out.stdout.splitlines()[0].split(",")]
            gpu = f"{util}% gpu, {used}/{total} MiB"
    except Exception:
        pass
    return f"cpu={cpu} gpu={gpu}"


def kill_experiment(name: str) -> None:
    pids = pids_for(name)
    if not pids:
        log(f"{name}: no process to kill")
        return
    for pid in sorted(set(pids)):
        log(f"{name}: taskkill /T /F /PID {pid}")
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(pid)], text=True, capture_output=True)


def best_eval_gen(solver: Path, name: str) -> int:
    rows = read_jsonl(exp_dir(solver, name) / "benchmarks" / "eval.jsonl")
    if not rows:
        return max(completed_gen(solver, name), 1)
    rows = [r for r in rows if r.get("gen") is not None]
    if not rows:
        return max(completed_gen(solver, name), 1)
    rows.sort(key=lambda r: (int(r.get("passed", 0) or 0), int(r.get("gen", 0) or 0)))
    return int(rows[-1]["gen"])


def finalize(name: str, solver: Path, gen: int, reason: str) -> None:
    py = solver / ".venv" / "Scripts" / "python.exe"
    if not py.exists():
        py = MAIN_PY
    res = run(
        [
            str(py),
            "-m",
            "sts2_solver.betaone.experiment_cli",
            "finalize",
            name,
            "--gen",
            str(gen),
            "--reason",
            reason,
        ],
        cwd=solver,
        timeout=120,
    )
    if res.returncode != 0:
        log(f"{name}: finalize failed rc={res.returncode} stdout={res.stdout!r} stderr={res.stderr!r}")
    else:
        log(f"{name}: finalized at gen {gen}")


def launch_train(name: str, solver: Path) -> None:
    py = solver / ".venv" / "Scripts" / "python.exe"
    exp = exp_dir(solver, name)
    exp.mkdir(parents=True, exist_ok=True)
    log_file = exp / "train.log"
    cmd = (
        f"Set-Location '{solver}'; "
        f"& '{py}' -m sts2_solver.betaone.experiment_cli train {name} *> '{log_file}'"
    )
    log(f"{name}: launching background train")
    subprocess.Popen(
        ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", cmd],
        cwd=str(solver),
        creationflags=getattr(subprocess, "DETACHED_PROCESS", 0)
        | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
    )


def sync_telemetry_files(source_solver: Path, target_solver: Path) -> None:
    for rel in TELEMETRY_FILES:
        src = source_solver / rel
        dst = target_solver / rel
        if src.exists() and dst.exists():
            shutil.copy2(src, dst)
            log(f"synced {rel} -> {target_solver}")


def repair_engine(name: str, solver: Path) -> None:
    py = solver / ".venv" / "Scripts" / "python.exe"
    res = run(
        [str(py), "-m", "sts2_solver.betaone.experiment_cli", "repair", name],
        cwd=solver,
        timeout=900,
    )
    if res.returncode != 0:
        log(f"{name}: repair failed rc={res.returncode} stdout={res.stdout!r} stderr={res.stderr!r}")
    else:
        log(f"{name}: repair complete")


def ensure_fallback_exists() -> bool:
    cfg = exp_dir(FALLBACK_SOLVER, FALLBACK_NAME) / "config.yaml"
    if cfg.exists():
        return True
    log(f"{FALLBACK_NAME}: creating fallback fork")
    res = run(
        [
            str(MAIN_PY),
            "-m",
            "sts2_solver.betaone.experiment_cli",
            "fork",
            FALLBACK_NAME,
            "--from",
            FALLBACK_SOURCE,
            "--checkpoint",
            "latest",
            "--override",
            "training.mcts.q_target_mix=0.25",
            "--override",
            "training.mcts.eval_every=1",
            "--override",
            "checkpoints.save_every=1",
        ],
        cwd=MAIN_SOLVER,
        timeout=1800,
    )
    if res.returncode != 0:
        log(f"{FALLBACK_NAME}: fork failed rc={res.returncode} stdout={res.stdout!r} stderr={res.stderr!r}")
        return False
    source_solver = EXPERIMENTS[FALLBACK_SOURCE]["solver"]
    sync_telemetry_files(source_solver, FALLBACK_SOLVER)
    repair_engine(FALLBACK_NAME, FALLBACK_SOLVER)
    return True


def maybe_start_fallback() -> None:
    running = running_experiment_names()
    active = running & (set(EXPERIMENTS) | {FALLBACK_NAME})
    if len(active) >= 2:
        return
    if FALLBACK_NAME not in active and ensure_fallback_exists():
        launch_train(FALLBACK_NAME, FALLBACK_SOLVER)


def should_kill(name: str, solver: Path, policy_floor: int, gen_gate: int, rescue_floor: float | None) -> tuple[bool, str]:
    gen = completed_gen(solver, name)
    if gen < gen_gate:
        return False, f"waiting for gen {gen_gate}; completed={gen}"

    bench = exp_dir(solver, name) / "benchmarks"
    ev = latest_for_gen(bench / "eval.jsonl", gen)
    mcts = latest_for_gen(bench / "mcts_eval.jsonl", gen)
    if not ev:
        return False, f"gen {gen} complete but eval row not present yet"

    passed = int(ev.get("passed", 0) or 0)
    rescue = None
    if mcts:
        rescue = mcts.get("real_rescue_rate", mcts.get("rescue_rate"))
        rescue = float(rescue) if rescue is not None else None

    if passed < policy_floor:
        return True, f"gen {gen} P-Eval {passed}/136 below floor {policy_floor}"
    if rescue_floor is not None and rescue is not None and rescue <= rescue_floor:
        return True, f"gen {gen} real rescue {rescue:.3f} <= floor {rescue_floor:.3f}"
    return False, f"gen {gen} ok: P-Eval {passed}/136, rescue={rescue}"


def monitor_once() -> None:
    running = running_experiment_names()
    log(f"running={sorted(running)} {hardware_snapshot()}")
    for name, spec in EXPERIMENTS.items():
        solver = spec["solver"]
        gen = completed_gen(solver, name)
        progress = read_json(exp_dir(solver, name) / "betaone_progress.json") or {}
        phase = progress.get("phase", "?")
        log(f"{name}: completed_gen={gen} phase={phase} running={name in running}")

        kill, why = should_kill(
            name,
            solver,
            policy_floor=int(spec["kill_policy_passed"]),
            gen_gate=int(spec["kill_gen"]),
            rescue_floor=spec.get("kill_rescue_floor"),
        )
        log(f"{name}: decision={why}")
        if kill:
            kill_experiment(name)
            gen_best = best_eval_gen(solver, name)
            finalize(name, solver, gen_best, f"Overnight watchdog stopped run: {why}.")
            maybe_start_fallback()
            continue

        if name not in running:
            config = exp_dir(solver, name) / "config.yaml"
            if config.exists():
                log(f"{name}: not running; repairing engine and restarting")
                repair_engine(name, solver)
                launch_train(name, solver)

    maybe_start_fallback()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hours", type=float, default=10.0)
    parser.add_argument("--interval-sec", type=int, default=300)
    args = parser.parse_args()

    stop_at = datetime.now() + timedelta(hours=args.hours)
    log(f"watchdog start stop_at={stop_at.isoformat(timespec='seconds')}")
    while datetime.now() < stop_at:
        try:
            monitor_once()
        except Exception as exc:
            log(f"ERROR {type(exc).__name__}: {exc}")
        time.sleep(args.interval_sec)
    log("watchdog complete")


if __name__ == "__main__":
    main()
