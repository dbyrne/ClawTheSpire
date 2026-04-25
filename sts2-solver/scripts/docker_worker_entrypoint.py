"""Container entrypoint for STS2 distributed self-play workers."""

from __future__ import annotations

import os
import platform
import sys


def _env(name: str, default: str | None = None) -> str:
    value = os.environ.get(name)
    if value is None or value == "":
        if default is None:
            raise SystemExit(f"{name} is required")
        return default
    return value


def main() -> None:
    coordinator = _env("COORDINATOR_URL")
    worker_id = _env("WORKER_ID", platform.node() or "sts2-worker")
    args = [
        sys.executable,
        "-m",
        "sts2_solver.betaone.distributed_worker",
        "--coordinator",
        coordinator,
        "--worker-id",
        worker_id,
        "--cache-dir",
        _env("CACHE_DIR", "/cache"),
        "--lease-s",
        _env("LEASE_S", "240"),
        "--idle-sleep-s",
        _env("IDLE_SLEEP_S", "5"),
    ]
    experiment = os.environ.get("EXPERIMENT")
    if experiment:
        args.extend(["--experiment", experiment])
    if os.environ.get("ONCE", "").lower() in {"1", "true", "yes"}:
        args.append("--once")
    os.execv(sys.executable, args)


if __name__ == "__main__":
    main()
