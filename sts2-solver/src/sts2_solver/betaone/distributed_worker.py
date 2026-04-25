"""Laptop worker for distributed BetaOne self-play.

Example:
    python -m sts2_solver.betaone.distributed_worker \
      --coordinator http://100.100.101.1:8765 \
      --experiment encoder-v2-cpuct3-qmix025 \
      --worker-id laptop-a
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import sys
import tempfile
import threading
import time
import traceback
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from . import distributed as dist


_PROC_STAT_PREV: tuple[int, int] | None = None


def _url(base: str, path: str) -> str:
    return base.rstrip("/") + "/" + path.lstrip("/")


def _request_json(url: str, payload: dict | None = None, timeout: float = 30.0) -> dict:
    data = None
    headers = {"Accept": "application/json"}
    method = "GET"
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
        method = "POST"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read()
    return json.loads(raw.decode("utf-8")) if raw else {}


def _read_proc_stat() -> tuple[int, int] | None:
    """Return (idle, total) jiffies for Linux host CPU accounting."""
    try:
        with open("/proc/stat", "r", encoding="utf-8") as f:
            fields = f.readline().split()
    except OSError:
        return None
    if not fields or fields[0] != "cpu":
        return None
    try:
        values = [int(x) for x in fields[1:]]
    except ValueError:
        return None
    if len(values) < 4:
        return None
    idle = values[3] + (values[4] if len(values) > 4 else 0)
    return idle, sum(values)


def _cpu_percent() -> float | None:
    global _PROC_STAT_PREV
    current = _read_proc_stat()
    if current is None:
        return None
    if _PROC_STAT_PREV is None:
        _PROC_STAT_PREV = current
        return None
    prev_idle, prev_total = _PROC_STAT_PREV
    idle, total = current
    _PROC_STAT_PREV = current
    total_delta = total - prev_total
    idle_delta = idle - prev_idle
    if total_delta <= 0:
        return None
    return max(0.0, min(100.0, 100.0 * (1.0 - idle_delta / total_delta)))


def _rss_mb() -> float | None:
    try:
        with open("/proc/self/statm", "r", encoding="utf-8") as f:
            parts = f.readline().split()
        pages = int(parts[1])
        page_size = os.sysconf("SC_PAGE_SIZE")
        return pages * page_size / (1024 * 1024)
    except (OSError, ValueError, IndexError, AttributeError):
        return None


def _env_int(name: str) -> int | None:
    value = os.environ.get(name)
    if not value:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _sleep_with_jitter(seconds: float) -> None:
    time.sleep(max(1.0, seconds) * random.uniform(0.5, 1.5))


def _worker_metrics(*, worker_id: str, active_shard: str | None = None) -> dict[str, Any]:
    cpu_count = os.cpu_count() or None
    load1 = load5 = load15 = None
    try:
        load1, load5, load15 = os.getloadavg()
    except (AttributeError, OSError):
        pass
    cpu_pct = _cpu_percent()
    rss_mb = _rss_mb()
    metrics: dict[str, Any] = {
        "worker_id": worker_id,
        "host": platform.node(),
        "pid": os.getpid(),
        "active_shard": active_shard,
        "sampled_at": time.time(),
        "cpu_count": cpu_count,
        "cpu_pct": round(cpu_pct, 1) if cpu_pct is not None else None,
        "load1": round(load1, 2) if load1 is not None else None,
        "load5": round(load5, 2) if load5 is not None else None,
        "load15": round(load15, 2) if load15 is not None else None,
        "load_per_cpu": (
            round(load1 / cpu_count, 3)
            if load1 is not None and cpu_count
            else None
        ),
        "rss_mb": round(rss_mb, 1) if rss_mb is not None else None,
        "rayon_threads": _env_int("RAYON_NUM_THREADS"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
    }
    for key, env_name in {
        "instance_id": "AWS_INSTANCE_ID",
        "instance_type": "AWS_INSTANCE_TYPE",
        "worker_group": "STS2_WORKER_GROUP",
        "git_sha": "STS2_GIT_SHA",
    }.items():
        value = os.environ.get(env_name)
        if value:
            metrics[key] = value
    return metrics


def _heartbeat_once(url: str, *, worker_id: str, lease_s: float, active_shard: str | None) -> dict:
    return _request_json(
        url,
        {
            "worker_id": worker_id,
            "lease_s": lease_s,
            "metrics": _worker_metrics(worker_id=worker_id, active_shard=active_shard),
        },
        timeout=20.0,
    )


def _download(url: str, path: Path, timeout: float = 120.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    req = urllib.request.Request(url, headers={"Accept": "application/octet-stream"})
    with urllib.request.urlopen(req, timeout=timeout) as resp, open(tmp, "wb") as f:
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
    os.replace(tmp, path)


def _post_bytes(url: str, data: bytes, *, worker_id: str, timeout: float = 300.0) -> dict:
    sep = "&" if "?" in url else "?"
    url = f"{url}{sep}{urllib.parse.urlencode({'worker_id': worker_id})}"
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Content-Type": "application/octet-stream",
            "Accept": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read()
    return json.loads(raw.decode("utf-8")) if raw else {}


def _heartbeat_loop(
    url: str,
    *,
    worker_id: str,
    lease_s: float,
    interval_s: float,
    active_shard: str,
    stop: threading.Event,
) -> None:
    while not stop.wait(max(5.0, interval_s)):
        try:
            _heartbeat_once(
                url,
                worker_id=worker_id,
                lease_s=lease_s,
                active_shard=active_shard,
            )
        except Exception as exc:
            print(f"[heartbeat] failed: {exc}", flush=True)


def _run_claim(claim: dict, *, worker_id: str, cache_dir: Path, lease_s: float) -> None:
    job = claim["job"]
    shared = claim["shared"]
    urls = claim["urls"]
    shard_id = job["shard_id"]
    experiment = job["experiment"]
    gen = job["gen"]
    onnx_hash = shared.get("onnx_sha256") or f"{experiment}-g{gen}"
    onnx_cache = cache_dir / "onnx" / str(onnx_hash)
    onnx_path = onnx_cache / "betaone.onnx"
    if not onnx_path.exists():
        print(f"[{experiment} g{gen} {shard_id}] downloading ONNX", flush=True)
        _download(urls["onnx"], onnx_path)
    if shared.get("onnx_data_file"):
        onnx_data_path = Path(f"{onnx_path}.data")
        if not onnx_data_path.exists():
            _download(urls["onnx_data"], onnx_data_path)

    stop = threading.Event()
    hb = threading.Thread(
        target=_heartbeat_loop,
        kwargs={
            "url": urls["heartbeat"],
            "worker_id": worker_id,
            "lease_s": lease_s,
            "interval_s": min(30.0, max(10.0, lease_s / 4.0)),
            "active_shard": shard_id,
            "stop": stop,
        },
        daemon=True,
    )
    try:
        _heartbeat_once(urls["heartbeat"], worker_id=worker_id, lease_s=lease_s, active_shard=shard_id)
    except Exception as exc:
        print(f"[heartbeat] initial failed: {exc}", flush=True)
    hb.start()
    started = time.perf_counter()
    try:
        print(
            f"[{experiment} g{gen} {shard_id}] running "
            f"{job.get('target_combats', '?')} combats",
            flush=True,
        )
        rollout = dist.run_selfplay_job(job, shared, onnx_path=onnx_path)
        data = dist.dumps_rollout(rollout)
        try:
            _heartbeat_once(urls["heartbeat"], worker_id=worker_id, lease_s=lease_s, active_shard=shard_id)
        except Exception:
            pass
        _post_bytes(urls["result"], data, worker_id=worker_id)
        elapsed = time.perf_counter() - started
        print(
            f"[{experiment} g{gen} {shard_id}] uploaded "
            f"{rollout.get('total_steps', 0)} steps in {elapsed:.1f}s",
            flush=True,
        )
    except Exception as exc:
        err = "".join(traceback.format_exception_only(type(exc), exc)).strip()
        try:
            _request_json(
                urls["fail"],
                {
                    "worker_id": worker_id,
                    "error": err,
                    "metrics": _worker_metrics(worker_id=worker_id, active_shard=shard_id),
                },
                timeout=30.0,
            )
        except Exception:
            pass
        raise
    finally:
        stop.set()
        hb.join(timeout=2.0)


def run_worker(
    *,
    coordinator: str,
    worker_id: str,
    experiment: str | None,
    cache_dir: Path,
    lease_s: float,
    idle_sleep_s: float,
    once: bool,
) -> int:
    cache_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"STS2 distributed worker: {worker_id} -> {coordinator}"
        + (f" ({experiment})" if experiment else ""),
        flush=True,
    )
    while True:
        try:
            claim = _request_json(
                _url(coordinator, "/api/distributed/claim"),
                {
                    "worker_id": worker_id,
                    "experiment": experiment,
                    "lease_s": lease_s,
                },
                timeout=30.0,
            )
            if not claim.get("job"):
                if once:
                    print("No shard available.", flush=True)
                    return 0
                _sleep_with_jitter(idle_sleep_s)
                continue
            _run_claim(claim, worker_id=worker_id, cache_dir=cache_dir, lease_s=lease_s)
            if once:
                return 0
        except KeyboardInterrupt:
            print("Stopping worker.", flush=True)
            return 130
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            print(f"[http {exc.code}] {body}", flush=True)
            if once:
                return 1
            _sleep_with_jitter(idle_sleep_s)
        except Exception:
            traceback.print_exc()
            if once:
                return 1
            _sleep_with_jitter(idle_sleep_s)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a laptop self-play worker")
    parser.add_argument("--coordinator", required=True, help="Companion API URL, e.g. http://100.100.101.1:8765")
    parser.add_argument("--experiment", default=None, help="Only claim shards for this experiment")
    parser.add_argument("--worker-id", default=None, help="Stable name shown in the companion app")
    parser.add_argument("--cache-dir", default=None, help="Worker cache dir for downloaded ONNX files")
    parser.add_argument("--lease-s", type=float, default=dist.DEFAULT_LEASE_S, help="Claim lease seconds")
    parser.add_argument("--idle-sleep-s", type=float, default=5.0, help="Sleep between empty polls")
    parser.add_argument("--once", action="store_true", help="Claim at most one shard, then exit")
    args = parser.parse_args()

    worker_id = args.worker_id or platform.node() or "sts2-worker"
    cache_dir = Path(args.cache_dir) if args.cache_dir else Path(tempfile.gettempdir()) / "sts2-distributed-worker"
    raise SystemExit(run_worker(
        coordinator=args.coordinator,
        worker_id=worker_id,
        experiment=args.experiment,
        cache_dir=cache_dir,
        lease_s=args.lease_s,
        idle_sleep_s=args.idle_sleep_s,
        once=args.once,
    ))


if __name__ == "__main__":
    main()
