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


def _download(url: str, path: Path, timeout: float = 120.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
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
    stop: threading.Event,
) -> None:
    while not stop.wait(max(5.0, interval_s)):
        try:
            _request_json(url, {"worker_id": worker_id, "lease_s": lease_s}, timeout=20.0)
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
            "interval_s": min(60.0, max(10.0, lease_s / 3.0)),
            "stop": stop,
        },
        daemon=True,
    )
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
            _request_json(urls["fail"], {"worker_id": worker_id, "error": err}, timeout=30.0)
        except Exception:
            pass
        raise
    finally:
        stop.set()


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
                time.sleep(max(1.0, idle_sleep_s))
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
            time.sleep(max(1.0, idle_sleep_s))
        except Exception:
            traceback.print_exc()
            if once:
                return 1
            time.sleep(max(1.0, idle_sleep_s))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a laptop self-play worker")
    parser.add_argument("--coordinator", required=True, help="Companion API URL, e.g. http://100.100.101.1:8765")
    parser.add_argument("--experiment", default=None, help="Only claim shards for this experiment")
    parser.add_argument("--worker-id", default=None, help="Stable name shown in the companion app")
    parser.add_argument("--cache-dir", default=None, help="Worker cache dir for downloaded ONNX files")
    parser.add_argument("--lease-s", type=float, default=900.0, help="Claim lease seconds")
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
