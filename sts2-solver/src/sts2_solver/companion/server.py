"""FastAPI app for the BetaOne companion web UI."""

from __future__ import annotations

import argparse
import asyncio
import functools
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

from . import data


def _executor_size(name: str, default: int) -> int:
    try:
        return max(4, int(os.environ.get(name, default)))
    except ValueError:
        return default


_DIST_EXECUTOR = ThreadPoolExecutor(
    max_workers=_executor_size("STS2_COMPANION_DIST_THREADS", 32),
    thread_name_prefix="sts2-dist-api",
)
_DASHBOARD_EXECUTOR = ThreadPoolExecutor(
    max_workers=_executor_size("STS2_COMPANION_DASHBOARD_THREADS", 4),
    thread_name_prefix="sts2-dashboard-api",
)


async def _run_dist_io(func, /, *args, **kwargs):
    loop = asyncio.get_running_loop()
    call = functools.partial(func, *args, **kwargs)
    return await loop.run_in_executor(_DIST_EXECUTOR, call)


async def _run_dashboard_io(func, /, *args, **kwargs):
    loop = asyncio.get_running_loop()
    call = functools.partial(func, *args, **kwargs)
    return await loop.run_in_executor(_DASHBOARD_EXECUTOR, call)


_COST_POLL_INTERVAL_S = float(os.environ.get("STS2_COMPANION_COST_POLL_S", "60"))


def _discover_cost_configs() -> list[tuple[str, Path, Path]]:
    """List (experiment_name, exp_dir, capacity_config_path) for all experiments
    that have at least one worker-capacity*.json file. Used by the cost poller
    so the dashboard sees fresh worker_costs.jsonl entries without any human
    needing to run `sts2-experiment workers cost`.

    First matching capacity file per experiment wins; multi-region setups are
    expected to live in a single config file with a `regions` block.
    """
    from ..betaone import experiment as exp_mod

    found: list[tuple[str, Path, Path]] = []
    for name, exp_dir in exp_mod._all_experiment_sources():
        configs = sorted(exp_dir.glob("worker-capacity*.json"))
        if configs:
            found.append((name, exp_dir, configs[0]))
    return found


def _poll_one_experiment_cost(name: str, capacity_path: Path) -> None:
    """Snapshot one experiment's EC2 cost into its worker_costs.jsonl ledger.

    Designed to be called from the cost poller. Swallows errors so one
    experiment's misconfig (missing AWS creds, bad capacity file, etc.)
    can't kill the polling loop for the rest. Logs to stderr.
    """
    import sys
    from ..betaone import worker_orchestration as workers
    try:
        config = workers.load_capacity_config(str(capacity_path))
        summary = workers.estimate_ec2_cost(
            experiment=name,
            config=config,
            include_terminated=True,
            include_recorded=True,
        )
        workers.record_cost_snapshot(name, summary)
    except Exception as exc:
        print(f"[cost-poller] {name}: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)


async def _cost_poller_loop(interval_s: float) -> None:
    """Periodically refresh every-experiment cost snapshots.

    The dashboard reads the tail of worker_costs.jsonl via _cost_summary
    in companion/data.py — without this poller, the ledger only updates
    when someone manually runs `sts2-experiment workers cost`, leaving
    the dashboard's "cost" panel blank for hours-long runs.
    """
    loop = asyncio.get_running_loop()
    while True:
        try:
            configs = await loop.run_in_executor(_DASHBOARD_EXECUTOR, _discover_cost_configs)
            for name, _exp_dir, capacity_path in configs:
                # Run AWS query in a thread; it's blocking subprocess work.
                await loop.run_in_executor(
                    _DASHBOARD_EXECUTOR,
                    _poll_one_experiment_cost,
                    name,
                    capacity_path,
                )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            import sys
            print(f"[cost-poller] loop error: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        await asyncio.sleep(interval_s)


def create_app(static_dir: Path | None = None) -> FastAPI:
    app = FastAPI(title="BetaOne Companion")

    _cost_poller_task: list[asyncio.Task] = []  # holder; populated on startup

    @app.on_event("startup")
    async def _start_cost_poller():
        if _COST_POLL_INTERVAL_S <= 0:
            return  # disabled via env var
        task = asyncio.create_task(
            _cost_poller_loop(_COST_POLL_INTERVAL_S),
            name="cost-poller",
        )
        _cost_poller_task.append(task)

    @app.on_event("shutdown")
    async def _stop_cost_poller():
        for task in _cost_poller_task:
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass

    # Dev: allow Next.js dev server to call the API cross-origin.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    @app.get("/api/health")
    async def health():
        return {"ok": True}

    @app.get("/api/experiments")
    async def experiments_list():
        return JSONResponse(await _run_dashboard_io(data.list_experiments))

    @app.get("/api/distill")
    async def distill_list():
        return JSONResponse(await _run_dashboard_io(data.list_distill))

    @app.get("/api/experiments/{name}")
    async def experiment_detail(name: str):
        exp = await _run_dashboard_io(data.get_experiment, name)
        if not exp:
            raise HTTPException(status_code=404, detail=f"experiment '{name}' not found")
        return JSONResponse(exp)

    @app.get("/api/benchmarks")
    async def benchmarks():
        return JSONResponse(await _run_dashboard_io(data.all_benchmarks))

    @app.get("/api/leaderboard")
    async def eval_leaderboard():
        return JSONResponse(await _run_dashboard_io(data.leaderboard))

    # ------------------------------------------------------------------
    # Card descriptions (for the labeling UI)
    # ------------------------------------------------------------------
    # Structured stats fields (damage, block, powers_applied) miss
    # CONDITIONAL effects like Blade of Ink ("this turn, on attack,
    # gain Strength"). The cards.json description_raw field is the real
    # in-game tooltip. Serve a slim {id: {description_raw, ...}} map so
    # the labeling UI can show real text alongside structured stats.

    _CARD_DATA_CACHE: dict[str, dict] | None = None

    def _load_card_data() -> dict[str, dict]:
        nonlocal _CARD_DATA_CACHE
        if _CARD_DATA_CACHE is not None:
            return _CARD_DATA_CACHE
        from ..betaone.paths import REPO_ROOT
        path = REPO_ROOT / "STS2-Agent" / "mcp_server" / "data" / "eng" / "cards.json"
        if not path.exists():
            _CARD_DATA_CACHE = {}
            return _CARD_DATA_CACHE
        with open(path, "r", encoding="utf-8") as f:
            cards = json.load(f)
        out: dict[str, dict] = {}
        for c in cards:
            cid = c.get("id")
            if not cid:
                continue
            out[cid] = {
                "name": c.get("name", cid),
                "description": c.get("description", ""),
                "description_raw": c.get("description_raw", ""),
                "cost": c.get("cost"),
                "type": c.get("type"),
                "rarity": c.get("rarity"),
                "target": c.get("target"),
            }
        _CARD_DATA_CACHE = out
        return out

    @app.get("/api/card-data")
    async def card_data_lookup():
        return JSONResponse(await _run_dashboard_io(_load_card_data))

    # ------------------------------------------------------------------
    # Human-in-the-loop labeling
    # ------------------------------------------------------------------
    # Pool: experiments/_labels/pool/<name>.jsonl  — header + decision records
    # Labels: experiments/_labels/labels.jsonl    — append-only label log
    #
    # User reviews policy decisions where the network's argmax pick can be
    # flagged "bad". Each "bad" label becomes both a training negative
    # example (penalize the policy's probability mass on that action) and
    # a P-Eval scenario (bad_actions populated). "skip" means reviewed-not-bad
    # — recorded so we don't show the same decision again.

    def _labels_root() -> Path:
        from ..betaone.paths import EXPERIMENTS_DIR
        return EXPERIMENTS_DIR / "_labels"

    def _labels_log_path() -> Path:
        return _labels_root() / "labels.jsonl"

    def _list_pool_files() -> list[Path]:
        pool_dir = _labels_root() / "pool"
        if not pool_dir.exists():
            return []
        return sorted(pool_dir.glob("*.jsonl"))

    def _read_labeled_ids() -> set[str]:
        path = _labels_log_path()
        if not path.exists():
            return set()
        seen: set[str] = set()
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                did = rec.get("decision_id")
                if did:
                    seen.add(str(did))
        return seen

    @app.get("/api/labels/pools")
    async def list_label_pools():
        pools = _list_pool_files()
        out = []
        for p in pools:
            try:
                # Read header (first line) for metadata
                with open(p, "r", encoding="utf-8") as f:
                    first = f.readline()
                meta = json.loads(first).get("_meta", {}) if first.strip() else {}
                # Count records (lines minus header)
                with open(p, "r", encoding="utf-8") as f:
                    total = sum(1 for _ in f) - 1
            except Exception:
                meta, total = {}, 0
            out.append({
                "name": p.stem,
                "path": str(p),
                "total": max(0, total),
                "meta": meta,
            })
        labeled = _read_labeled_ids()
        return JSONResponse({"pools": out, "total_labeled": len(labeled)})

    @app.get("/api/labels/pool/{name}/next")
    async def next_unlabeled_decision(name: str, offset: int = 0):
        """Return the next unlabeled decision (or null if pool is exhausted)."""
        pool_dir = _labels_root() / "pool"
        path = pool_dir / f"{name}.jsonl"
        if not path.exists():
            raise HTTPException(status_code=404, detail=f"pool {name!r} not found")
        labeled = _read_labeled_ids()
        with open(path, "r", encoding="utf-8") as f:
            seen = 0
            skipped_for_offset = 0
            total_unlabeled = 0
            next_decision = None
            for i, line in enumerate(f):
                if i == 0:
                    continue  # skip header
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                did = rec.get("id")
                if not did or did in labeled:
                    continue
                total_unlabeled += 1
                if skipped_for_offset < offset:
                    skipped_for_offset += 1
                    continue
                if next_decision is None:
                    next_decision = rec
                seen += 1
        return JSONResponse({
            "decision": next_decision,
            "remaining": total_unlabeled,
        })

    @app.post("/api/labels/submit")
    async def submit_label(request: Request):
        try:
            payload = await request.json()
        except Exception:
            raise HTTPException(status_code=400, detail="invalid JSON body")
        decision_id = payload.get("decision_id")
        label = payload.get("label")
        if not decision_id or label not in {"bad", "skip"}:
            raise HTTPException(
                status_code=400,
                detail="body must be {decision_id: str, label: 'bad'|'skip'}",
            )
        bad_action_slot = payload.get("bad_action_slot")
        log_path = _labels_log_path()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "decision_id": str(decision_id),
            "label": label,
            "bad_action_slot": bad_action_slot,
            "labeled_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        return JSONResponse({"ok": True, "record": record})

    def _experiment_dir(name: str) -> Path:
        from ..betaone import experiment as exp_mod

        for n, d in exp_mod._all_experiment_sources():
            if n == name:
                return d
        raise HTTPException(status_code=404, detail=f"experiment '{name}' not found")

    def _job_root(name: str, gen: int, kind: str = "selfplay") -> Path:
        """Resolve a shard root by experiment + gen + kind.

        kind="selfplay" -> shards/gen<N>/  (existing layout)
        kind="reanalyse" -> shards/gen<N>-reanalyse/  (added 2026-04-26)
        """
        from ..betaone import distributed as dist

        if str(kind).lower() == "reanalyse":
            root = dist.reanalyse_root(_experiment_dir(name), gen)
        else:
            root = dist.gen_root(_experiment_dir(name), gen)
        if not (root / "plan.json").exists():
            raise HTTPException(
                status_code=404,
                detail=f"no distributed plan for {name} gen {gen} (kind={kind})",
            )
        return root

    def _claim_response(request: Request, claimed) -> dict:
        shared = dict(claimed.shared)
        gen = int(shared["gen"])
        shard_id = claimed.shard_id
        experiment = claimed.experiment
        # Reanalyse shards share the same URL routes as selfplay; the
        # kind=reanalyse query param tells the handler which root to look
        # under. Selfplay URLs omit the param (default kind=selfplay).
        kind = str(shared.get("kind") or claimed.job.get("kind") or "selfplay").lower()
        kind_qs = "?kind=reanalyse" if kind == "reanalyse" else ""

        def _url(route_name: str) -> str:
            return str(request.url_for(
                route_name,
                experiment=experiment,
                gen=gen,
                shard_id=shard_id,
            )) + kind_qs

        urls = {
            "onnx": _url("distributed_onnx"),
            "heartbeat": _url("distributed_heartbeat"),
            "result": _url("distributed_result"),
            "fail": _url("distributed_fail"),
        }
        if shared.get("onnx_data_file"):
            urls["onnx_data"] = _url("distributed_onnx_data")
        return {
            "job": claimed.job,
            "shared": shared,
            "status": claimed.status,
            "urls": urls,
        }

    @app.post("/api/distributed/claim")
    async def distributed_claim(request: Request):
        from ..betaone import distributed as dist
        from ..betaone import experiment as exp_mod

        try:
            payload = await request.json()
        except Exception:
            payload = {}
        worker_id = str(payload.get("worker_id") or "unknown-worker")
        experiment_name = payload.get("experiment")
        worker_fingerprint = payload.get("fingerprint")
        if not experiment_name:
            raise HTTPException(status_code=400, detail="distributed workers must set --experiment")
        if not isinstance(worker_fingerprint, dict):
            raise HTTPException(status_code=400, detail="distributed workers must send a code fingerprint")
        lease_s = dist.normalize_lease_s(payload.get("lease_s"))
        claimed = await _run_dist_io(
            dist.claim_next_job,
            exp_mod._all_experiment_sources(),
            worker_id=worker_id,
            experiment=str(experiment_name),
            worker_fingerprint=worker_fingerprint,
            lease_s=lease_s,
        )
        if not claimed:
            def conflict_detail() -> dict | None:
                exp_dir = _experiment_dir(str(experiment_name))
                now = dist.utc_ts()
                for root in dist.iter_experiment_roots(exp_dir):
                    if not dist._root_has_claimable_status(root, now):
                        continue
                    shared = dist.read_json(root / "shared.json")
                    mismatches = dist.fingerprint_mismatches(
                        shared.get("required_worker_fingerprint"),
                        worker_fingerprint,
                    )
                    if mismatches:
                        return {
                            "message": "worker code fingerprint does not match experiment",
                            "mismatches": mismatches,
                            "expected": shared.get("required_worker_fingerprint"),
                            "actual": worker_fingerprint,
                        }
                return None

            conflict = await _run_dist_io(conflict_detail)
            if conflict:
                raise HTTPException(status_code=409, detail=conflict)
            return {"job": None}
        return _claim_response(request, claimed)

    @app.get("/api/distributed/jobs/{experiment}/{gen}/{shard_id}")
    def distributed_job(request: Request, experiment: str, gen: int, shard_id: str, kind: str = "selfplay"):
        from ..betaone import distributed as dist

        root = _job_root(experiment, gen, kind)
        job = dist.read_json(dist.job_path(root, shard_id))
        if not job:
            raise HTTPException(status_code=404, detail=f"job '{shard_id}' not found")
        claimed = dist.ClaimedJob(
            experiment=experiment,
            root=root,
            shard_id=shard_id,
            job=job,
            shared=dist.read_json(root / "shared.json"),
            status=dist.read_json(dist.status_path(root, shard_id)),
        )
        return _claim_response(request, claimed)

    @app.get("/api/distributed/jobs/{experiment}/{gen}/{shard_id}/onnx", name="distributed_onnx")
    def distributed_onnx(experiment: str, gen: int, shard_id: str, kind: str = "selfplay"):
        from ..betaone import distributed as dist

        root = _job_root(experiment, gen, kind)
        shared = dist.read_json(root / "shared.json")
        path = root / shared.get("onnx_file", "shared/betaone.onnx")
        if not path.exists():
            raise HTTPException(status_code=404, detail="ONNX file not found")
        return FileResponse(path)

    @app.get(
        "/api/distributed/jobs/{experiment}/{gen}/{shard_id}/onnx-data",
        name="distributed_onnx_data",
    )
    def distributed_onnx_data(experiment: str, gen: int, shard_id: str, kind: str = "selfplay"):
        from ..betaone import distributed as dist

        root = _job_root(experiment, gen, kind)
        shared = dist.read_json(root / "shared.json")
        rel = shared.get("onnx_data_file")
        if not rel:
            raise HTTPException(status_code=404, detail="ONNX external data file not used")
        path = root / rel
        if not path.exists():
            raise HTTPException(status_code=404, detail="ONNX external data file not found")
        return FileResponse(path)

    @app.post(
        "/api/distributed/jobs/{experiment}/{gen}/{shard_id}/heartbeat",
        name="distributed_heartbeat",
    )
    async def distributed_heartbeat(request: Request, experiment: str, gen: int, shard_id: str, kind: str = "selfplay"):
        from ..betaone import distributed as dist

        try:
            payload = await request.json()
        except Exception:
            payload = {}
        worker_id = str(payload.get("worker_id") or request.query_params.get("worker_id") or "unknown-worker")
        lease_s = dist.normalize_lease_s(payload.get("lease_s") or request.query_params.get("lease_s"))
        try:
            return await _run_dist_io(
                dist.heartbeat,
                _job_root(experiment, gen, kind),
                shard_id,
                worker_id=worker_id,
                lease_s=lease_s,
                worker_fingerprint=payload.get("fingerprint"),
                worker_metrics=payload.get("metrics") or payload.get("worker_metrics"),
            )
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail=f"job '{shard_id}' not found")

    @app.post(
        "/api/distributed/jobs/{experiment}/{gen}/{shard_id}/result",
        name="distributed_result",
    )
    async def distributed_result(request: Request, experiment: str, gen: int, shard_id: str, kind: str = "selfplay"):
        from ..betaone import distributed as dist

        worker_id = str(request.query_params.get("worker_id") or "unknown-worker")
        fingerprint_header = request.headers.get("x-sts2-fingerprint")
        if not fingerprint_header:
            raise HTTPException(status_code=400, detail="distributed workers must send a code fingerprint")
        if fingerprint_header:
            try:
                worker_fingerprint = json.loads(fingerprint_header)
            except Exception:
                raise HTTPException(status_code=400, detail="invalid worker code fingerprint")
        if not isinstance(worker_fingerprint, dict):
            raise HTTPException(status_code=400, detail="invalid worker code fingerprint")
        body = await request.body()
        if not body:
            raise HTTPException(status_code=400, detail="empty result body")
        try:
            return await _run_dist_io(
                dist.mark_complete,
                _job_root(experiment, gen, kind),
                shard_id,
                worker_id=worker_id,
                result_bytes=body,
                worker_fingerprint=worker_fingerprint,
            )
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail=f"job '{shard_id}' not found")

    @app.post(
        "/api/distributed/jobs/{experiment}/{gen}/{shard_id}/fail",
        name="distributed_fail",
    )
    async def distributed_fail(request: Request, experiment: str, gen: int, shard_id: str, kind: str = "selfplay"):
        from ..betaone import distributed as dist

        try:
            payload = await request.json()
        except Exception:
            payload = {}
        worker_id = str(payload.get("worker_id") or request.query_params.get("worker_id") or "unknown-worker")
        error = str(payload.get("error") or "worker failed")
        try:
            return await _run_dist_io(
                dist.mark_failed,
                _job_root(experiment, gen, kind),
                shard_id,
                worker_id=worker_id,
                error=error,
                worker_metrics=payload.get("metrics") or payload.get("worker_metrics"),
            )
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail=f"job '{shard_id}' not found")

    # Serve the built Next.js static export when available (prod). Falls back
    # to an explanatory JSON response if not yet built — dev mode runs the
    # Next.js dev server separately on :3000.
    if static_dir and static_dir.exists() and (static_dir / "_next").exists():
        # Next.js static export puts the HTML/JS under `out/`.
        app.mount("/_next", StaticFiles(directory=str(static_dir / "_next")), name="next-assets")

        @app.get("/{full_path:path}")
        def spa_fallback(full_path: str):
            # Try an exact file match first (css, images, etc.)
            candidate = static_dir / full_path
            if candidate.is_file():
                return FileResponse(candidate)
            # Then an .html match (Next.js static export generates per-route htmls)
            html_candidate = static_dir / f"{full_path}.html"
            if html_candidate.is_file():
                return FileResponse(html_candidate)
            # Fallback to index.html (SPA-style)
            index = static_dir / "index.html"
            if index.is_file():
                return FileResponse(index)
            raise HTTPException(status_code=404, detail="not found")
    else:
        @app.get("/")
        def dev_hint():
            return {
                "message": (
                    "Companion API is live. Run the Next.js frontend in dev with "
                    "`cd companion-web && npm install && npm run dev`, then open "
                    "http://localhost:3000/."
                ),
                "api": {
                    "health": "/api/health",
                    "experiments": "/api/experiments",
                    "experiment_detail": "/api/experiments/{name}",
                    "benchmarks": "/api/benchmarks",
                    "leaderboard": "/api/leaderboard",
                },
            }

    return app


def main():
    parser = argparse.ArgumentParser(description="BetaOne companion web server")
    parser.add_argument("--host", default="0.0.0.0", help="bind host (default 0.0.0.0 for Tailscale)")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--static",
        default=None,
        help="Path to Next.js static export dir (companion-web/out). If unset, API-only mode.",
    )
    parser.add_argument("--reload", action="store_true", help="dev: hot-reload on code changes")
    args = parser.parse_args()

    static_dir = Path(args.static) if args.static else None
    # Sensible default: look for companion-web/out next to sts2-solver.
    if static_dir is None:
        guess = Path(__file__).resolve().parents[3] / "companion-web" / "out"
        if guess.exists():
            static_dir = guess

    import uvicorn

    if args.reload:
        # uvicorn reload only works with a module-level import string.
        os.environ["BETAONE_COMPANION_STATIC"] = str(static_dir) if static_dir else ""
        uvicorn.run(
            "sts2_solver.companion.server:_reloadable_app",
            host=args.host,
            port=args.port,
            reload=True,
            factory=True,
        )
    else:
        app = create_app(static_dir)
        uvicorn.run(app, host=args.host, port=args.port)


def _reloadable_app() -> FastAPI:
    """Factory used by `--reload` mode."""
    static = os.environ.get("BETAONE_COMPANION_STATIC")
    return create_app(Path(static) if static else None)


if __name__ == "__main__":
    main()
