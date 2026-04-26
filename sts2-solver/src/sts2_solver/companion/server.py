"""FastAPI app for the BetaOne companion web UI."""

from __future__ import annotations

import argparse
import asyncio
import functools
import json
import os
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


def create_app(static_dir: Path | None = None) -> FastAPI:
    app = FastAPI(title="BetaOne Companion")

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

    def _experiment_dir(name: str) -> Path:
        from ..betaone import experiment as exp_mod

        for n, d in exp_mod._all_experiment_sources():
            if n == name:
                return d
        raise HTTPException(status_code=404, detail=f"experiment '{name}' not found")

    def _job_root(name: str, gen: int) -> Path:
        from ..betaone import distributed as dist

        root = dist.gen_root(_experiment_dir(name), gen)
        if not (root / "plan.json").exists():
            raise HTTPException(
                status_code=404,
                detail=f"no distributed plan for {name} gen {gen}",
            )
        return root

    def _claim_response(request: Request, claimed) -> dict:
        shared = dict(claimed.shared)
        gen = int(shared["gen"])
        shard_id = claimed.shard_id
        experiment = claimed.experiment
        urls = {
            "onnx": str(request.url_for(
                "distributed_onnx",
                experiment=experiment,
                gen=gen,
                shard_id=shard_id,
            )),
            "heartbeat": str(request.url_for(
                "distributed_heartbeat",
                experiment=experiment,
                gen=gen,
                shard_id=shard_id,
            )),
            "result": str(request.url_for(
                "distributed_result",
                experiment=experiment,
                gen=gen,
                shard_id=shard_id,
            )),
            "fail": str(request.url_for(
                "distributed_fail",
                experiment=experiment,
                gen=gen,
                shard_id=shard_id,
            )),
        }
        if shared.get("onnx_data_file"):
            urls["onnx_data"] = str(request.url_for(
                "distributed_onnx_data",
                experiment=experiment,
                gen=gen,
                shard_id=shard_id,
            ))
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
    def distributed_job(request: Request, experiment: str, gen: int, shard_id: str):
        from ..betaone import distributed as dist

        root = _job_root(experiment, gen)
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
    def distributed_onnx(experiment: str, gen: int, shard_id: str):
        from ..betaone import distributed as dist

        root = _job_root(experiment, gen)
        shared = dist.read_json(root / "shared.json")
        path = root / shared.get("onnx_file", "shared/betaone.onnx")
        if not path.exists():
            raise HTTPException(status_code=404, detail="ONNX file not found")
        return FileResponse(path)

    @app.get(
        "/api/distributed/jobs/{experiment}/{gen}/{shard_id}/onnx-data",
        name="distributed_onnx_data",
    )
    def distributed_onnx_data(experiment: str, gen: int, shard_id: str):
        from ..betaone import distributed as dist

        root = _job_root(experiment, gen)
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
    async def distributed_heartbeat(request: Request, experiment: str, gen: int, shard_id: str):
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
                _job_root(experiment, gen),
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
    async def distributed_result(request: Request, experiment: str, gen: int, shard_id: str):
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
                _job_root(experiment, gen),
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
    async def distributed_fail(request: Request, experiment: str, gen: int, shard_id: str):
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
                _job_root(experiment, gen),
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
