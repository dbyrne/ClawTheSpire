"""FastAPI app for the BetaOne companion web UI."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from . import data


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
    def health():
        return {"ok": True}

    @app.get("/api/experiments")
    def experiments_list():
        return data.list_experiments()

    @app.get("/api/distill")
    def distill_list():
        return data.list_distill()

    @app.get("/api/experiments/{name}")
    def experiment_detail(name: str):
        exp = data.get_experiment(name)
        if not exp:
            raise HTTPException(status_code=404, detail=f"experiment '{name}' not found")
        return exp

    @app.get("/api/benchmarks")
    def benchmarks():
        return data.all_benchmarks()

    @app.get("/api/leaderboard")
    def eval_leaderboard():
        return data.leaderboard()

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
        lease_s = dist.normalize_lease_s(payload.get("lease_s"))
        claimed = dist.claim_next_job(
            exp_mod._all_experiment_sources(),
            worker_id=worker_id,
            experiment=str(experiment_name) if experiment_name else None,
            lease_s=lease_s,
        )
        if not claimed:
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
            return dist.heartbeat(_job_root(experiment, gen), shard_id, worker_id=worker_id, lease_s=lease_s)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail=f"job '{shard_id}' not found")

    @app.post(
        "/api/distributed/jobs/{experiment}/{gen}/{shard_id}/result",
        name="distributed_result",
    )
    async def distributed_result(request: Request, experiment: str, gen: int, shard_id: str):
        from ..betaone import distributed as dist

        worker_id = str(request.query_params.get("worker_id") or "unknown-worker")
        body = await request.body()
        if not body:
            raise HTTPException(status_code=400, detail="empty result body")
        try:
            return dist.mark_complete(
                _job_root(experiment, gen),
                shard_id,
                worker_id=worker_id,
                result_bytes=body,
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
            return dist.mark_failed(_job_root(experiment, gen), shard_id, worker_id=worker_id, error=error)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail=f"job '{shard_id}' not found")

    # Serve the built Next.js static export when available (prod). Falls back
    # to an explanatory JSON response if not yet built — dev mode runs the
    # Next.js dev server separately on :3000.
    if static_dir and static_dir.exists():
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
