"""FastAPI app for the BetaOne companion web UI."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
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
        allow_methods=["GET"],
        allow_headers=["*"],
    )

    @app.get("/api/health")
    def health():
        return {"ok": True}

    @app.get("/api/experiments")
    def experiments_list():
        return data.list_experiments()

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
