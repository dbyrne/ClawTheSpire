"""Companion web-app backend for monitoring BetaOne experiments.

Read-only FastAPI service that aggregates across main + worktrees using the
same discovery logic as sts2-experiment CLI. Meant to be served behind
Tailscale — no app-level auth.

Entry: `python -m sts2_solver.companion` or `sts2-companion`.
"""

from .server import main

__all__ = ["main"]
