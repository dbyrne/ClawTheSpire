"""Centralized path constants for the BetaOne training pipeline.

All path arithmetic lives here — no more parents[4] scattered across modules.
"""

from pathlib import Path

# sts2-solver/ (the project root for the solver package)
SOLVER_ROOT = Path(__file__).resolve().parents[3]

# sts2_solver/ (the Python package root)
SOLVER_PKG = SOLVER_ROOT / "src" / "sts2_solver"

# STS2-Agent game data (monsters, cards, etc.)
GAME_DATA_DIR = SOLVER_ROOT.parent / "STS2-Agent" / "mcp_server" / "data" / "eng"

# Experiment infrastructure
EXPERIMENTS_DIR = SOLVER_ROOT / "experiments"
BENCHMARK_DIR = EXPERIMENTS_DIR / "_benchmark"
TEMPLATES_DIR = EXPERIMENTS_DIR / "_templates"

# Git repo root (containing the sts2-solver subdir + main .git). Used for
# worktree creation: experiments are now isolated in sibling worktrees
# (C:/coding-projects/sts2-<name>/) rather than sharing trunk.
REPO_ROOT = SOLVER_ROOT.parent
