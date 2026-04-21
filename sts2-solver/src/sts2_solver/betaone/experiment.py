"""Experiment management for BetaOne training.

Each experiment is a named directory under experiments/ containing:
  config.yaml                 — frozen training config (the record of intent)
  card_vocab.json             — frozen copy from creation time
  recorded_encounters.jsonl   — recorded encounters (if applicable)
  betaone_latest.pt           — resume checkpoint
  betaone_gen{N}.pt           — milestone checkpoints
  betaone_history.jsonl       — per-generation log
  betaone_progress.json       — live progress snapshot
  benchmarks/results.jsonl    — append-only benchmark log
  onnx/betaone.onnx           — latest ONNX export

The flat layout matches what the training scripts expect as output_dir,
so Experiment.output_dir() can be passed directly to train().
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from .network import ARCH_META, network_stats
from .paths import EXPERIMENTS_DIR, BENCHMARK_DIR, TEMPLATES_DIR, REPO_ROOT


# ---------------------------------------------------------------------------
# Worktree helpers
#
# Experiments run in sibling git worktrees (../sts2-<name>/) so structural
# code changes stay isolated per-experiment rather than leaking into trunk
# as feature flags. Shared venv would force a single installed sts2_engine
# wheel across all worktrees — instead each worktree has its own venv with
# --system-site-packages so torch/etc are inherited but sts2_engine is
# per-worktree. Merging back is a manual `git merge experiment/<name>`
# once the experiment ships.
# ---------------------------------------------------------------------------


def _experiment_branch(name: str) -> str:
    """Convention: all experiment branches live under experiment/* namespace."""
    return f"experiment/{name}"


def _experiment_worktree_path(name: str) -> Path:
    """Convention: worktree is a sibling of the main checkout.
    C:/coding-projects/STS2/  (main, repo root + working tree)
    C:/coding-projects/sts2-<name>/  (experiment worktree)
    The experiment branch's sts2-solver dir is at <worktree>/sts2-solver/.
    """
    return REPO_ROOT.parent / f"sts2-{name}"


def _run_git(args: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess:
    """Run a git command, raising on non-zero exit. Captures output."""
    result = subprocess.run(
        ["git", *args],
        cwd=str(cwd) if cwd else None,
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"git {' '.join(args)} failed (cwd={cwd}):\n"
            f"  stdout: {result.stdout}\n  stderr: {result.stderr}"
        )
    return result


def _create_worktree(name: str, base_branch: str = "main") -> Path:
    """Create a git worktree for experiment <name> on a new branch.

    Returns the path to the worktree's sts2-solver directory (where the CLI
    should cd to operate). base_branch can be 'main' or 'experiment/<parent>'
    when forking — the new branch starts at base_branch's HEAD.
    """
    worktree_root = _experiment_worktree_path(name)
    branch = _experiment_branch(name)

    if worktree_root.exists():
        raise FileExistsError(
            f"Worktree path {worktree_root} already exists. "
            "Pick a different name or `sts2-experiment archive <existing>` first."
        )
    # Let git's own error surface if the branch already exists or base_branch
    # doesn't exist — clearer than recreating the check here.
    _run_git(
        ["worktree", "add", str(worktree_root), "-b", branch, base_branch],
        cwd=REPO_ROOT,
    )
    # Junction STS2-Agent from main so game data loaders in the worktree
    # find it at the expected relative path. Read-only static data, safe
    # to share across all worktrees.
    _create_sts2_agent_junction(worktree_root)
    return worktree_root / "sts2-solver"


def _venv_python(venv_dir: Path) -> Path:
    import sys
    return venv_dir / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")


def _venv_maturin(venv_dir: Path) -> Path:
    import sys
    return venv_dir / ("Scripts/maturin.exe" if sys.platform == "win32" else "bin/maturin")


def _venv_ok(venv_dir: Path) -> bool:
    """True iff venv exists with --system-site-packages=true and a working
    python interpreter. A venv without system-site-packages is not OK: we'd
    be missing torch/numpy which are inherited from the system Python."""
    cfg = venv_dir / "pyvenv.cfg"
    if not cfg.exists() or not _venv_python(venv_dir).exists():
        return False
    try:
        return "include-system-site-packages = true" in cfg.read_text()
    except OSError:
        return False


def _venv_solver_installed(venv_dir: Path) -> bool:
    """True iff the venv can import sts2_solver (this worktree's editable
    install)."""
    result = subprocess.run(
        [str(_venv_python(venv_dir)), "-c", "import sts2_solver"],
        capture_output=True,
    )
    return result.returncode == 0


def _venv_has_engine(venv_dir: Path) -> bool:
    """True iff sts2_engine is importable from the venv."""
    result = subprocess.run(
        [str(_venv_python(venv_dir)), "-c", "import sts2_engine"],
        capture_output=True,
    )
    return result.returncode == 0


def _setup_worktree_venv(worktree_solver: Path, *, rebuild_engine: bool = True) -> None:
    """Create .venv/ in the worktree's sts2-solver dir and install sts2_engine.

    Idempotent: each step is gated on whether it's already done, so re-running
    after a partial/interrupted setup picks up where it left off. Pass
    rebuild_engine=False to skip the maturin develop step when you know the
    Rust source hasn't changed since the last build (rare; the default always
    rebuilds since cargo incremental makes it cheap when nothing's changed).

    Uses --system-site-packages so torch/rich/etc come from system Python;
    only sts2_engine is worktree-specific (the one thing that genuinely
    differs between worktrees). Runs maturin develop --release against the
    worktree's Rust source so the installed wheel matches the worktree's
    code, not main's.
    """
    import sys, shutil
    venv_dir = worktree_solver / ".venv"

    # Step 1: Venv exists with system-site-packages. If it exists without
    # the flag (e.g. `uv sync` created it), recreate — flipping the flag
    # in pyvenv.cfg post-hoc is fragile because the venv's paths may have
    # been baked in wrong.
    if _venv_ok(venv_dir):
        print("  [1/4] venv ok (skip)")
    else:
        print("  [1/4] creating venv (--system-site-packages)...")
        if venv_dir.exists():
            shutil.rmtree(venv_dir)
        subprocess.run(
            [sys.executable, "-m", "venv", "--system-site-packages", str(venv_dir)],
            check=True,
        )

    venv_python = _venv_python(venv_dir)

    # Step 2: sts2-solver editable install. --force-reinstall --no-deps
    # because the system Python already has main's sts2-solver editable, and
    # without --force-reinstall pip resolves `import sts2_solver` to main's
    # path (not the worktree's), silently breaking the whole point of having
    # separate worktrees. Check uses the venv's own .pth file, not just
    # whether the import succeeds (which it always does via system site-
    # packages).
    editable_pth = venv_dir / "Lib" / "site-packages" / "_editable_impl_sts2_solver.pth"
    unix_pth = venv_dir / "lib" / "python3.11" / "site-packages" / "_editable_impl_sts2_solver.pth"
    if editable_pth.exists() or unix_pth.exists():
        print("  [2/4] sts2-solver ok (skip)")
    else:
        print("  [2/4] installing sts2-solver (editable, --force-reinstall)...")
        subprocess.run(
            [str(venv_python), "-m", "pip", "install",
             "--force-reinstall", "--no-deps", "-e", "."],
            cwd=str(worktree_solver), check=True,
        )

    # Step 3: maturin in venv Scripts. --force-reinstall --no-deps because
    # maturin is already importable via --system-site-packages (system Python
    # has it), and without --force-reinstall pip says "already satisfied" and
    # skips — leaving no maturin.exe in the venv's Scripts/, and maturin's
    # Python entry point then fails with "Unable to find maturin script".
    if _venv_maturin(venv_dir).exists():
        print("  [3/4] maturin ok (skip)")
    else:
        print("  [3/4] installing maturin into venv...")
        subprocess.run(
            [str(venv_python), "-m", "pip", "install",
             "--force-reinstall", "--no-deps", "maturin"],
            check=True,
        )

    # Step 4: build + install sts2_engine from this worktree's Rust source.
    # Default is to always run: cargo incremental makes it fast (~5-10s) when
    # nothing changed, and it's the only way to guarantee the installed wheel
    # matches the current source. Callers can opt out via rebuild_engine=False.
    if not rebuild_engine and _venv_has_engine(venv_dir):
        print("  [4/4] sts2_engine ok (skip, rebuild_engine=False)")
    else:
        print("  [4/4] building sts2_engine (maturin develop --release)...")
        subprocess.run(
            [str(venv_python), "-m", "maturin", "develop", "--release"],
            cwd=str(worktree_solver / "sts2-engine"), check=True,
        )


def _is_our_worktree(worktree_root: Path, name: str) -> bool:
    """True iff the path is a git worktree checked out on experiment/<name>.
    Used to decide whether an existing worktree dir is ours to resume setup
    on, vs. a stray dir with the same name that we shouldn't touch."""
    expected = _experiment_branch(name)
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=str(worktree_root), capture_output=True, text=True, check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False
    return result.stdout.strip() == expected


def _create_sts2_agent_junction(worktree_root: Path) -> None:
    """Create a directory junction (Windows) / symlink (Unix) from the
    worktree to main's STS2-Agent so game data loaders find it at the
    expected relative path (`<worktree_root>/STS2-Agent/...`).

    Game data is static read-only (cards.json, enemy_profiles.json, etc.) —
    sharing across worktrees is both safe and desirable (one copy to keep
    in sync). Per-worktree copies would also mean ~50MB duplicated per
    worktree and a real risk of stale game data on some worktrees.

    No admin privileges needed on Windows (mklink /J makes a junction,
    not a symlink). Requires NTFS. On Unix creates a symlink.
    """
    import sys
    target = REPO_ROOT / "STS2-Agent"
    link = worktree_root / "STS2-Agent"
    if link.exists() or link.is_symlink():
        return
    if not target.exists():
        print(f"Warning: STS2-Agent not found at {target}. Game data loaders "
              "in the worktree will fail until this is resolved.")
        return
    if sys.platform == "win32":
        subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(link), str(target)],
            check=True, capture_output=True,
        )
    else:
        link.symlink_to(target)


def _activation_hint(worktree_solver: Path) -> str:
    """Print-ready instructions for the user to activate the worktree venv."""
    import sys
    if sys.platform == "win32":
        activate = worktree_solver / ".venv" / "Scripts" / "activate"
    else:
        activate = worktree_solver / ".venv" / "bin" / "activate"
    return (
        f"To work in this experiment:\n"
        f"  cd {worktree_solver}\n"
        f"  source {activate.relative_to(worktree_solver)}\n"
        f"  sts2-experiment train <name>"
    )


def _all_experiment_sources() -> list[tuple[str, Path]]:
    """Return [(experiment_name, experiment_dir)] across main + worktrees.

    Main contributes ALL its experiments/<N>/ subdirs (legacy experiments
    + finalized records synced back via `ship`).

    Each worktree contributes EXACTLY ONE experiment — the one whose name
    matches the worktree's name convention (`sts2-<name>` at repo root).
    Other experiment dirs inside a worktree are just inherited copies from
    when the worktree was branched off main — they'd be stale and confuse
    aggregation (e.g., before `ship` syncs finalize to main, the worktree
    still has the pre-finalize copy). Filtering by name makes each
    experiment have exactly one authoritative source.
    """
    results: list[tuple[str, Path]] = []

    # Main's experiments/
    if EXPERIMENTS_DIR.exists():
        for d in EXPERIMENTS_DIR.iterdir():
            if d.is_dir() and not d.name.startswith("_") and (d / "config.yaml").exists():
                results.append((d.name, d))

    # Each worktree contributes only its own named experiment.
    try:
        out = subprocess.run(
            ["git", "worktree", "list", "--porcelain"],
            cwd=str(REPO_ROOT),
            capture_output=True, text=True, check=True,
        ).stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        return results

    main_root = REPO_ROOT.resolve()
    for line in out.splitlines():
        if not line.startswith("worktree "):
            continue
        path = Path(line[len("worktree "):].strip()).resolve()
        if path == main_root:
            continue  # main already handled above
        # Worktree convention: ../sts2-<name>/
        if not path.name.startswith("sts2-"):
            continue
        exp_name = path.name[len("sts2-"):]
        exp_dir = path / "sts2-solver" / "experiments" / exp_name
        if exp_dir.exists() and (exp_dir / "config.yaml").exists():
            # Worktree copy wins over main's copy (if any) — the worktree is
            # authoritative for its own name (live training state).
            results = [(n, d) for (n, d) in results if n != exp_name]
            results.append((exp_name, exp_dir))

    return results


def _find_experiment_dir(name: str) -> Path:
    """Return the experiments/<name>/ dir via the source aggregation.

    If not found anywhere, returns main's EXPERIMENTS_DIR/<name> as the
    default (that's what `create` uses — soon-to-exist path).
    """
    for n, d in _all_experiment_sources():
        if n == name:
            return d
    return EXPERIMENTS_DIR / name


def _archive_keep_set(
    source_dir: Path, prefix: str, concluded_gen: int | None
) -> list[Path]:
    """Allow-list of files to retain on archive (existing paths only).

    Aggressive policy — keep the experiment's RECORD plus just the pinned
    checkpoint. Specifically:
      - config.yaml, PLAN.md, card_vocab.json
      - {prefix}_history.jsonl, {prefix}_progress.json
      - benchmarks/eval.jsonl, value_eval.jsonl, results.jsonl
      - {prefix}_gen{concluded_gen}.pt, OR {prefix}_latest.pt if its recorded
        gen matches concluded_gen (mirrors finalize()'s checkpoint-resolution)

    Drops everything else (non-concluded gen-N .pts, .venv, onnx/, logs).
    """
    keep: list[Path] = []
    for name in [
        "config.yaml", "PLAN.md", "card_vocab.json",
        f"{prefix}_history.jsonl", f"{prefix}_progress.json",
    ]:
        p = source_dir / name
        if p.exists():
            keep.append(p)
    bench = source_dir / "benchmarks"
    if bench.exists():
        for name in ["eval.jsonl", "value_eval.jsonl", "results.jsonl"]:
            p = bench / name
            if p.exists():
                keep.append(p)
    if concluded_gen is not None:
        gen_pt = source_dir / f"{prefix}_gen{concluded_gen}.pt"
        if gen_pt.exists():
            keep.append(gen_pt)
        else:
            latest_pt = source_dir / f"{prefix}_latest.pt"
            if latest_pt.exists():
                try:
                    import torch
                    ck = torch.load(
                        str(latest_pt), map_location="cpu", weights_only=False
                    )
                    if ck.get("gen") == concluded_gen:
                        keep.append(latest_pt)
                except Exception:
                    pass
    return keep


# MCTS knobs that behave as training hyperparameters. Kept here so TUI and any
# other consumer can show the *effective* value for an experiment even when the
# config omits a key (i.e. the code default is in effect). Update in this one
# place when a default changes; to_train_kwargs below inlines the same values.
MCTS_DEFAULTS = {
    "num_sims": 400,
    "temperature": 1.0,
    "train_epochs": 4,
    "value_coef": 1.0,
    "replay_capacity": 200_000,
    "turn_boundary_eval": False,
    "c_puct": 2.5,
    "pomcp": False,
    "mcts_bootstrap": False,
    "noise_frac": 0.25,
    "pw_k": 1.0,
    "q_target_mix": 0.0,
    "q_target_temp": 0.5,
    "eval_every": 0,
}


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------

@dataclass
class ExperimentConfig:
    # Metadata
    name: str
    method: str  # "mcts_selfplay" or "ppo" (BetaOne) or "decknet_selfplay" (DeckNet)
    description: str = ""
    created: str = ""
    parent: str | None = None
    parent_checkpoint: str | None = None
    # Which network trains this experiment. "betaone" is the combat network
    # (legacy default for back-compat). "decknet" is the deck-building
    # network. Used by experiment_cli to route train/eval to the right
    # module and by the TUI to pick appropriate metrics.
    network_type: str = "betaone"

    # Architecture (recorded for compatibility checking)
    architecture: dict = field(default_factory=lambda: dict(ARCH_META))

    # Training hyperparameters
    training: dict = field(default_factory=dict)

    # Data source — must point at a frozen encounter set.
    data: dict = field(default_factory=lambda: {
        "mode": "encounter_set",
        "encounter_set": None,
    })

    # Checkpoint policy
    checkpoints: dict = field(default_factory=lambda: {
        "save_every": 10, "keep_best": True, "cold_start": False,
    })

    # Finalization: when an experiment is "done," promote a specific gen as its
    # canonical result. `info`, `compare`, and `list` then report scores at
    # concluded_gen instead of latest, so later readers don't have to re-derive
    # which gen was the peak. `None` on all three fields = not finalized.
    concluded_gen: int | None = None
    concluded_reason: str | None = None
    concluded_at: str | None = None  # ISO timestamp

    @classmethod
    def from_yaml(cls, path: str | Path) -> ExperimentConfig:
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        return cls(
            name=raw["name"],
            method=raw["method"],
            description=raw.get("description", ""),
            created=raw.get("created", ""),
            parent=raw.get("parent"),
            parent_checkpoint=raw.get("parent_checkpoint"),
            network_type=raw.get("network_type", "betaone"),
            architecture=raw.get("architecture", dict(ARCH_META)),
            training=raw.get("training", {}),
            data=raw.get("data", {"mode": "encounter_set", "encounter_set": None}),
            checkpoints=raw.get("checkpoints", {}),
            concluded_gen=raw.get("concluded_gen"),
            concluded_reason=raw.get("concluded_reason"),
            concluded_at=raw.get("concluded_at"),
        )

    def to_yaml(self, path: str | Path) -> None:
        data = {
            "name": self.name,
            "method": self.method,
            "description": self.description,
            "created": self.created,
            "parent": self.parent,
            "parent_checkpoint": self.parent_checkpoint,
            "network_type": self.network_type,
            "architecture": self.architecture,
            "training": self.training,
            "data": self.data,
            "checkpoints": self.checkpoints,
        }
        if self.concluded_gen is not None:
            data["concluded_gen"] = self.concluded_gen
            data["concluded_reason"] = self.concluded_reason
            data["concluded_at"] = self.concluded_at
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    def to_train_kwargs(self) -> dict[str, Any]:
        """Convert config to kwargs for the appropriate train() function.

        Dispatch is network_type-first (betaone vs decknet), then method-
        specific within betaone.
        """
        t = self.training
        d = self.data
        ck = self.checkpoints

        def _float(v, default=0.0):
            """Coerce string-encoded floats from YAML (e.g., '3e-4')."""
            if v is None:
                return default
            return float(v)

        # --- DeckNet: full-run self-play training, no encounter set ---
        if self.network_type == "decknet":
            dn = t.get("decknet", {})
            return {
                "num_generations": t.get("generations", 10),
                "runs_per_gen": t.get("runs_per_gen", 100),
                "mcts_sims": dn.get("mcts_sims", 200),
                "temperature": _float(dn.get("temperature"), 1.0),
                "combat_replays": dn.get("combat_replays", 1),
                "lr": _float(t.get("lr"), 3e-4),
                "batch_size": t.get("batch_size", 256),
                "train_steps_per_gen": dn.get("train_steps_per_gen", 50),
                "option_epsilon": _float(dn.get("option_epsilon"), 0.15),
                "replay_capacity": dn.get("replay_capacity", 50_000),
                "betaone_checkpoint": dn.get("betaone_checkpoint", ""),
                # Target mode: "policy_bootstrap" (MCTS revealed preference,
                # default) or "run_outcome" (legacy broadcast credit).
                "target_mode": dn.get("target_mode", "policy_bootstrap"),
            }

        # --- BetaOne: encounter-set-gated ---
        encounter_set_id = d.get("encounter_set")
        if not encounter_set_id:
            raise ValueError(
                f"Experiment '{self.name}' has no data.encounter_set. Training "
                "now runs exclusively against frozen encounter sets — legacy "
                "modes (mixed/recorded/curriculum) are no longer supported."
            )

        arch = self.architecture or {}
        value_head_layers = int(arch.get("value_head_layers", 1))
        trunk_layers = int(arch.get("trunk_layers", 2))
        trunk_hidden = int(arch.get("trunk_hidden", 128))
        policy_head_type = str(arch.get("policy_head_type", "dot_product"))
        policy_mlp_hidden = int(arch.get("policy_mlp_hidden", 64))

        if self.method == "mcts_selfplay":
            mcts = t.get("mcts", {})
            return {
                "num_generations": t.get("generations", 3000),
                "combats_per_gen": t.get("combats_per_gen", 256),
                "num_sims": mcts.get("num_sims", 400),
                "lr": _float(t.get("lr"), 3e-4),
                "lr_schedule": t.get("lr_schedule", "constant"),
                "lr_warmup_frac": _float(t.get("lr_warmup_frac"), 0.05),
                "lr_min_frac": _float(t.get("lr_min_frac"), 0.1),
                "value_coef": _float(mcts.get("value_coef"), 1.0),
                "train_epochs": mcts.get("train_epochs", 4),
                "batch_size": t.get("batch_size", 512),
                "temperature": _float(mcts.get("temperature"), 1.0),
                "replay_capacity": mcts.get("replay_capacity", 200_000),
                "cold_start": ck.get("cold_start", False),
                "encounter_set_id": encounter_set_id,
                "turn_boundary_eval": mcts.get("turn_boundary_eval", False),
                "c_puct": _float(mcts.get("c_puct"), 2.5),
                "pomcp": mcts.get("pomcp", False),
                "mcts_bootstrap": mcts.get("mcts_bootstrap", False),
                "noise_frac": _float(mcts.get("noise_frac"), 0.25),
                "pw_k": _float(mcts.get("pw_k"), 1.0),
                "q_target_mix": _float(mcts.get("q_target_mix"), 0.0),
                "q_target_temp": _float(mcts.get("q_target_temp"), 0.5),
                "eval_every": mcts.get("eval_every", 0),
                "value_head_layers": value_head_layers,
                "trunk_layers": trunk_layers,
                "trunk_hidden": trunk_hidden,
                "policy_head_type": policy_head_type,
                "policy_mlp_hidden": policy_mlp_hidden,
                "grad_conflict_sample_every": mcts.get(
                    "grad_conflict_sample_every", 10
                ),
                "save_every": ck.get("save_every", 10),
            }
        else:  # ppo
            ppo = t.get("ppo", {})
            return {
                "num_generations": t.get("generations", 2000),
                "combats_per_gen": t.get("combats_per_gen", 256),
                "lr": _float(t.get("lr"), 3e-4),
                "gamma": _float(ppo.get("gamma"), 0.99),
                "lam": _float(ppo.get("lam"), 0.95),
                "temperature_start": _float(ppo.get("temperature_start"), 1.0),
                "temperature_end": _float(ppo.get("temperature_end"), 0.5),
                "entropy_coef": _float(ppo.get("entropy_coef"), 0.03),
                "clip_ratio": _float(ppo.get("clip_ratio"), 0.2),
                "value_coef": _float(ppo.get("value_coef"), 0.5),
                "max_grad_norm": _float(ppo.get("max_grad_norm"), 0.5),
                "ppo_epochs": ppo.get("ppo_epochs", 4),
                "ppo_batch_size": ppo.get("ppo_batch_size", 256),
                "encounter_set_id": encounter_set_id,
                "value_head_layers": value_head_layers,
            }


# ---------------------------------------------------------------------------
# Experiment class
# ---------------------------------------------------------------------------

class Experiment:
    """Manages a single experiment directory."""

    def __init__(self, name: str):
        self.name = name
        # Resolve across main + worktrees so commands run from main find
        # experiments living in sibling worktrees automatically. Falls back
        # to main's experiments/<name> for create() (dir doesn't exist yet).
        self.dir = _find_experiment_dir(name)
        self.config_path = self.dir / "config.yaml"
        self.benchmarks_dir = self.dir / "benchmarks"

    @property
    def exists(self) -> bool:
        return self.config_path.exists()

    @property
    def config(self) -> ExperimentConfig:
        return ExperimentConfig.from_yaml(self.config_path)

    @property
    def _ckpt_prefix(self) -> str:
        return "decknet" if self.config.network_type == "decknet" else "betaone"

    def output_dir(self) -> str:
        """Return the path training scripts should use as output_dir."""
        return str(self.dir)

    @staticmethod
    def create(
        name: str,
        config: ExperimentConfig | None = None,
        template: str | None = None,
        overrides: dict | None = None,
    ) -> Experiment:
        """Create a new experiment from a config or template."""
        exp = Experiment(name)
        if exp.exists:
            raise FileExistsError(f"Experiment '{name}' already exists at {exp.dir}")

        # Build config
        if config is None and template:
            template_path = TEMPLATES_DIR / f"{template}.yaml"
            if not template_path.exists():
                raise FileNotFoundError(f"Template not found: {template_path}")
            config = ExperimentConfig.from_yaml(template_path)
            config.name = name

        if config is None:
            raise ValueError("Must provide config or template")

        # Set creation timestamp
        config.created = datetime.now(timezone.utc).isoformat(timespec="seconds")

        # Architecture metadata: recorded for compatibility checking. BetaOne
        # uses the combat ARCH_META; DeckNet tracks its own shape so mixing
        # networks in one dashboard doesn't confuse readers.
        if config.network_type == "decknet":
            from ..decknet.network import DeckNet
            vocab_path = BENCHMARK_DIR / "card_vocab.json"
            if vocab_path.exists():
                import json as _json
                num_cards = len(_json.loads(vocab_path.read_text(encoding="utf-8")))
            else:
                num_cards = 120
            _net = DeckNet(num_cards=num_cards)
            config.architecture = {
                "network": "decknet",
                "num_cards": num_cards,
                "total_params": _net.param_count(),
            }
        else:
            # Merge defaults UNDER the existing architecture instead of
            # overwriting it. Fresh `create` calls have config.architecture
            # = dict(ARCH_META) from the dataclass default, so the merge is
            # idempotent. Fork calls have config.architecture = parent's
            # architecture (potentially with per-experiment overrides like
            # value_head_layers=3) — those overrides survive the merge,
            # newly-added defaults from ARCH_META get filled in for missing
            # keys. This fixes the silent vhl-reset bug where forks from a
            # vhl=3 baseline silently downgraded to vhl=1, then warm-load
            # would reset value_head to random init due to shape mismatch.
            config.architecture = {**ARCH_META, **(config.architecture or {})}
            vocab_path = BENCHMARK_DIR / "card_vocab.json"
            if vocab_path.exists():
                import json as _json
                num_cards = len(_json.loads(vocab_path.read_text(encoding="utf-8")))
            else:
                num_cards = config.architecture.get("num_cards", 120)
            config.architecture["num_cards"] = num_cards

        # Apply overrides AFTER the architecture reset so architecture.*
        # overrides (e.g., value_head_layers) survive. This also means any
        # training/data/checkpoint overrides land on the parent-inherited
        # values as expected.
        if overrides:
            _apply_overrides(config, overrides)

        # Recompute params/stats from the FINAL architecture (overrides may
        # have changed value_head_layers / trunk config, which affects total_params).
        if config.network_type != "decknet":
            arch = config.architecture
            stats = network_stats(
                num_cards=int(arch.get("num_cards", 120)),
                value_head_layers=int(arch.get("value_head_layers", 1)),
                trunk_layers=int(arch.get("trunk_layers", 2)),
                trunk_hidden=int(arch.get("trunk_hidden", 128)),
                policy_head_type=str(arch.get("policy_head_type", "dot_product")),
                policy_mlp_hidden=int(arch.get("policy_mlp_hidden", 64)),
            )
            arch["total_params"] = stats["total_params"]
            arch["state_dim"] = stats["state_dim"]
            arch["trunk_input"] = stats["trunk_input"]

        # Create directory structure
        exp.dir.mkdir(parents=True, exist_ok=True)
        (exp.dir / "onnx").mkdir(exist_ok=True)
        exp.benchmarks_dir.mkdir(exist_ok=True)

        # Write config
        config.to_yaml(exp.config_path)

        # Copy canonical card vocab
        vocab_src = BENCHMARK_DIR / "card_vocab.json"
        if vocab_src.exists():
            shutil.copy2(vocab_src, exp.dir / "card_vocab.json")

        # Copy recorded encounters if needed
        if config.data.get("mode") in ("recorded", "mixed"):
            rec_src = BENCHMARK_DIR / "benchmark_recorded.jsonl"
            if rec_src.exists():
                shutil.copy2(rec_src, exp.dir / "recorded_encounters.jsonl")

        return exp

    @staticmethod
    def fork(
        new_name: str,
        source_name: str,
        checkpoint: str = "auto",
        overrides: dict | None = None,
    ) -> Experiment:
        """Fork a new experiment from an existing one's checkpoint.

        `checkpoint` resolution:
          - "auto" (default): use the source's concluded_gen if finalized,
            else use "latest". This matches the principled default — if
            someone marked an experiment concluded, that's the gen they
            want future work to build on, not whatever random latest.pt
            happened to be left over from extra training past the mark.
          - "finalized" / "concluded": require the source to be finalized
            and use its concluded_gen. Raises if source isn't finalized.
          - "latest": use betaone_latest.pt (last saved weights).
          - "genN" or raw "N": use betaone_genN.pt explicitly.
        """
        source = Experiment(source_name)
        if not source.exists:
            raise FileNotFoundError(f"Source experiment '{source_name}' not found")

        src_cfg = source.config

        # Resolve the checkpoint keyword to an actual .pt filename.
        if checkpoint == "auto":
            if src_cfg.concluded_gen is not None:
                resolved = f"gen{src_cfg.concluded_gen}"
            else:
                resolved = "latest"
        elif checkpoint in ("finalized", "concluded"):
            if src_cfg.concluded_gen is None:
                raise ValueError(
                    f"Source '{source_name}' is not finalized — no concluded_gen "
                    "to fork from. Use --checkpoint latest or finalize it first."
                )
            resolved = f"gen{src_cfg.concluded_gen}"
        else:
            resolved = checkpoint

        config = src_cfg
        config.name = new_name
        config.parent = source_name
        config.parent_checkpoint = resolved
        # A fork is a fresh experiment — don't inherit the parent's finalize
        # marker or we'd ship the child as "done at gen N" before it runs.
        config.concluded_gen = None
        config.concluded_reason = None
        config.concluded_at = None

        exp = Experiment.create(new_name, config=config, overrides=overrides)

        # Copy checkpoint with gen reset to 0
        prefix = source._ckpt_prefix
        if resolved == "latest":
            src_ckpt = source.dir / f"{prefix}_latest.pt"
        else:
            src_ckpt = source.dir / f"{prefix}_{resolved}.pt"

        if not src_ckpt.exists():
            raise FileNotFoundError(
                f"Source checkpoint {src_ckpt.name} not found in {source.dir}. "
                f"Available: {prefix}_latest.pt + any {prefix}_gen<N>.pt files."
            )

        import torch
        ckpt = torch.load(str(src_ckpt), map_location="cpu", weights_only=False)
        ckpt["gen"] = 0
        ckpt["win_rate"] = 0.0
        torch.save(ckpt, str(exp.dir / f"{exp._ckpt_prefix}_latest.pt"))

        return exp

    @staticmethod
    def list_all() -> list[dict]:
        """List all experiments with summary info across main + worktrees.

        Uses _all_experiment_sources which guarantees one authoritative dir
        per name (worktrees contribute only their own named experiment, not
        inherited copies from branching). Sorted by mtime.
        """
        results = []
        # Sort by mtime so most recently active rise to the bottom.
        sources = sorted(
            _all_experiment_sources(), key=lambda nd: nd[1].stat().st_mtime
        )
        for name, d in sources:
            config_path = d / "config.yaml"
            config = ExperimentConfig.from_yaml(config_path)
            # DeckNet writes decknet_progress.json; BetaOne writes betaone_progress.json
            prog_name = (
                "decknet_progress.json" if config.network_type == "decknet"
                else "betaone_progress.json"
            )
            progress = _read_progress(d / prog_name)
            if config.network_type == "decknet":
                dn = config.training.get("decknet", {})
                sims = dn.get("mcts_sims", "?")
                method_str = f"DeckNet-{sims}"
            elif config.method == "ppo":
                method_str = "PPO"
            else:
                mcts = config.training.get("mcts", {})
                sims = mcts.get("num_sims", "?")
                prefix = "POMCP" if mcts.get("pomcp", False) else "MCTS"
                method_str = f"{prefix}-{sims}"
            results.append({
                "name": config.name,
                "method": method_str,
                "description": config.description,
                "created": config.created,
                "gen": progress.get("gen", 0) if progress else 0,
                "win_rate": progress.get("win_rate", 0.0) if progress else 0.0,
                "best_win_rate": progress.get("best_win_rate", 0.0) if progress else 0.0,
                "concluded_gen": config.concluded_gen,
            })
        return results

    def archive(self, force: bool = False) -> dict:
        """Archive the experiment: keep the record + concluded-gen .pt, drop the rest.

        Writes the allow-list (see _archive_keep_set) to
        experiments/_archive/<name>/ on main. For worktree experiments, runs
        `git worktree remove --force` after copying — branch + commits stay
        intact, so `git worktree add <path> experiment/<name>` restores it.
        For legacy in-tree experiments, the source dir is deleted after
        copying.

        Requires concluded_gen to be set (run `finalize` first) unless
        force=True. Archiving without a pinned gen keeps only the record
        (config/PLAN/benchmarks/history/progress) — no checkpoint, since
        there's no canonical one to retain.
        """
        cfg = self.config
        if cfg.concluded_gen is None and not force:
            raise ValueError(
                f"'{self.name}' is not finalized. Finalize it first with "
                f"`sts2-experiment finalize {self.name} --gen <N> --reason \"...\"`, "
                "or pass --force to archive without a pinned gen."
            )

        archive_root = EXPERIMENTS_DIR / "_archive"
        archive_root.mkdir(exist_ok=True)
        dest = archive_root / self.name
        if dest.exists():
            raise FileExistsError(f"Archive destination exists: {dest}")

        keep_paths = _archive_keep_set(
            self.dir, self._ckpt_prefix, cfg.concluded_gen
        )
        dest.mkdir(parents=True)
        copied: list[str] = []
        total_bytes = 0
        for src in keep_paths:
            rel = src.relative_to(self.dir)
            dst = dest / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            copied.append(str(rel).replace("\\", "/"))
            total_bytes += src.stat().st_size

        # Clean up source. Worktree detection: self.dir may have been resolved
        # into a sibling worktree by _find_experiment_dir, so check whether it
        # actually lives under the expected worktree path before invoking git.
        worktree_path = _experiment_worktree_path(self.name)
        is_worktree = (
            worktree_path.exists()
            and self.dir.resolve().is_relative_to(worktree_path.resolve())
        )
        if is_worktree:
            _run_git(
                ["worktree", "remove", "--force", str(worktree_path)],
                cwd=REPO_ROOT,
            )
        else:
            shutil.rmtree(str(self.dir))

        return {
            "dest": dest,
            "kept": copied,
            "kept_bytes": total_bytes,
            "source_kind": "worktree" if is_worktree else "in-tree",
            "branch_retained": is_worktree,
        }

    def info(self) -> dict:
        """Return detailed info about this experiment.

        If the experiment is finalized (concluded_gen set), `latest_eval` and
        `latest_value_eval` are pinned to that gen. Raw "most recent row"
        readers still work — callers that want the true latest should read
        the jsonl directly.
        """
        config = self.config
        progress = _read_progress(self.dir / "betaone_progress.json")
        benchmarks = _read_latest_benchmark(self.benchmarks_dir / "results.jsonl")

        eval_path = self.benchmarks_dir / "eval.jsonl"
        value_eval_path = self.benchmarks_dir / "value_eval.jsonl"
        if config.concluded_gen is not None:
            eval_result = _read_eval_at_gen(eval_path, config.concluded_gen)
            value_eval_result = _read_eval_at_gen(value_eval_path, config.concluded_gen)
        else:
            eval_result = _read_latest_benchmark(eval_path)
            value_eval_result = _read_latest_benchmark(value_eval_path)

        return {
            "config": config,
            "progress": progress,
            "latest_benchmark": benchmarks,
            "latest_eval": eval_result,
            "latest_value_eval": value_eval_result,
            "is_concluded": config.concluded_gen is not None,
        }

    def resolve_checkpoint(self, spec: str = "auto") -> Path:
        """Resolve a checkpoint spec to a .pt file path.

        Accepts:
          - "auto" (default): finalized gen if set, else betaone_latest.pt.
            This is the principled default for benchmark/eval: whatever the
            experiment was canonically concluded as, not whatever random
            training checkpoint happens to be newest on disk.
          - "finalized" / "concluded": errors if not finalized.
          - "latest": always betaone_latest.pt.
          - "genN" or raw "N": betaone_genN.pt.

        For "auto" / "finalized" / "concluded": if betaone_gen{N}.pt has
        been rotated out, falls back to betaone_latest.pt when its recorded
        gen matches concluded_gen. Mirrors finalize()'s validation logic —
        users shouldn't have to know which form of the checkpoint survived
        disk rotation to benchmark their finalized experiment.
        """
        cfg = self.config
        prefix = self._ckpt_prefix

        def _finalized_path() -> Path:
            gen_ckpt = self.dir / f"{prefix}_gen{cfg.concluded_gen}.pt"
            if gen_ckpt.exists():
                return gen_ckpt
            latest = self.dir / f"{prefix}_latest.pt"
            if latest.exists():
                try:
                    import torch
                    ck = torch.load(str(latest), map_location="cpu", weights_only=False)
                    if ck.get("gen") == cfg.concluded_gen:
                        return latest
                except Exception:
                    pass
            # Return the gen-specific name so the caller's existence check
            # produces a precise error message.
            return gen_ckpt

        if spec == "auto":
            if cfg.concluded_gen is not None:
                return _finalized_path()
            return self.dir / f"{prefix}_latest.pt"
        if spec in ("finalized", "concluded"):
            if cfg.concluded_gen is None:
                raise ValueError(
                    f"'{self.name}' is not finalized — no concluded_gen. "
                    "Use --checkpoint latest or finalize it first."
                )
            return _finalized_path()
        if spec == "latest":
            return self.dir / f"{prefix}_latest.pt"
        # "gen30" or raw "30"
        gen_str = spec if spec.startswith("gen") else f"gen{spec}"
        return self.dir / f"{prefix}_{gen_str}.pt"

    def finalize(self, gen: int, reason: str) -> None:
        """Mark a specific gen as the experiment's canonical conclusion.

        Finalize is primarily an assertion about *scores* at a gen, so the
        hard requirement is that eval data exists at that gen. A matching
        checkpoint (either betaone_genN.pt or betaone_latest.pt recording the
        same gen) is also required so the pinned weights are recoverable.
        """
        from datetime import datetime, timezone

        # Require at least one of {eval, value_eval} at this gen — otherwise
        # we'd be pinning to empty scores.
        eval_ok = _read_eval_at_gen(self.benchmarks_dir / "eval.jsonl", gen) is not None
        vev_ok = _read_eval_at_gen(self.benchmarks_dir / "value_eval.jsonl", gen) is not None
        if not (eval_ok or vev_ok):
            raise ValueError(
                f"No eval or value_eval data at gen {gen} for {self.name}. "
                "Run `sts2-experiment eval <name>` at that gen before finalizing."
            )

        # Require a recoverable checkpoint. Accept either the gen-specific .pt
        # or the latest .pt if its recorded gen matches.
        prefix = self._ckpt_prefix
        gen_ckpt = self.dir / f"{prefix}_gen{gen}.pt"
        latest_ckpt = self.dir / f"{prefix}_latest.pt"
        ckpt_ok = gen_ckpt.exists()
        if not ckpt_ok and latest_ckpt.exists():
            import torch
            try:
                ck = torch.load(str(latest_ckpt), map_location="cpu", weights_only=False)
                ckpt_ok = ck.get("gen") == gen
            except Exception:
                pass
        if not ckpt_ok:
            raise FileNotFoundError(
                f"No checkpoint for gen {gen}. Expected {gen_ckpt.name} or "
                f"{prefix}_latest.pt@gen={gen}. The .pt may have been rotated out."
            )

        config = self.config
        config.concluded_gen = int(gen)
        config.concluded_reason = reason
        config.concluded_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        config.to_yaml(self.config_path)

    def unfinalize(self) -> None:
        """Clear the concluded-gen marker."""
        config = self.config
        config.concluded_gen = None
        config.concluded_reason = None
        config.concluded_at = None
        config.to_yaml(self.config_path)

    def save_eval(self, result: dict, suite_id: str | None = None) -> None:
        """Append eval harness results to benchmarks/eval.jsonl."""
        self.benchmarks_dir.mkdir(exist_ok=True)
        entry = {
            "suite": suite_id,
            "timestamp": time.time(),
            "gen": result.get("gen", "?"),
            "passed": result["passed"],
            "total": result["total"],
            "score": round(result["passed"] / max(result["total"], 1), 4),
            "end_turn_avg": result.get("end_turn_avg"),
            "end_turn_high": result.get("end_turn_high", 0),
            "by_category": {
                cat: {
                    "passed": sum(1 for r in results if r["passed"]),
                    "total": len(results),
                }
                for cat, results in result.get("by_category", {}).items()
            },
        }
        with open(self.benchmarks_dir / "eval.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

    def save_value_eval(self, result: dict, suite_id: str | None = None) -> None:
        """Append value head eval results to benchmarks/value_eval.jsonl."""
        self.benchmarks_dir.mkdir(exist_ok=True)
        entry = {
            "suite": suite_id,
            "timestamp": time.time(),
            "gen": result.get("gen", "?"),
            "passed": result["passed"],
            "total": result["total"],
            "score": round(result["passed"] / max(result["total"], 1), 4),
            "by_category": result.get("by_category", {}),
        }
        with open(self.benchmarks_dir / "value_eval.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

    def save_benchmark(self, result: dict, suite_id: str | None = None,
                       checkpoint: str = "latest") -> None:
        """Save a benchmark result, aggregating with existing results.

        If a result with the same inference-config key exists, wins and games
        are summed and WR/CI recomputed from the totals. This lets you
        accumulate results across runs for tighter CIs.

        Dedup key: (suite, mode, mcts_sims, pw_k, c_puct, pomcp,
        turn_boundary_eval, gen). Every MCTS inference knob that changes
        search semantics is part of the key so runs with different settings
        never silently merge. `gen` is in the key because benchmarks of
        different checkpoints are different data even with identical
        inference config — without gen, benchmarking gen 61 then gen 70
        of the same experiment silently merged their wins/games.
        """
        import math

        self.benchmarks_dir.mkdir(exist_ok=True)
        path = self.benchmarks_dir / "results.jsonl"

        def _float_key(val):
            if val is None:
                return None
            return round(float(val), 4)

        def _build_key(row_or_result) -> tuple:
            return (
                row_or_result.get("suite", suite_id),  # suite uses arg on new rows
                row_or_result.get("mode"),
                row_or_result.get("mcts_sims", 0),
                _float_key(row_or_result.get("pw_k")),
                _float_key(row_or_result.get("c_puct")),
                row_or_result.get("pomcp"),
                row_or_result.get("turn_boundary_eval"),
                row_or_result.get("gen"),
            )

        key = (
            suite_id,
            result["mode"],
            result.get("mcts_sims", 0),
            _float_key(result.get("pw_k")),
            _float_key(result.get("c_puct")),
            result.get("pomcp"),
            result.get("turn_boundary_eval"),
            result.get("gen"),
        )
        new_wins = result.get("wins", 0)
        new_games = result.get("games", 0)

        # Read existing results, find matching row
        rows = []
        merged = False
        if path.exists():
            with open(path, encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    if _build_key(row) == key:
                        # Aggregate: sum wins and games
                        row["wins"] = row.get("wins", 0) + new_wins
                        row["games"] = row.get("games", 0) + new_games
                        n = row["games"]
                        row["win_rate"] = round(row["wins"] / max(n, 1), 4)
                        # Recompute Wilson CI
                        z = 1.96
                        p_hat = row["wins"] / max(n, 1)
                        denom = 1 + z * z / n
                        center = (p_hat + z * z / (2 * n)) / denom
                        margin = z * math.sqrt((p_hat * (1 - p_hat) + z * z / (4 * n)) / n) / denom
                        row["ci95_lo"] = round(max(0, center - margin), 4)
                        row["ci95_hi"] = round(min(1, center + margin), 4)
                        row["timestamp"] = time.time()
                        row["gen"] = result.get("gen")
                        merged = True
                    rows.append(row)

        if not merged:
            rows.append({
                "suite": suite_id,
                "mode": result["mode"],
                "mcts_sims": result.get("mcts_sims", 0),
                "pw_k": result.get("pw_k"),
                "c_puct": result.get("c_puct"),
                "pomcp": result.get("pomcp"),
                "turn_boundary_eval": result.get("turn_boundary_eval"),
                "timestamp": time.time(),
                "checkpoint": checkpoint,
                "gen": result.get("gen"),
                "win_rate": result["win_rate"],
                "wins": new_wins,
                "games": new_games,
                "ci95_lo": result.get("ci95_lo"),
                "ci95_hi": result.get("ci95_hi"),
            })

        # Rewrite file
        with open(path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _apply_overrides(config: ExperimentConfig, overrides: dict) -> None:
    """Apply dot-notation overrides like {'training.lr': 1e-3}."""
    for key, value in overrides.items():
        parts = key.split(".")
        obj: Any = config
        for part in parts[:-1]:
            if isinstance(obj, dict):
                obj = obj.setdefault(part, {})
            else:
                obj = getattr(obj, part)
        final = parts[-1]
        if isinstance(obj, dict):
            obj[final] = value
        else:
            setattr(obj, final, value)


def _read_progress(path: Path) -> dict | None:
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return None


def _read_latest_benchmark(path: Path) -> dict | None:
    if not path.exists():
        return None
    last_line = None
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                last_line = line
    if last_line:
        return json.loads(last_line)
    return None


def _read_eval_at_gen(path: Path, gen: int) -> dict | None:
    """Return the most recent eval row for the given gen, or None if absent.
    Picks the last entry so re-runs of eval at the same gen overwrite earlier
    ones in the surfaced view.
    """
    if not path.exists():
        return None
    matched = None
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("gen") == gen or row.get("generation") == gen:
                matched = row
    return matched
