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
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from .network import ARCH_META, network_stats
from .paths import EXPERIMENTS_DIR, BENCHMARK_DIR, TEMPLATES_DIR


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------

@dataclass
class ExperimentConfig:
    # Metadata
    name: str
    method: str  # "mcts_selfplay" or "ppo"
    description: str = ""
    created: str = ""
    parent: str | None = None
    parent_checkpoint: str | None = None

    # Architecture (recorded for compatibility checking)
    architecture: dict = field(default_factory=lambda: dict(ARCH_META))

    # Training hyperparameters
    training: dict = field(default_factory=dict)

    # Data source
    data: dict = field(default_factory=lambda: {"mode": "mixed", "recorded_frac": 0.5})

    # Curriculum
    curriculum: dict = field(default_factory=lambda: {
        "start_tier": 0, "lock_tier": None, "skip_to_final": False,
    })

    # Checkpoint policy
    checkpoints: dict = field(default_factory=lambda: {
        "save_every": 10, "keep_best": True, "cold_start": False,
    })

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
            architecture=raw.get("architecture", dict(ARCH_META)),
            training=raw.get("training", {}),
            data=raw.get("data", {"mode": "mixed", "recorded_frac": 0.5}),
            curriculum=raw.get("curriculum", {}),
            checkpoints=raw.get("checkpoints", {}),
        )

    def to_yaml(self, path: str | Path) -> None:
        data = {
            "name": self.name,
            "method": self.method,
            "description": self.description,
            "created": self.created,
            "parent": self.parent,
            "parent_checkpoint": self.parent_checkpoint,
            "architecture": self.architecture,
            "training": self.training,
            "data": self.data,
            "curriculum": self.curriculum,
            "checkpoints": self.checkpoints,
        }
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    def to_train_kwargs(self) -> dict[str, Any]:
        """Convert config to kwargs for selfplay_train.train() or train.train()."""
        t = self.training
        d = self.data
        c = self.curriculum
        ck = self.checkpoints

        if self.method == "mcts_selfplay":
            mcts = t.get("mcts", {})
            return {
                "num_generations": t.get("generations", 3000),
                "combats_per_gen": t.get("combats_per_gen", 256),
                "num_sims": mcts.get("num_sims", 400),
                "lr": t.get("lr", 3e-4),
                "value_coef": mcts.get("value_coef", 1.0),
                "train_epochs": mcts.get("train_epochs", 4),
                "batch_size": t.get("batch_size", 512),
                "temperature": mcts.get("temperature", 1.0),
                "replay_capacity": mcts.get("replay_capacity", 200_000),
                "skip_to_final": c.get("skip_to_final", False),
                "recorded_encounters": d.get("mode") in ("recorded", "mixed"),
                "mixed": d.get("mode") == "mixed",
                "recorded_frac": d.get("recorded_frac", 0.5),
                "cold_start": ck.get("cold_start", False),
            }
        else:  # ppo
            ppo = t.get("ppo", {})
            return {
                "num_generations": t.get("generations", 2000),
                "combats_per_gen": t.get("combats_per_gen", 256),
                "lr": t.get("lr", 3e-4),
                "gamma": ppo.get("gamma", 0.99),
                "lam": ppo.get("lam", 0.95),
                "temperature_start": ppo.get("temperature_start", 1.0),
                "temperature_end": ppo.get("temperature_end", 0.5),
                "entropy_coef": ppo.get("entropy_coef", 0.03),
                "clip_ratio": ppo.get("clip_ratio", 0.2),
                "value_coef": ppo.get("value_coef", 0.5),
                "max_grad_norm": ppo.get("max_grad_norm", 0.5),
                "ppo_epochs": ppo.get("ppo_epochs", 4),
                "ppo_batch_size": ppo.get("ppo_batch_size", 256),
                "skip_to_final": c.get("skip_to_final", False),
                "lock_tier": c.get("lock_tier"),
                "recorded_encounters": d.get("mode") in ("recorded", "mixed"),
                "mixed": d.get("mode") == "mixed",
                "recorded_frac": d.get("recorded_frac", 0.5),
            }


# ---------------------------------------------------------------------------
# Experiment class
# ---------------------------------------------------------------------------

class Experiment:
    """Manages a single experiment directory."""

    def __init__(self, name: str):
        self.name = name
        self.dir = EXPERIMENTS_DIR / name
        self.config_path = self.dir / "config.yaml"
        self.benchmarks_dir = self.dir / "benchmarks"

    @property
    def exists(self) -> bool:
        return self.config_path.exists()

    @property
    def config(self) -> ExperimentConfig:
        return ExperimentConfig.from_yaml(self.config_path)

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

        # Apply overrides
        if overrides:
            _apply_overrides(config, overrides)

        # Set creation timestamp
        config.created = datetime.now(timezone.utc).isoformat(timespec="seconds")

        # Ensure architecture matches current code and compute network stats
        config.architecture = dict(ARCH_META)
        num_cards = config.architecture.get("num_cards", 120)
        stats = network_stats(num_cards)
        config.architecture["num_cards"] = stats["num_cards"]
        config.architecture["total_params"] = stats["total_params"]
        config.architecture["state_dim"] = stats["state_dim"]
        config.architecture["trunk_input"] = stats["trunk_input"]

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
        checkpoint: str = "latest",
        overrides: dict | None = None,
    ) -> Experiment:
        """Fork a new experiment from an existing one's checkpoint."""
        source = Experiment(source_name)
        if not source.exists:
            raise FileNotFoundError(f"Source experiment '{source_name}' not found")

        config = source.config
        config.name = new_name
        config.parent = source_name
        config.parent_checkpoint = checkpoint

        exp = Experiment.create(new_name, config=config, overrides=overrides)

        # Copy checkpoint
        if checkpoint == "latest":
            src_ckpt = source.dir / "betaone_latest.pt"
        else:
            src_ckpt = source.dir / f"betaone_{checkpoint}.pt"

        if src_ckpt.exists():
            shutil.copy2(src_ckpt, exp.dir / "betaone_latest.pt")

        return exp

    @staticmethod
    def list_all() -> list[dict]:
        """List all experiments with summary info."""
        if not EXPERIMENTS_DIR.exists():
            return []
        results = []
        for d in sorted(EXPERIMENTS_DIR.iterdir()):
            if d.name.startswith("_") or not d.is_dir():
                continue
            config_path = d / "config.yaml"
            if not config_path.exists():
                continue
            config = ExperimentConfig.from_yaml(config_path)
            progress = _read_progress(d / "betaone_progress.json")
            results.append({
                "name": config.name,
                "method": config.method,
                "description": config.description,
                "created": config.created,
                "gen": progress.get("gen", 0) if progress else 0,
                "win_rate": progress.get("win_rate", 0.0) if progress else 0.0,
                "best_win_rate": progress.get("best_win_rate", 0.0) if progress else 0.0,
            })
        return results

    def archive(self) -> None:
        """Move experiment to _archive/."""
        archive_dir = EXPERIMENTS_DIR / "_archive"
        archive_dir.mkdir(exist_ok=True)
        dest = archive_dir / self.name
        if dest.exists():
            raise FileExistsError(f"Archive destination exists: {dest}")
        shutil.move(str(self.dir), str(dest))

    def info(self) -> dict:
        """Return detailed info about this experiment."""
        config = self.config
        progress = _read_progress(self.dir / "betaone_progress.json")
        benchmarks = _read_latest_benchmark(self.benchmarks_dir / "results.jsonl")
        eval_result = _read_latest_benchmark(self.benchmarks_dir / "eval.jsonl")
        return {
            "config": config,
            "progress": progress,
            "latest_benchmark": benchmarks,
            "latest_eval": eval_result,
        }

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

    def save_benchmark(self, result: dict, suite_id: str | None = None,
                       checkpoint: str = "latest") -> None:
        """Append a single benchmark result to benchmarks/results.jsonl.

        Args:
            result: Dict with mode, win_rate, wins, games, ci95_lo, ci95_hi, gen.
            suite_id: Benchmark suite this was measured against.
            checkpoint: Which checkpoint was tested.
        """
        self.benchmarks_dir.mkdir(exist_ok=True)
        entry = {
            "suite": suite_id,
            "mode": result["mode"],
            "timestamp": time.time(),
            "checkpoint": checkpoint,
            "gen": result.get("gen"),
            "win_rate": result["win_rate"],
            "wins": result.get("wins"),
            "games": result.get("games"),
            "ci95_lo": result.get("ci95_lo"),
            "ci95_hi": result.get("ci95_hi"),
        }
        with open(self.benchmarks_dir / "results.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")


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
