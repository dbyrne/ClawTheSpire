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
    "dense_value_targets": False,
    "gamma": 0.99,
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
            }

        # --- BetaOne: encounter-set-gated ---
        encounter_set_id = d.get("encounter_set")
        if not encounter_set_id:
            raise ValueError(
                f"Experiment '{self.name}' has no data.encounter_set. Training "
                "now runs exclusively against frozen encounter sets — legacy "
                "modes (mixed/recorded/curriculum) are no longer supported."
            )

        if self.method == "mcts_selfplay":
            mcts = t.get("mcts", {})
            return {
                "num_generations": t.get("generations", 3000),
                "combats_per_gen": t.get("combats_per_gen", 256),
                "num_sims": mcts.get("num_sims", 400),
                "lr": _float(t.get("lr"), 3e-4),
                "value_coef": _float(mcts.get("value_coef"), 1.0),
                "train_epochs": mcts.get("train_epochs", 4),
                "batch_size": t.get("batch_size", 512),
                "temperature": _float(mcts.get("temperature"), 1.0),
                "replay_capacity": mcts.get("replay_capacity", 200_000),
                "cold_start": ck.get("cold_start", False),
                "encounter_set_id": encounter_set_id,
                "turn_boundary_eval": mcts.get("turn_boundary_eval", False),
                "dense_value_targets": mcts.get("dense_value_targets", False),
                "gamma": _float(mcts.get("gamma"), 0.99),
                "c_puct": _float(mcts.get("c_puct"), 2.5),
                "pomcp": mcts.get("pomcp", False),
                "mcts_bootstrap": mcts.get("mcts_bootstrap", False),
                "noise_frac": _float(mcts.get("noise_frac"), 0.25),
                "pw_k": _float(mcts.get("pw_k"), 1.0),
                "q_target_mix": _float(mcts.get("q_target_mix"), 0.0),
                "q_target_temp": _float(mcts.get("q_target_temp"), 0.5),
                "eval_every": mcts.get("eval_every", 0),
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
            config.architecture = dict(ARCH_META)
            vocab_path = BENCHMARK_DIR / "card_vocab.json"
            if vocab_path.exists():
                import json as _json
                num_cards = len(_json.loads(vocab_path.read_text(encoding="utf-8")))
            else:
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

        # Copy checkpoint with gen reset to 0
        if checkpoint == "latest":
            src_ckpt = source.dir / "betaone_latest.pt"
        else:
            src_ckpt = source.dir / f"betaone_{checkpoint}.pt"

        if src_ckpt.exists():
            import torch
            ckpt = torch.load(str(src_ckpt), map_location="cpu", weights_only=False)
            ckpt["gen"] = 0
            ckpt["win_rate"] = 0.0
            torch.save(ckpt, str(exp.dir / "betaone_latest.pt"))

        return exp

    @staticmethod
    def list_all() -> list[dict]:
        """List all experiments with summary info."""
        if not EXPERIMENTS_DIR.exists():
            return []
        results = []
        for d in sorted(EXPERIMENTS_DIR.iterdir(), key=lambda p: p.stat().st_mtime):
            if d.name.startswith("_") or not d.is_dir():
                continue
            config_path = d / "config.yaml"
            if not config_path.exists():
                continue
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

        If a result with the same (suite, mode, mcts_sims) key exists,
        wins and games are summed and WR/CI recomputed from the totals.
        This lets you accumulate results across runs for tighter CIs.
        """
        import math

        self.benchmarks_dir.mkdir(exist_ok=True)
        path = self.benchmarks_dir / "results.jsonl"

        key = (suite_id, result["mode"], result.get("mcts_sims", 0))
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
                    row_key = (row.get("suite"), row.get("mode"), row.get("mcts_sims", 0))
                    if row_key == key:
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
