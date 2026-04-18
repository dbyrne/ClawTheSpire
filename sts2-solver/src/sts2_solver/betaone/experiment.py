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
        hand_agg_lean = bool(arch.get("hand_agg_lean", False))

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
                "c_puct": _float(mcts.get("c_puct"), 2.5),
                "pomcp": mcts.get("pomcp", False),
                "mcts_bootstrap": mcts.get("mcts_bootstrap", False),
                "noise_frac": _float(mcts.get("noise_frac"), 0.25),
                "pw_k": _float(mcts.get("pw_k"), 1.0),
                "q_target_mix": _float(mcts.get("q_target_mix"), 0.0),
                "q_target_temp": _float(mcts.get("q_target_temp"), 0.5),
                "eval_every": mcts.get("eval_every", 0),
                "value_head_layers": value_head_layers,
                "hand_agg_lean": hand_agg_lean,
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
            config.architecture["num_cards"] = num_cards

        # Apply overrides AFTER the architecture reset so architecture.*
        # overrides (e.g., value_head_layers) survive. This also means any
        # training/data/checkpoint overrides land on the parent-inherited
        # values as expected.
        if overrides:
            _apply_overrides(config, overrides)

        # Recompute params/stats from the FINAL architecture (overrides may
        # have changed value_head_layers, which affects total_params).
        if config.network_type != "decknet":
            arch = config.architecture
            value_head_layers = int(arch.get("value_head_layers", 1))
            stats = network_stats(
                num_cards=int(arch.get("num_cards", 120)),
                value_head_layers=value_head_layers,
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
        if resolved == "latest":
            src_ckpt = source.dir / "betaone_latest.pt"
        else:
            src_ckpt = source.dir / f"betaone_{resolved}.pt"

        if not src_ckpt.exists():
            raise FileNotFoundError(
                f"Source checkpoint {src_ckpt.name} not found in {source.dir}. "
                f"Available: betaone_latest.pt + any betaone_gen<N>.pt files."
            )

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
                "concluded_gen": config.concluded_gen,
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

        def _finalized_path() -> Path:
            gen_ckpt = self.dir / f"betaone_gen{cfg.concluded_gen}.pt"
            if gen_ckpt.exists():
                return gen_ckpt
            latest = self.dir / "betaone_latest.pt"
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
            return self.dir / "betaone_latest.pt"
        if spec in ("finalized", "concluded"):
            if cfg.concluded_gen is None:
                raise ValueError(
                    f"'{self.name}' is not finalized — no concluded_gen. "
                    "Use --checkpoint latest or finalize it first."
                )
            return _finalized_path()
        if spec == "latest":
            return self.dir / "betaone_latest.pt"
        # "gen30" or raw "30"
        gen_str = spec if spec.startswith("gen") else f"gen{spec}"
        return self.dir / f"betaone_{gen_str}.pt"

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
        gen_ckpt = self.dir / f"betaone_gen{gen}.pt"
        latest_ckpt = self.dir / "betaone_latest.pt"
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
                f"betaone_latest.pt@gen={gen}. The .pt may have been rotated out."
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
        turn_boundary_eval). Every MCTS inference knob that changes search
        semantics is part of the key so runs with different settings never
        silently merge.
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
            )

        key = (
            suite_id,
            result["mode"],
            result.get("mcts_sims", 0),
            _float_key(result.get("pw_k")),
            _float_key(result.get("c_puct")),
            result.get("pomcp"),
            result.get("turn_boundary_eval"),
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
