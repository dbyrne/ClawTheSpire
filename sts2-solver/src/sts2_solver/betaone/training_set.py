"""Immutable, versioned training sets for BetaOne.

A training set is a pre-calibrated collection of encounters with fixed HP
values. It doesn't change during training — calibration is done offline.

Training sets live in experiments/_benchmark/training_sets/.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import yaml

from .paths import BENCHMARK_DIR

TRAINING_SETS_DIR = BENCHMARK_DIR / "training_sets"


def _content_hash(data: str) -> str:
    return hashlib.sha256(data.encode("utf-8")).hexdigest()[:12]


def load_training_set(ts_id: str) -> dict:
    """Load a training set by ID. Returns the full definition."""
    path = TRAINING_SETS_DIR / f"{ts_id}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Training set not found: {path}")
    with open(path, encoding="utf-8") as f:
        ts = yaml.safe_load(f)

    # Load the actual data files
    ts_dir = path.parent
    rec_file = ts.get("recorded_file")
    if rec_file:
        rec_path = ts_dir / rec_file
        if rec_path.exists():
            with open(rec_path, encoding="utf-8") as f:
                ts["recorded_data"] = [json.loads(l) for l in f if l.strip()]
        else:
            ts["recorded_data"] = []

    pkg_file = ts.get("packages_file")
    if pkg_file:
        pkg_path = ts_dir / pkg_file
        if pkg_path.exists():
            with open(pkg_path, encoding="utf-8") as f:
                ts["packages_data"] = json.load(f)
        else:
            ts["packages_data"] = {}

    return ts


def save_training_set(
    name: str,
    recorded_encounters: list[dict],
    package_hps: dict[str, dict[str, int]],
    calibrated_with: dict | None = None,
) -> str:
    """Save a training set. Returns the training_set_id.

    Args:
        name: human-readable name
        recorded_encounters: list of encounter dicts with calibrated_hp
        package_hps: {package_name: {encounter_key: hp}}
        calibrated_with: metadata about how calibration was done
    """
    TRAINING_SETS_DIR.mkdir(parents=True, exist_ok=True)

    # Write recorded encounters
    rec_lines = [json.dumps(r) for r in recorded_encounters]
    rec_content = "\n".join(rec_lines) + "\n" if rec_lines else ""
    rec_hash = _content_hash(rec_content)
    rec_filename = f"recorded-{rec_hash}.jsonl"
    with open(TRAINING_SETS_DIR / rec_filename, "w", encoding="utf-8") as f:
        f.write(rec_content)

    # Write package calibration
    pkg_content = json.dumps(package_hps, indent=2, sort_keys=True)
    pkg_hash = _content_hash(pkg_content)
    pkg_filename = f"packages-{pkg_hash}.json"
    with open(TRAINING_SETS_DIR / pkg_filename, "w", encoding="utf-8") as f:
        f.write(pkg_content)

    # Compute training set ID from combined content
    ts_id = f"ts-{_content_hash(rec_content + pkg_content)}"

    # Write the YAML definition
    ts = {
        "training_set_id": ts_id,
        "name": name,
        "recorded_file": rec_filename,
        "recorded_count": len(recorded_encounters),
        "recorded_avg_hp": round(
            sum(r.get("calibrated_hp", 70) for r in recorded_encounters)
            / max(len(recorded_encounters), 1), 1
        ),
        "packages_file": pkg_filename,
        "packages_count": sum(len(v) for v in package_hps.values()),
        "calibrated_with": calibrated_with,
    }
    with open(TRAINING_SETS_DIR / f"{ts_id}.yaml", "w", encoding="utf-8") as f:
        yaml.dump(ts, f, default_flow_style=False, sort_keys=False)

    return ts_id


def list_training_sets() -> list[dict]:
    """List all saved training sets."""
    if not TRAINING_SETS_DIR.exists():
        return []
    results = []
    for f in sorted(TRAINING_SETS_DIR.iterdir()):
        if f.suffix == ".yaml" and f.stem.startswith("ts-"):
            with open(f, encoding="utf-8") as fh:
                results.append(yaml.safe_load(fh))
    return results
