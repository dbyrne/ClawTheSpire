"""Benchmark suite versioning.

A suite is a named, versioned definition of what's being tested.
Results are only comparable when measured against the same suite.
Suite definitions live in experiments/_benchmark/suites/.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import yaml

from .paths import BENCHMARK_DIR

SUITES_DIR = BENCHMARK_DIR / "suites"


def _file_hash(path: str | Path) -> str:
    """SHA-256 of a file's contents, truncated to 12 hex chars."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


def _content_hash(data: str | bytes) -> str:
    """SHA-256 of arbitrary content, truncated to 12 hex chars."""
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Suite definitions
# ---------------------------------------------------------------------------

def compute_final_exam_suite(
    encounter_pool_path: str,
    seed: int = 42,
    combats: int = 256,
) -> dict:
    """Suite for the curriculum tier-10 final exam (mixed encounters, 70 HP)."""
    return {
        "name": "final-exam",
        "type": "final-exam",
        "seed": seed,
        "combats": combats,
        "encounter_pool_hash": _file_hash(encounter_pool_path) if os.path.exists(encounter_pool_path) else None,
    }


def compute_recorded_suite(
    recorded_path: str,
    combats_per: int = 32,
) -> dict:
    """Suite for the frozen recorded death encounters with calibrated HP."""
    if not os.path.exists(recorded_path):
        return {
            "name": "recorded",
            "type": "recorded",
            "recorded_count": 0,
            "recorded_hash": None,
            "combats_per": combats_per,
        }
    with open(recorded_path, encoding="utf-8") as f:
        lines = [l for l in f if l.strip()]
    return {
        "name": "recorded",
        "type": "recorded",
        "recorded_count": len(lines),
        "recorded_hash": _content_hash("".join(lines)),
        "combats_per": combats_per,
    }


def compute_eval_suite() -> dict:
    """Compute an eval benchmark suite definition from current scenario set.

    Hashes BOTH policy-eval scenarios (build_scenarios) AND value-eval
    comparisons (build_value_comparisons) so any change to either harness
    bumps the suite_id. Previously only policy scenarios were hashed, which
    let the value-eval set silently grow from 25 -> 121 items while the
    suite_id stayed identical — rows tagged with that ID were NOT actually
    apples-to-apples.
    """
    from .eval import build_scenarios, build_value_comparisons
    scenarios = build_scenarios()
    comparisons = build_value_comparisons()

    policy_sig_parts = [
        f"{sc.category}/{sc.name}:{sorted(sc.best_actions)}" for sc in scenarios
    ]
    value_sig_parts = [
        f"{c.category}/{c.name}" for c in comparisons
    ]
    policy_sig = "\n".join(sorted(policy_sig_parts))
    value_sig = "\n".join(sorted(value_sig_parts))
    combined_sig = f"POLICY:\n{policy_sig}\nVALUE:\n{value_sig}"

    categories = sorted(set(sc.category for sc in scenarios))

    return {
        "name": "eval",
        "type": "eval",
        "scenario_count": len(scenarios),
        "value_scenario_count": len(comparisons),
        "scenario_hash": _content_hash(combined_sig),
        "categories": categories,
    }


def suite_id(suite: dict) -> str:
    """Generate a unique version string for a suite definition.

    Format: {name}-{hash} where hash is derived from the suite's
    content-defining fields (not timestamps).
    """
    name = suite["name"]
    stype = suite["type"]
    if stype == "final-exam":
        key = f"{suite['encounter_pool_hash']}:{suite['seed']}:{suite['combats']}"
    elif stype == "recorded":
        key = f"{suite['recorded_hash']}:{suite['recorded_count']}:{suite['combats_per']}"
    elif stype == "training-set":
        key = f"{suite.get('training_set_id', '')}:{suite.get('recorded_hash', '')}"
    elif stype == "encounter-set":
        key = f"{suite['encounter_set_id']}:{suite['content_hash']}"
    elif stype == "eval":
        key = f"{suite['scenario_hash']}:{suite['scenario_count']}"
    else:
        key = json.dumps(suite, sort_keys=True)
    return f"{name}-{_content_hash(key)}"


# ---------------------------------------------------------------------------
# Suite persistence
# ---------------------------------------------------------------------------

def save_suite(suite: dict) -> str:
    """Save a suite definition to _benchmark/suites/. Returns the suite_id."""
    sid = suite_id(suite)
    SUITES_DIR.mkdir(parents=True, exist_ok=True)
    path = SUITES_DIR / f"{sid}.yaml"
    if not path.exists():
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump({"suite_id": sid, **suite}, f, default_flow_style=False, sort_keys=False)
    return sid


def load_suite(sid: str) -> dict | None:
    """Load a suite definition by its ID."""
    path = SUITES_DIR / f"{sid}.yaml"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def list_suites() -> list[dict]:
    """List all saved suite definitions."""
    if not SUITES_DIR.exists():
        return []
    results = []
    for f in sorted(SUITES_DIR.iterdir()):
        if f.suffix == ".yaml":
            with open(f, encoding="utf-8") as fh:
                results.append(yaml.safe_load(fh))
    return results


def get_current_final_exam_suite() -> tuple[dict, str]:
    """Compute and save the current final exam suite. Returns (suite, suite_id)."""
    from .paths import SOLVER_PKG
    suite = compute_final_exam_suite(
        encounter_pool_path=str(SOLVER_PKG / "encounter_pool.json"),
    )
    sid = save_suite(suite)
    return suite, sid


def get_current_recorded_suite() -> tuple[dict, str]:
    """Compute and save the current recorded encounters suite. Returns (suite, suite_id)."""
    suite = compute_recorded_suite(
        recorded_path=str(BENCHMARK_DIR / "benchmark_recorded.jsonl"),
    )
    sid = save_suite(suite)
    return suite, sid


def compute_encounter_set_suite(es_id: str) -> dict:
    """Suite for benchmarking against an encounter set."""
    from .encounter_set import load_encounter_set, load_encounter_set_meta
    try:
        encounters = load_encounter_set(es_id)
        meta = load_encounter_set_meta(es_id) or {}
    except FileNotFoundError:
        # Fall back to legacy training set
        from .training_set import load_training_set
        ts = load_training_set(es_id)
        encounters = ts.get("recorded_data", [])
        meta = {"name": ts.get("name", es_id), "training_set_id": ts.get("training_set_id", es_id)}

    content = json.dumps(encounters, sort_keys=True)
    return {
        "name": f"es-{meta.get('name', es_id)}",
        "type": "encounter-set",
        "encounter_set_id": meta.get("encounter_set_id", es_id),
        "encounter_set_name": meta.get("name", es_id),
        "encounter_count": len(encounters),
        "content_hash": _content_hash(content),
    }


def get_encounter_set_suite(es_id: str) -> tuple[dict, str]:
    """Compute and save an encounter set benchmark suite. Returns (suite, suite_id)."""
    suite = compute_encounter_set_suite(es_id)
    sid = save_suite(suite)
    return suite, sid


# Legacy aliases
compute_training_set_suite = compute_encounter_set_suite
get_training_set_suite = get_encounter_set_suite


def get_current_eval_suite() -> tuple[dict, str]:
    """Compute and save the current eval suite. Returns (suite, suite_id)."""
    suite = compute_eval_suite()
    sid = save_suite(suite)
    return suite, sid
