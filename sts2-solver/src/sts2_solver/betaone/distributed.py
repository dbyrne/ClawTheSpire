"""File-backed distributed self-play coordinator.

The trainer owns scheduling. Laptop workers talk to the companion API, claim
one shard at a time, run the normal Rust self-play entry point, then upload a
compressed rollout artifact for the trainer to aggregate.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import pickle
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1
CODE_PROTOCOL_VERSION = 2
RESULT_SUFFIX = ".pkl.gz"
DEFAULT_LEASE_S = 240.0
MIN_LEASE_S = 60.0
MAX_LEASE_S = 240.0
_CLAIM_LOCK = threading.Lock()


def utc_ts() -> float:
    return time.time()


def iso_ts(ts: float | None = None) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(utc_ts() if ts is None else ts))


def normalize_lease_s(lease_s: float | int | str | None) -> float:
    try:
        value = float(lease_s) if lease_s is not None else DEFAULT_LEASE_S
    except (TypeError, ValueError):
        value = DEFAULT_LEASE_S
    return max(MIN_LEASE_S, min(MAX_LEASE_S, value))


def _git_sha() -> str:
    env_sha = os.environ.get("STS2_GIT_SHA")
    if env_sha:
        return env_sha
    try:
        root = Path(__file__).resolve().parents[4]
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def code_fingerprint() -> dict[str, Any]:
    from .network_constants import ACTION_DIM, ARCH_VERSION, MAX_ACTIONS, MAX_HAND, STATE_DIM

    return {
        "code_protocol": CODE_PROTOCOL_VERSION,
        "git_sha": _git_sha(),
        "network_arch_version": int(ARCH_VERSION),
        "state_dim": int(STATE_DIM),
        "action_dim": int(ACTION_DIM),
        "max_actions": int(MAX_ACTIONS),
        "max_hand": int(MAX_HAND),
    }


def fingerprint_mismatches(expected: dict | None, actual: dict | None) -> list[str]:
    if not expected:
        return []
    if not isinstance(actual, dict):
        return ["missing worker fingerprint"]
    mismatches: list[str] = []
    for key, expected_value in expected.items():
        actual_value = actual.get(key)
        if actual_value != expected_value:
            mismatches.append(f"{key}: expected {expected_value!r}, got {actual_value!r}")
    return mismatches


def clean_worker_metrics(metrics: Any) -> dict[str, Any] | None:
    if not isinstance(metrics, dict):
        return None
    out: dict[str, Any] = {}
    allowed_str = {
        "worker_id", "host", "active_shard", "python", "platform", "machine",
        "instance_id", "instance_type", "worker_group", "git_sha",
    }
    allowed_num = {
        "pid", "sampled_at", "cpu_count", "cpu_pct", "load1", "load5", "load15",
        "load_per_cpu", "rss_mb", "rayon_threads",
    }
    for key in allowed_str:
        value = metrics.get(key)
        if value is None:
            continue
        text = str(value)
        out[key] = text[:200]
    for key in allowed_num:
        value = metrics.get(key)
        if value is None:
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if key in {"pid", "cpu_count", "rayon_threads"}:
            out[key] = int(number)
        else:
            out[key] = number
    return out or None


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_write_json(path: Path, payload: dict) -> None:
    data = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
    _atomic_write_bytes(path, data)


def read_json(path: Path) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def gen_root(output_dir: str | Path, gen: int) -> Path:
    return Path(output_dir) / "shards" / f"gen{int(gen):04d}"


def status_dir(root: Path) -> Path:
    return root / "status"


def jobs_dir(root: Path) -> Path:
    return root / "jobs"


def results_dir(root: Path) -> Path:
    return root / "results"


def shared_dir(root: Path) -> Path:
    return root / "shared"


def status_path(root: Path, shard_id: str) -> Path:
    return status_dir(root) / f"{shard_id}.json"


def job_path(root: Path, shard_id: str) -> Path:
    return jobs_dir(root) / f"{shard_id}.json"


def result_path(root: Path, shard_id: str) -> Path:
    return results_dir(root) / f"{shard_id}{RESULT_SUFFIX}"


def dumps_rollout(rollout: dict) -> bytes:
    return gzip.compress(pickle.dumps(rollout, protocol=pickle.HIGHEST_PROTOCOL))


def loads_rollout(data: bytes) -> dict:
    payload = pickle.loads(gzip.decompress(data))
    if not isinstance(payload, dict):
        raise ValueError("rollout artifact did not contain a dict")
    return payload


def save_rollout(path: Path, rollout: dict) -> None:
    _atomic_write_bytes(path, dumps_rollout(rollout))


def load_rollout(path: Path) -> dict:
    with open(path, "rb") as f:
        return loads_rollout(f.read())


def merge_rollouts(rollouts: list[tuple[int, str, dict]]) -> dict:
    """Merge shard rollouts into the single-call shape expected by training."""
    merged: dict[str, Any] = {
        "total_steps": 0,
        "states": [],
        "action_features": [],
        "action_masks": [],
        "hand_card_ids": [],
        "action_card_ids": [],
        "draw_pile_ids": [],
        "discard_pile_ids": [],
        "exhaust_pile_ids": [],
        "policies": [],
        "child_visits": [],
        "child_q_values": [],
        "mcts_values": [],
        "state_jsons": [],
        "combat_indices": [],
        "outcomes": [],
        "final_hps": [],
        "combat_durations_ms": [],
        "combat_decisions": [],
        "combat_search_ms": [],
        "combat_eval_calls": [],
        "combat_value_calls": [],
        "combat_eval_ms": [],
        "combat_value_ms": [],
    }
    passthrough = [
        "states", "action_features", "action_masks", "hand_card_ids",
        "action_card_ids", "draw_pile_ids", "discard_pile_ids",
        "exhaust_pile_ids", "policies", "child_visits", "child_q_values",
        "mcts_values", "state_jsons", "outcomes", "final_hps",
        "combat_durations_ms", "combat_decisions", "combat_search_ms",
        "combat_eval_calls", "combat_value_calls", "combat_eval_ms",
        "combat_value_ms",
    ]
    for combat_offset, _shard_id, rollout in sorted(rollouts, key=lambda x: x[0]):
        merged["total_steps"] += int(rollout.get("total_steps") or 0)
        for key in passthrough:
            merged[key].extend(list(rollout.get(key, [])))
        merged["combat_indices"].extend(
            int(i) + int(combat_offset)
            for i in rollout.get("combat_indices", [])
        )
    return merged


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _plan_id(gen: int, onnx_path: str, n_combats: int, num_sims: int, shard_size: int) -> str:
    st = os.stat(onnx_path)
    raw = f"{gen}|{onnx_path}|{st.st_size}|{st.st_mtime_ns}|{n_combats}|{num_sims}|{shard_size}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _chunk_ranges(n: int, chunk_size: int) -> list[tuple[int, int]]:
    chunk_size = max(1, int(chunk_size))
    return [(i, min(i + chunk_size, n)) for i in range(0, n, chunk_size)]


def _write_pending_status(
    path: Path,
    *,
    experiment: str,
    gen: int,
    shard_id: str,
    plan_id: str,
    combat_offset: int,
    target_combats: int,
    lease_s: float,
) -> None:
    now = utc_ts()
    atomic_write_json(path, {
        "schema_version": SCHEMA_VERSION,
        "experiment": experiment,
        "gen": int(gen),
        "shard_id": shard_id,
        "plan_id": plan_id,
        "state": "pending",
        "worker": None,
        "combat_offset": int(combat_offset),
        "target_combats": int(target_combats),
        "completed_combats": 0,
        "attempts": 0,
        "created_at": now,
        "updated_at": now,
        "stale_after_s": max(normalize_lease_s(lease_s) * 1.5, 300.0),
    })


def schedule_selfplay_generation(
    *,
    output_dir: str | Path,
    experiment: str,
    gen: int,
    flat_encounters: list,
    flat_decks: list,
    flat_relics: list,
    flat_hps: list[int],
    flat_seeds: list[int],
    flat_potions: list | None = None,
    onnx_path: str,
    card_vocab_json: str,
    monster_data_json: str,
    enemy_profiles_json: str,
    num_sims: int,
    temperature: float,
    player_max_hp: int,
    player_max_energy: int,
    turn_boundary_eval: bool,
    c_puct: float,
    pomcp: bool,
    noise_frac: float,
    pw_k: float,
    shard_size: int = 16,
    lease_s: float = DEFAULT_LEASE_S,
) -> dict:
    """Create/refresh shard job files for one self-play generation."""
    lengths = {
        len(flat_encounters),
        len(flat_decks),
        len(flat_relics),
        len(flat_hps),
        len(flat_seeds),
    }
    if flat_potions is not None:
        lengths.add(len(flat_potions))
    if len(lengths) != 1:
        raise ValueError("distributed self-play inputs are not length-aligned")

    n_combats = len(flat_encounters)
    root = gen_root(output_dir, gen)
    for d in (status_dir(root), jobs_dir(root), results_dir(root), shared_dir(root)):
        d.mkdir(parents=True, exist_ok=True)

    lease_s = normalize_lease_s(lease_s)
    plan_id = _plan_id(gen, onnx_path, n_combats, num_sims, shard_size)
    onnx_dest = shared_dir(root) / "betaone.onnx"
    shutil.copy2(onnx_path, onnx_dest)
    onnx_data_src = Path(f"{onnx_path}.data")
    onnx_data_dest = Path(f"{onnx_dest}.data")
    has_onnx_data = onnx_data_src.exists()
    if has_onnx_data:
        shutil.copy2(onnx_data_src, onnx_data_dest)

    shared = {
        "schema_version": SCHEMA_VERSION,
        "experiment": experiment,
        "gen": int(gen),
        "plan_id": plan_id,
        "created_at": utc_ts(),
        "created_at_iso": iso_ts(),
        "onnx_file": onnx_dest.relative_to(root).as_posix(),
        "onnx_sha256": _sha256_file(onnx_dest),
        "onnx_data_file": onnx_data_dest.relative_to(root).as_posix() if has_onnx_data else None,
        "onnx_data_sha256": _sha256_file(onnx_data_dest) if has_onnx_data else None,
        "card_vocab_json": card_vocab_json,
        "monster_data_json": monster_data_json,
        "enemy_profiles_json": enemy_profiles_json,
        "num_sims": int(num_sims),
        "temperature": float(temperature),
        "player_max_hp": int(player_max_hp),
        "player_max_energy": int(player_max_energy),
        "turn_boundary_eval": bool(turn_boundary_eval),
        "c_puct": float(c_puct),
        "pomcp": bool(pomcp),
        "noise_frac": float(noise_frac),
        "pw_k": float(pw_k),
        "add_noise": True,
        "shard_size": int(shard_size),
        "combats": int(n_combats),
        "required_worker_fingerprint": code_fingerprint(),
    }
    atomic_write_json(root / "shared.json", shared)

    shards: list[dict] = []
    for shard_index, (start, end) in enumerate(_chunk_ranges(n_combats, shard_size)):
        shard_id = f"shard-{shard_index:04d}"
        job = {
            "schema_version": SCHEMA_VERSION,
            "experiment": experiment,
            "gen": int(gen),
            "plan_id": plan_id,
            "shard_id": shard_id,
            "shard_index": shard_index,
            "combat_offset": int(start),
            "target_combats": int(end - start),
            "encounters": flat_encounters[start:end],
            "decks": flat_decks[start:end],
            "relics": flat_relics[start:end],
            "potions": (flat_potions or [[] for _ in range(n_combats)])[start:end],
            "player_hps": [int(x) for x in flat_hps[start:end]],
            "seeds": [int(x) for x in flat_seeds[start:end]],
        }
        atomic_write_json(job_path(root, shard_id), job)
        sp = status_path(root, shard_id)
        current = read_json(sp)
        if current.get("plan_id") != plan_id or current.get("state") not in {"running", "done"}:
            _write_pending_status(
                sp,
                experiment=experiment,
                gen=gen,
                shard_id=shard_id,
                plan_id=plan_id,
                combat_offset=start,
                target_combats=end - start,
                lease_s=lease_s,
            )
        shards.append({
            "shard_id": shard_id,
            "combat_offset": start,
            "target_combats": end - start,
        })

    plan = {
        "schema_version": SCHEMA_VERSION,
        "experiment": experiment,
        "gen": int(gen),
        "plan_id": plan_id,
        "root": str(root),
        "num_shards": len(shards),
        "combats": n_combats,
        "shard_size": int(shard_size),
        "shards": shards,
    }
    atomic_write_json(root / "plan.json", plan)
    return plan


def _status_is_claimable(status: dict, now: float) -> bool:
    if not status:
        return False
    state = str(status.get("state") or "").lower()
    if state in {"pending", "failed", "stale"}:
        return True
    if state == "running":
        expires = status.get("lease_expires_at")
        try:
            return expires is not None and float(expires) < now
        except (TypeError, ValueError):
            return False
    return False


@dataclass
class ClaimedJob:
    experiment: str
    root: Path
    shard_id: str
    job: dict
    shared: dict
    status: dict

    @property
    def onnx_path(self) -> Path:
        return self.root / self.shared["onnx_file"]


def _load_claimed(root: Path, shard_id: str) -> ClaimedJob:
    shared = read_json(root / "shared.json")
    job = read_json(job_path(root, shard_id))
    status = read_json(status_path(root, shard_id))
    return ClaimedJob(
        experiment=str(shared.get("experiment") or status.get("experiment") or job.get("experiment")),
        root=root,
        shard_id=shard_id,
        job=job,
        shared=shared,
        status=status,
    )


def claim_next_job_in_root(
    root: Path,
    *,
    worker_id: str,
    worker_fingerprint: dict | None = None,
    lease_s: float = DEFAULT_LEASE_S,
) -> ClaimedJob | None:
    now = utc_ts()
    lease_s = normalize_lease_s(lease_s)
    with _CLAIM_LOCK:
        shared = read_json(root / "shared.json")
        if fingerprint_mismatches(shared.get("required_worker_fingerprint"), worker_fingerprint):
            return None
        for sp in sorted(status_dir(root).glob("*.json")):
            status = read_json(sp)
            if not status:
                continue
            if not _status_is_claimable(status, now):
                continue
            shard_id = str(status.get("shard_id") or sp.stem)
            if not job_path(root, shard_id).exists():
                continue
            status.update({
                "state": "running",
                "worker": worker_id,
                "claimed_at": status.get("claimed_at") or now,
                "updated_at": now,
                "lease_expires_at": now + float(lease_s),
                "attempts": int(status.get("attempts") or 0) + 1,
            })
            if worker_fingerprint:
                status["worker_fingerprint"] = worker_fingerprint
            atomic_write_json(sp, status)
            return _load_claimed(root, shard_id)
    return None


def iter_experiment_roots(exp_dir: Path) -> list[Path]:
    root = exp_dir / "shards"
    if not root.exists():
        return []
    return sorted(
        (p for p in root.glob("gen*") if (p / "plan.json").exists()),
        key=lambda p: p.name,
        reverse=True,
    )


def claim_next_job(
    experiments: list[tuple[str, Path]],
    *,
    worker_id: str,
    experiment: str | None = None,
    worker_fingerprint: dict | None = None,
    lease_s: float = DEFAULT_LEASE_S,
) -> ClaimedJob | None:
    lease_s = normalize_lease_s(lease_s)
    for name, exp_dir in experiments:
        if experiment and name != experiment:
            continue
        for root in iter_experiment_roots(exp_dir):
            claimed = claim_next_job_in_root(
                root,
                worker_id=worker_id,
                worker_fingerprint=worker_fingerprint,
                lease_s=lease_s,
            )
            if claimed:
                return claimed
    return None


def heartbeat(
    root: Path,
    shard_id: str,
    *,
    worker_id: str,
    lease_s: float,
    worker_fingerprint: dict | None = None,
    worker_metrics: dict | None = None,
) -> dict:
    sp = status_path(root, shard_id)
    with _CLAIM_LOCK:
        status = read_json(sp)
        if not status:
            raise FileNotFoundError(sp)
        if status.get("state") == "done":
            return status
        shared = read_json(root / "shared.json")
        mismatches = fingerprint_mismatches(
            shared.get("required_worker_fingerprint"),
            worker_fingerprint or status.get("worker_fingerprint"),
        )
        if mismatches:
            status["stale_heartbeat_ignored"] = {
                "worker": worker_id,
                "mismatches": mismatches,
                "ignored_at": utc_ts(),
            }
            atomic_write_json(sp, status)
            return status
        now = utc_ts()
        lease_s = normalize_lease_s(lease_s)
        metrics = clean_worker_metrics(worker_metrics)
        status.update({
            "state": "running",
            "worker": worker_id,
            "updated_at": now,
            "heartbeat_at": now,
            "lease_expires_at": now + float(lease_s),
        })
        if metrics:
            status["worker_metrics"] = metrics
        atomic_write_json(sp, status)
        return status


def mark_failed(
    root: Path,
    shard_id: str,
    *,
    worker_id: str,
    error: str,
    worker_metrics: dict | None = None,
) -> dict:
    sp = status_path(root, shard_id)
    with _CLAIM_LOCK:
        status = read_json(sp)
        if not status:
            raise FileNotFoundError(sp)
        now = utc_ts()
        metrics = clean_worker_metrics(worker_metrics)
        status.update({
            "state": "failed",
            "worker": worker_id,
            "updated_at": now,
            "failed_at": now,
            "error": error[-1000:],
        })
        if metrics:
            status["worker_metrics"] = metrics
        atomic_write_json(sp, status)
        return status


def mark_complete(
    root: Path,
    shard_id: str,
    *,
    worker_id: str,
    result_bytes: bytes,
    worker_fingerprint: dict | None = None,
) -> dict:
    sp = status_path(root, shard_id)
    with _CLAIM_LOCK:
        status = read_json(sp)
        if not status:
            raise FileNotFoundError(sp)
        if str(status.get("state") or "").lower() == "done":
            return status
        current_worker = str(status.get("worker") or "")
        if current_worker != worker_id:
            status["stale_result_ignored"] = {
                "worker": worker_id,
                "current_worker": current_worker or None,
                "ignored_at": utc_ts(),
            }
            atomic_write_json(sp, status)
            return status
        shared = read_json(root / "shared.json")
        mismatches = fingerprint_mismatches(
            shared.get("required_worker_fingerprint"),
            worker_fingerprint or status.get("worker_fingerprint"),
        )
        if mismatches:
            status["stale_result_ignored"] = {
                "worker": worker_id,
                "mismatches": mismatches,
                "ignored_at": utc_ts(),
            }
            atomic_write_json(sp, status)
            return status
        rp = result_path(root, shard_id)
        _atomic_write_bytes(rp, result_bytes)
        now = utc_ts()
        target = int(status.get("target_combats") or 0)
        status.update({
            "state": "done",
            "worker": worker_id,
            "updated_at": now,
            "completed_at": now,
            "completed_combats": target,
            "result_path": rp.relative_to(root).as_posix(),
            "result_bytes": len(result_bytes),
            "lease_expires_at": None,
        })
        atomic_write_json(sp, status)
        return status


def run_selfplay_job(job: dict, shared: dict, *, onnx_path: str | Path) -> dict:
    import sts2_engine

    hps = [int(x) for x in job["player_hps"]]
    if not hps:
        raise ValueError("self-play shard has no combats")
    return sts2_engine.betaone_mcts_selfplay(
        encounters_json=json.dumps(job["encounters"]),
        decks_json=json.dumps(job["decks"]),
        player_hp=hps[0],
        player_max_hp=int(shared["player_max_hp"]),
        player_max_energy=int(shared["player_max_energy"]),
        relics_json=json.dumps(job["relics"]),
        potions_json=json.dumps(job.get("potions", [[]])[0] if job.get("potions") else []),
        monster_data_json=shared["monster_data_json"],
        enemy_profiles_json=shared["enemy_profiles_json"],
        onnx_path=str(onnx_path),
        card_vocab_json=shared["card_vocab_json"],
        num_sims=int(shared["num_sims"]),
        temperature=float(shared["temperature"]),
        seeds=[int(x) for x in job["seeds"]],
        gen_id=int(shared["gen"]),
        add_noise=bool(shared.get("add_noise", True)),
        turn_boundary_eval=bool(shared["turn_boundary_eval"]),
        c_puct=float(shared["c_puct"]),
        pomcp=bool(shared["pomcp"]),
        noise_frac=float(shared["noise_frac"]),
        pw_k=float(shared["pw_k"]),
        player_hps_json=json.dumps(hps),
        potions_per_combat_json=json.dumps(job.get("potions", [])),
    )


def run_claimed_job_locally(claimed: ClaimedJob, *, worker_id: str) -> dict:
    start = utc_ts()
    rollout = run_selfplay_job(claimed.job, claimed.shared, onnx_path=claimed.onnx_path)
    data = dumps_rollout(rollout)
    status = mark_complete(
        claimed.root,
        claimed.shard_id,
        worker_id=worker_id,
        result_bytes=data,
        worker_fingerprint=claimed.status.get("worker_fingerprint"),
    )
    status["duration_s"] = round(utc_ts() - start, 3)
    atomic_write_json(status_path(claimed.root, claimed.shard_id), status)
    return rollout


def root_done(root: Path) -> bool:
    statuses = [read_json(p) for p in sorted(status_dir(root).glob("*.json"))]
    return bool(statuses) and all(s.get("state") == "done" for s in statuses)


def load_generation_rollouts(root: Path) -> list[tuple[int, str, dict]]:
    out: list[tuple[int, str, dict]] = []
    for sp in sorted(status_dir(root).glob("*.json")):
        status = read_json(sp)
        if status.get("state") != "done":
            raise RuntimeError(f"shard not done: {sp.name} state={status.get('state')}")
        shard_id = str(status.get("shard_id") or sp.stem)
        rp = root / status.get("result_path", str(result_path(root, shard_id).relative_to(root)))
        rollout = load_rollout(rp)
        out.append((int(status.get("combat_offset") or 0), shard_id, rollout))
    out.sort(key=lambda x: x[0])
    return out


def _completed_generation_is_reusable(
    root: Path,
    *,
    experiment: str,
    gen: int,
    onnx_path: str,
    n_combats: int,
    num_sims: int,
    shard_size: int,
) -> bool:
    if not root_done(root):
        return False
    shared = read_json(root / "shared.json")
    if not shared:
        return False
    checks = {
        "experiment": experiment,
        "gen": int(gen),
        "combats": int(n_combats),
        "num_sims": int(num_sims),
        "shard_size": int(shard_size),
    }
    for key, expected in checks.items():
        if shared.get(key) != expected:
            return False
    expected_hash = shared.get("onnx_sha256")
    try:
        if expected_hash == _sha256_file(Path(onnx_path)):
            return True
    except OSError:
        pass
    try:
        shared_onnx = root / str(shared.get("onnx_file") or "")
        return bool(expected_hash) and expected_hash == _sha256_file(shared_onnx)
    except OSError:
        return False


def run_distributed_selfplay_generation(
    *,
    output_dir: str | Path,
    experiment: str,
    gen: int,
    flat_encounters: list,
    flat_decks: list,
    flat_relics: list,
    flat_hps: list[int],
    flat_seeds: list[int],
    flat_potions: list | None = None,
    onnx_path: str,
    card_vocab_json: str,
    monster_data_json: str,
    enemy_profiles_json: str,
    num_sims: int,
    temperature: float,
    player_max_hp: int,
    player_max_energy: int,
    turn_boundary_eval: bool,
    c_puct: float,
    pomcp: bool,
    noise_frac: float,
    pw_k: float,
    shard_size: int,
    poll_s: float = 2.0,
    lease_s: float = DEFAULT_LEASE_S,
    local_fallback_after_s: float | None = 60.0,
    timeout_s: float | None = None,
) -> tuple[list[tuple[int, str, dict]], dict]:
    lease_s = normalize_lease_s(lease_s)
    existing_root = gen_root(output_dir, gen)
    n_combats = len(flat_encounters)
    if _completed_generation_is_reusable(
        existing_root,
        experiment=experiment,
        gen=gen,
        onnx_path=onnx_path,
        n_combats=n_combats,
        num_sims=num_sims,
        shard_size=shard_size,
    ):
        rollouts = load_generation_rollouts(existing_root)
        plan = read_json(existing_root / "plan.json")
        return rollouts, {
            "plan_id": plan.get("plan_id"),
            "num_shards": plan.get("num_shards"),
            "shard_size": int(shard_size),
            "local_shards": 0,
            "polls": 0,
            "elapsed_s": 0.0,
            "reused_completed": True,
        }

    plan = schedule_selfplay_generation(
        output_dir=output_dir,
        experiment=experiment,
        gen=gen,
        flat_encounters=flat_encounters,
        flat_decks=flat_decks,
        flat_relics=flat_relics,
        flat_potions=flat_potions,
        flat_hps=flat_hps,
        flat_seeds=flat_seeds,
        onnx_path=onnx_path,
        card_vocab_json=card_vocab_json,
        monster_data_json=monster_data_json,
        enemy_profiles_json=enemy_profiles_json,
        num_sims=num_sims,
        temperature=temperature,
        player_max_hp=player_max_hp,
        player_max_energy=player_max_energy,
        turn_boundary_eval=turn_boundary_eval,
        c_puct=c_puct,
        pomcp=pomcp,
        noise_frac=noise_frac,
        pw_k=pw_k,
        shard_size=shard_size,
        lease_s=lease_s,
    )
    root = Path(plan["root"])
    start = utc_ts()
    local_worker = f"{os.environ.get('COMPUTERNAME') or 'coordinator'}-local"
    local_shards = 0
    polls = 0

    while not root_done(root):
        elapsed = utc_ts() - start
        if timeout_s is not None and elapsed > timeout_s:
            raise TimeoutError(f"distributed self-play timed out after {elapsed:.1f}s")
        if local_fallback_after_s is not None and elapsed >= local_fallback_after_s:
            claimed = claim_next_job_in_root(root, worker_id=local_worker, lease_s=lease_s)
            if claimed:
                run_claimed_job_locally(claimed, worker_id=local_worker)
                local_shards += 1
                continue
        polls += 1
        time.sleep(max(float(poll_s), 0.25))

    return load_generation_rollouts(root), {
        "plan_id": plan["plan_id"],
        "num_shards": plan["num_shards"],
        "shard_size": int(shard_size),
        "local_shards": local_shards,
        "polls": polls,
        "elapsed_s": round(utc_ts() - start, 2),
    }
