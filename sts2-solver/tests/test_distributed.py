import time

from sts2_solver.betaone import distributed as dist
from sts2_solver.betaone import distributed_worker as worker


def _write_claimable_shard(root, *, fingerprint=None):
    dist.status_dir(root).mkdir(parents=True)
    dist.jobs_dir(root).mkdir(parents=True)
    dist.results_dir(root).mkdir(parents=True)
    dist.atomic_write_json(
        root / "shared.json",
        {
            "experiment": "exp-a",
            "gen": 1,
            "required_worker_fingerprint": fingerprint,
        },
    )
    dist.atomic_write_json(
        dist.job_path(root, "shard-0000"),
        {
            "schema_version": 1,
            "experiment": "exp-a",
            "gen": 1,
            "plan_id": "p1",
            "shard_id": "shard-0000",
        },
    )
    dist.atomic_write_json(
        dist.status_path(root, "shard-0000"),
        {
            "state": "pending",
            "target_combats": 8,
            "combat_offset": 0,
            "plan_id": "p1",
            "shard_id": "shard-0000",
        },
    )


def test_claim_requires_matching_code_fingerprint(tmp_path):
    root = tmp_path / "shards" / "gen0001"
    expected = {"code_protocol": 2, "git_sha": "abc", "state_dim": 123}
    _write_claimable_shard(root, fingerprint=expected)

    stale = dist.claim_next_job_in_root(
        root,
        worker_id="old-worker",
        worker_fingerprint={**expected, "git_sha": "def"},
    )
    current = dist.claim_next_job_in_root(
        root,
        worker_id="current-worker",
        worker_fingerprint=expected,
    )

    assert stale is None
    assert current is not None
    assert current.status["worker"] == "current-worker"
    assert current.status["worker_fingerprint"] == expected


def test_mark_complete_rejects_mismatched_code_fingerprint(tmp_path):
    root = tmp_path / "shards" / "gen0001"
    expected = {"code_protocol": 2, "git_sha": "abc", "state_dim": 123}
    _write_claimable_shard(root, fingerprint=expected)
    status = dist.claim_next_job_in_root(
        root,
        worker_id="current-worker",
        worker_fingerprint=expected,
    )
    assert status is not None

    stale_status = dist.mark_complete(
        root,
        "shard-0000",
        worker_id="current-worker",
        result_bytes=dist.dumps_rollout({"worker": "old-code"}),
        worker_fingerprint={**expected, "state_dim": 999},
    )

    assert stale_status["state"] == "running"
    assert stale_status["stale_result_ignored"]["worker"] == "current-worker"
    assert "state_dim" in stale_status["stale_result_ignored"]["mismatches"][0]
    assert not dist.result_path(root, "shard-0000").exists()


def test_mark_complete_keeps_first_result_authoritative(tmp_path):
    root = tmp_path / "shards" / "gen0001"
    dist.status_dir(root).mkdir(parents=True)
    dist.results_dir(root).mkdir(parents=True)
    dist.atomic_write_json(
        dist.status_path(root, "shard-0000"),
        {
            "state": "running",
            "target_combats": 8,
            "combat_offset": 0,
            "plan_id": "p1",
            "shard_id": "shard-0000",
            "worker": "first-worker",
        },
    )

    first = dist.dumps_rollout({"worker": "first"})
    second = dist.dumps_rollout({"worker": "second"})
    first_status = dist.mark_complete(
        root,
        "shard-0000",
        worker_id="first-worker",
        result_bytes=first,
    )
    second_status = dist.mark_complete(
        root,
        "shard-0000",
        worker_id="second-worker",
        result_bytes=second,
    )

    assert first_status["worker"] == "first-worker"
    assert second_status["worker"] == "first-worker"
    assert dist.loads_rollout(dist.result_path(root, "shard-0000").read_bytes()) == {
        "worker": "first"
    }


def test_mark_complete_ignores_stale_worker_result(tmp_path):
    root = tmp_path / "shards" / "gen0001"
    dist.status_dir(root).mkdir(parents=True)
    dist.results_dir(root).mkdir(parents=True)
    dist.atomic_write_json(
        dist.status_path(root, "shard-0000"),
        {
            "state": "running",
            "target_combats": 8,
            "combat_offset": 0,
            "plan_id": "p1",
            "shard_id": "shard-0000",
            "worker": "current-worker",
        },
    )

    stale_status = dist.mark_complete(
        root,
        "shard-0000",
        worker_id="old-worker",
        result_bytes=dist.dumps_rollout({"worker": "old"}),
    )

    assert stale_status["state"] == "running"
    assert stale_status["worker"] == "current-worker"
    assert stale_status["stale_result_ignored"]["worker"] == "old-worker"
    assert not dist.result_path(root, "shard-0000").exists()


def test_heartbeat_does_not_steal_reclaimed_shard(tmp_path):
    root = tmp_path / "shards" / "gen0001"
    dist.status_dir(root).mkdir(parents=True)
    dist.atomic_write_json(
        root / "shared.json",
        {"experiment": "exp-a", "gen": 1, "required_worker_fingerprint": None},
    )
    dist.atomic_write_json(
        dist.status_path(root, "shard-0000"),
        {
            "state": "running",
            "target_combats": 8,
            "combat_offset": 0,
            "plan_id": "p1",
            "shard_id": "shard-0000",
            "worker": "current-worker",
        },
    )

    status = dist.heartbeat(
        root,
        "shard-0000",
        worker_id="old-worker",
        lease_s=60,
    )

    assert status["worker"] == "current-worker"
    assert status["stale_heartbeat_ignored"]["worker"] == "old-worker"
    assert status["state"] == "running"


def test_mark_failed_does_not_clobber_reclaimed_shard(tmp_path):
    root = tmp_path / "shards" / "gen0001"
    dist.status_dir(root).mkdir(parents=True)
    dist.atomic_write_json(
        dist.status_path(root, "shard-0000"),
        {
            "state": "running",
            "target_combats": 8,
            "combat_offset": 0,
            "plan_id": "p1",
            "shard_id": "shard-0000",
            "worker": "current-worker",
        },
    )

    status = dist.mark_failed(
        root,
        "shard-0000",
        worker_id="old-worker",
        error="late failure",
    )

    assert status["worker"] == "current-worker"
    assert status["state"] == "running"
    assert status["stale_failure_ignored"]["worker"] == "old-worker"
    assert "error" not in status


def test_atomic_write_retries_windows_replace_race(tmp_path, monkeypatch):
    path = tmp_path / "status.json"
    original_replace = dist.os.replace
    calls = {"count": 0}

    def flaky_replace(src, dst):
        calls["count"] += 1
        if calls["count"] < 3:
            raise PermissionError("temporarily locked")
        return original_replace(src, dst)

    monkeypatch.setattr(dist.os, "replace", flaky_replace)

    dist._atomic_write_bytes(path, b'{"ok": true}')

    assert calls["count"] == 3
    assert path.read_bytes() == b'{"ok": true}'


def test_worker_heartbeats_before_onnx_download(tmp_path, monkeypatch):
    events = []
    claim = {
        "job": {
            "shard_id": "shard-0000",
            "experiment": "exp-a",
            "gen": 7,
            "target_combats": 1,
        },
        "shared": {"onnx_sha256": "model-a"},
        "urls": {
            "onnx": "http://coordinator/model.onnx",
            "heartbeat": "http://coordinator/heartbeat",
            "result": "http://coordinator/result",
            "fail": "http://coordinator/fail",
        },
    }

    def fake_heartbeat(*args, **kwargs):
        events.append("heartbeat")
        return {}

    def fake_download(*args, **kwargs):
        events.append("download")

    def fake_run_selfplay_job(*args, **kwargs):
        events.append("run")
        return {"total_steps": 1}

    def fake_post(*args, **kwargs):
        events.append("post")
        return {}

    monkeypatch.setattr(worker, "_heartbeat_once", fake_heartbeat)
    monkeypatch.setattr(worker, "_download", fake_download)
    monkeypatch.setattr(worker, "_post_bytes", fake_post)
    monkeypatch.setattr(worker.dist, "run_selfplay_job", fake_run_selfplay_job)
    monkeypatch.setattr(worker.dist, "dumps_rollout", lambda rollout: b"rollout")

    worker._run_claim(
        claim,
        worker_id="worker-a",
        fingerprint={"code_protocol": 2},
        cache_dir=tmp_path,
        lease_s=60,
    )

    assert events[:3] == ["heartbeat", "download", "run"]
    assert events[-1] == "post"


def test_claim_skips_completed_locked_root(tmp_path):
    root = tmp_path / "shards" / "gen0001"
    dist.status_dir(root).mkdir(parents=True)
    dist.atomic_write_json(
        root / "shared.json",
        {"experiment": "exp-a", "gen": 1, "required_worker_fingerprint": None},
    )
    dist.atomic_write_json(
        dist.status_path(root, "shard-0000"),
        {
            "state": "done",
            "target_combats": 8,
            "combat_offset": 0,
            "plan_id": "p1",
            "shard_id": "shard-0000",
        },
    )
    (root / ".coordination.lock").write_text("stale-ish", encoding="ascii")

    started = time.perf_counter()
    claimed = dist.claim_next_job_in_root(root, worker_id="worker-a")

    assert claimed is None
    assert time.perf_counter() - started < 1.0


def test_claim_only_considers_latest_generation(tmp_path):
    exp_dir = tmp_path / "exp-a"
    older = exp_dir / "shards" / "gen0001"
    latest = exp_dir / "shards" / "gen0002"
    _write_claimable_shard(older)
    dist.atomic_write_json(older / "plan.json", {"gen": 1})
    dist.status_dir(latest).mkdir(parents=True)
    dist.atomic_write_json(latest / "plan.json", {"gen": 2})
    dist.atomic_write_json(
        latest / "shared.json",
        {"experiment": "exp-a", "gen": 2, "required_worker_fingerprint": None},
    )
    dist.atomic_write_json(
        dist.status_path(latest, "shard-0000"),
        {
            "state": "done",
            "target_combats": 8,
            "combat_offset": 0,
            "plan_id": "p2",
            "shard_id": "shard-0000",
        },
    )

    claimed = dist.claim_next_job([("exp-a", exp_dir)], worker_id="worker-a")

    assert claimed is None
