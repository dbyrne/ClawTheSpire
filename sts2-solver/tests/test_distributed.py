from sts2_solver.betaone import distributed as dist


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
