from sts2_solver.betaone import worker_orchestration as workers


def test_ecr_repository_for_region_rewrites_region_only():
    repo = "700694289572.dkr.ecr.us-east-1.amazonaws.com/sts2-worker"

    assert (
        workers.ecr_repository_for_region(repo, "us-west-2")
        == "700694289572.dkr.ecr.us-west-2.amazonaws.com/sts2-worker"
    )


def test_render_worker_user_data_keeps_cloud_init_shebang(tmp_path):
    template = tmp_path / "cloud-init.sh"
    template.write_text("#!/usr/bin/env bash\nset -euo pipefail\necho boot\n", encoding="utf-8")

    script = workers.render_worker_user_data(
        experiment="exp-a",
        coordinator_url="http://100.100.101.1:8765",
        tailscale_auth_key="tskey-test",
        worker_image="700694289572.dkr.ecr.us-east-1.amazonaws.com/sts2-worker:exp-a-abc",
        worker_git_sha="abc",
        aws_region="us-east-1",
        threads_per_worker=1,
        worker_count=7,
        template_path=template,
    )

    lines = script.splitlines()
    assert lines[0] == "#!/usr/bin/env bash"
    assert "export EXPERIMENT=exp-a" in script
    assert "export WORKER_COUNT=7" in script
    assert "set -euo pipefail" in script


def test_launch_plan_caps_last_instance_worker_count(monkeypatch):
    monkeypatch.setattr(workers, "experiment_git_sha", lambda name: "abc")
    monkeypatch.setattr(workers, "describe_instance_vcpus", lambda instance_type, region: 16)
    config = {
        "coordinator_url": "http://100.100.101.1:8765",
        "ami": "ami-test",
        "security_group_id": "sg-test",
        "subnet_id": "subnet-test",
    }

    plan = workers.make_launch_plan(
        experiment="exp-a",
        max_workers=17,
        config=config,
        regions=["us-east-1"],
        instance_types=["c7i.4xlarge"],
        image="700694289572.dkr.ecr.us-east-1.amazonaws.com/sts2-worker:exp-a-abc",
        threads_per_worker=1,
    )

    assert [unit.workers for unit in plan.units] == [16, 1]
    assert plan.planned_workers == 17


def test_latest_image_record_can_select_generation_sha(tmp_path, monkeypatch):
    exp_dir = tmp_path / "experiments" / "exp-a"
    gen_root = exp_dir / "shards" / "gen0005"
    gen_root.mkdir(parents=True)
    (gen_root / "shared.json").write_text(
        '{"required_worker_fingerprint": {"git_sha": "sha-new"}}',
        encoding="utf-8",
    )
    (exp_dir / workers.WORKER_IMAGE_RECORD).write_text(
        "\n".join(
            [
                '{"git_sha": "sha-old", "image": "repo:old"}',
                '{"git_sha": "sha-new", "image": "repo:new"}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    class FakeExperiment:
        def __init__(self, name):
            self.dir = exp_dir

    monkeypatch.setattr(workers, "Experiment", FakeExperiment)

    assert workers.latest_image_record("exp-a", gen=5)["image"] == "repo:new"


def test_launch_plan_requires_recorded_image_region(monkeypatch):
    monkeypatch.setattr(workers, "experiment_git_sha", lambda name: "abc")
    config = {
        "coordinator_url": "http://100.100.101.1:8765",
        "ami": "ami-test",
        "security_group_id": "sg-test",
        "subnet_id": "subnet-test",
    }
    record = {
        "git_sha": "abc",
        "image": "700694289572.dkr.ecr.us-east-1.amazonaws.com/sts2-worker:exp-a-abc",
        "images_by_region": {
            "us-east-1": "700694289572.dkr.ecr.us-east-1.amazonaws.com/sts2-worker:exp-a-abc"
        },
    }

    try:
        workers.make_launch_plan(
            experiment="exp-a",
            max_workers=1,
            config=config,
            regions=["us-west-2"],
            instance_types=["c7i.4xlarge"],
            image=record["image"],
            image_record=record,
            threads_per_worker=1,
        )
    except ValueError as exc:
        assert "us-west-2" in str(exc)
    else:
        raise AssertionError("expected missing region image to fail")
