"""Experiment-aware worker image and EC2 orchestration helpers."""

from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import distributed as dist
from .experiment import Experiment


WORKER_IMAGE_RECORD = "worker_images.jsonl"
WORKER_COORDINATOR_RECORD = "coordinator.json"

DEFAULT_INSTANCE_TYPES = [
    "c7i.4xlarge",
    "c7a.4xlarge",
    "c6i.4xlarge",
    "m7i.4xlarge",
    "m7a.4xlarge",
    "m6i.4xlarge",
]

FINGERPRINT_PATHS = [
    "sts2-solver/Dockerfile.worker",
    "sts2-solver/pyproject.toml",
    "sts2-solver/src/sts2_solver",
    "sts2-solver/scripts/docker_worker_entrypoint.py",
    "sts2-solver/sts2-engine",
]

ECR_RE = re.compile(
    r"^(?P<account>\d+)\.dkr\.ecr\.(?P<region>[a-z0-9-]+)\.amazonaws\.com/(?P<name>.+)$"
)


@dataclass
class ImageBuildPlan:
    experiment: str
    solver_root: Path
    repo_root: Path
    git_sha: str
    git_branch: str
    image: str
    images_by_region: dict[str, str]
    commands: list[list[str]]
    dirty_paths: list[str]
    fingerprint: dict[str, Any]


@dataclass
class LaunchUnit:
    region: str
    instance_type: str
    workers: int
    threads_per_worker: int
    image: str
    subnet_id: str | None
    security_group_ids: list[str]
    ami: str
    iam_instance_profile: str | None
    key_name: str | None

    @property
    def planned_worker_count(self) -> int:
        return int(self.workers)


@dataclass
class LaunchPlan:
    experiment: str
    max_workers: int
    git_sha: str
    coordinator_url: str
    market: str
    units: list[LaunchUnit]

    @property
    def planned_workers(self) -> int:
        return sum(unit.planned_worker_count for unit in self.units)


def _run(
    args: list[str],
    *,
    cwd: Path | None = None,
    input_text: str | None = None,
    capture: bool = False,
) -> subprocess.CompletedProcess:
    return subprocess.run(
        args,
        cwd=str(cwd) if cwd else None,
        input=input_text,
        text=True,
        capture_output=capture,
        check=True,
    )


def _git(args: list[str], *, cwd: Path) -> str:
    result = _run(["git", *args], cwd=cwd, capture=True)
    return result.stdout.strip()


def experiment_solver_root(name: str) -> Path:
    exp = Experiment(name)
    if not exp.exists:
        raise ValueError(f"experiment '{name}' not found")
    return exp.dir.parent.parent


def experiment_repo_root(name: str) -> Path:
    return experiment_solver_root(name).parent


def experiment_git_sha(name: str) -> str:
    return _git(["rev-parse", "HEAD"], cwd=experiment_repo_root(name))


def experiment_git_branch(name: str) -> str:
    branch = _git(["branch", "--show-current"], cwd=experiment_repo_root(name))
    return branch or _git(["rev-parse", "--short", "HEAD"], cwd=experiment_repo_root(name))


def scoped_dirty_paths(repo_root: Path) -> list[str]:
    out = _git(["status", "--porcelain", "--", *FINGERPRINT_PATHS], cwd=repo_root)
    return [line for line in out.splitlines() if line.strip()]


def experiment_code_fingerprint(name: str) -> dict[str, Any]:
    solver_root = experiment_solver_root(name)
    env = os.environ.copy()
    src = str(solver_root / "src")
    env["PYTHONPATH"] = src + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    code = (
        "import json; "
        "from sts2_solver.betaone.distributed import code_fingerprint; "
        "print(json.dumps(code_fingerprint(), sort_keys=True))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(solver_root),
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    return json.loads(result.stdout)


def generation_fingerprint(name: str, gen: int) -> dict[str, Any] | None:
    exp = Experiment(name)
    path = exp.dir / "shards" / f"gen{int(gen):04d}" / "shared.json"
    payload = dist.read_json(path)
    fingerprint = payload.get("required_worker_fingerprint")
    return fingerprint if isinstance(fingerprint, dict) else None


def assert_generation_compatible(name: str, gen: int, fingerprint: dict[str, Any]) -> None:
    expected = generation_fingerprint(name, gen)
    if not expected:
        raise ValueError(f"{name} gen {gen} has no recorded worker fingerprint")
    mismatches = dist.fingerprint_mismatches(expected, fingerprint)
    if mismatches:
        raise ValueError(
            f"current worktree is not compatible with {name} gen {gen}: "
            + "; ".join(mismatches)
        )


def sanitize_tag_prefix(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", text).strip(".-")
    return cleaned or "worker"


def _repository_without_tag(repository: str) -> str:
    tail = repository.rsplit("/", 1)[-1]
    if ":" in tail:
        raise ValueError("image repository must not include a tag")
    return repository.rstrip("/")


def ecr_parts(repository: str) -> dict[str, str] | None:
    match = ECR_RE.match(_repository_without_tag(repository))
    return match.groupdict() if match else None


def ecr_region(repository: str) -> str | None:
    parts = ecr_parts(repository)
    return parts["region"] if parts else None


def ecr_registry(repository: str) -> str:
    return _repository_without_tag(repository).split("/", 1)[0]


def ecr_repository_name(repository: str) -> str:
    return _repository_without_tag(repository).split("/", 1)[1]


def ecr_repository_for_region(repository: str, region: str) -> str:
    repository = _repository_without_tag(repository)
    parts = ecr_parts(repository)
    if not parts:
        return repository
    return (
        f"{parts['account']}.dkr.ecr.{region}.amazonaws.com/"
        f"{parts['name']}"
    )


def _aws_cli() -> str:
    found = shutil.which("aws")
    if found:
        return found
    win = Path(r"C:\Program Files\Amazon\AWSCLIV2\aws.exe")
    if win.exists():
        return str(win)
    return "aws"


def aws_json(args: list[str], *, region: str | None = None) -> Any:
    cmd = [_aws_cli()]
    if region:
        cmd.extend(["--region", region])
    cmd.extend(args)
    result = _run(cmd, capture=True)
    return json.loads(result.stdout) if result.stdout.strip() else None


def ecr_login(repository: str, *, region: str) -> None:
    password = _run(
        [_aws_cli(), "ecr", "get-login-password", "--region", region],
        capture=True,
    ).stdout
    _run(
        ["docker", "login", "--username", "AWS", "--password-stdin", ecr_registry(repository)],
        input_text=password,
    )


def ensure_ecr_repository(repository: str, *, region: str) -> None:
    name = ecr_repository_name(repository)
    try:
        aws_json(["ecr", "describe-repositories", "--repository-names", name], region=region)
    except subprocess.CalledProcessError:
        aws_json(["ecr", "create-repository", "--repository-name", name], region=region)


def worker_image_tag(repository: str, tag_prefix: str, git_sha: str) -> str:
    return f"{_repository_without_tag(repository)}:{sanitize_tag_prefix(tag_prefix)}-{git_sha}"


def build_worker_image(
    *,
    experiment: str,
    repository: str,
    tag_prefix: str | None = None,
    push: bool = False,
    ecr_login_enabled: bool = False,
    regions: list[str] | None = None,
    ensure_repository: bool = False,
    gen: int | None = None,
    allow_dirty: bool = False,
    dry_run: bool = False,
) -> ImageBuildPlan:
    solver_root = experiment_solver_root(experiment)
    repo_root = solver_root.parent
    git_sha = _git(["rev-parse", "HEAD"], cwd=repo_root)
    git_branch = _git(["branch", "--show-current"], cwd=repo_root) or git_sha[:12]
    fingerprint = experiment_code_fingerprint(experiment)
    if gen is not None:
        assert_generation_compatible(experiment, gen, fingerprint)

    dirty = scoped_dirty_paths(repo_root)
    if dirty and not allow_dirty:
        raise ValueError(
            "worker image source has uncommitted fingerprint-affecting changes; "
            "commit them or pass --allow-dirty"
        )

    prefix = tag_prefix or experiment
    image = worker_image_tag(repository, prefix, git_sha)
    primary_region = ecr_region(repository)
    push_regions = list(dict.fromkeys(regions or ([primary_region] if primary_region else [])))
    images_by_region: dict[str, str] = {}
    commands: list[list[str]] = []

    build_cmd = [
        "docker",
        "build",
        "--build-arg",
        f"STS2_GIT_SHA={git_sha}",
        "--build-arg",
        f"STS2_IMAGE_SOURCE={git_branch}",
        "-f",
        str(solver_root / "Dockerfile.worker"),
        "-t",
        image,
        str(repo_root),
    ]
    commands.append(build_cmd)
    if not dry_run:
        _run(build_cmd)

    if push:
        if not push_regions:
            push_regions = [primary_region or ""]
        for region in push_regions:
            region_repo = ecr_repository_for_region(repository, region) if region else repository
            region_image = worker_image_tag(region_repo, prefix, git_sha)
            if region and ensure_repository:
                if not dry_run:
                    ensure_ecr_repository(region_repo, region=region)
            if region and ecr_login_enabled:
                if not dry_run:
                    ecr_login(region_repo, region=region)
            if region_image != image:
                tag_cmd = ["docker", "tag", image, region_image]
                commands.append(tag_cmd)
                if not dry_run:
                    _run(tag_cmd)
            push_cmd = ["docker", "push", region_image]
            commands.append(push_cmd)
            if not dry_run:
                _run(push_cmd)
            images_by_region[region or "default"] = region_image
    else:
        if primary_region:
            images_by_region[primary_region] = image
        else:
            images_by_region["default"] = image

    plan = ImageBuildPlan(
        experiment=experiment,
        solver_root=solver_root,
        repo_root=repo_root,
        git_sha=git_sha,
        git_branch=git_branch,
        image=image,
        images_by_region=images_by_region,
        commands=commands,
        dirty_paths=dirty,
        fingerprint=fingerprint,
    )
    if not dry_run:
        record_worker_image(plan, pushed=push)
    return plan


def record_worker_image(plan: ImageBuildPlan, *, pushed: bool) -> None:
    exp = Experiment(plan.experiment)
    record = {
        "experiment": plan.experiment,
        "git_sha": plan.git_sha,
        "git_branch": plan.git_branch,
        "image": plan.image,
        "images_by_region": plan.images_by_region,
        "fingerprint": plan.fingerprint,
        "pushed": bool(pushed),
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_solver_root": str(plan.solver_root),
    }
    path = exp.dir / WORKER_IMAGE_RECORD
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")


def load_image_records(experiment: str) -> list[dict[str, Any]]:
    path = Experiment(experiment).dir / WORKER_IMAGE_RECORD
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                records.append(payload)
    return records


def latest_image_record(
    experiment: str,
    *,
    git_sha: str | None = None,
    gen: int | None = None,
) -> dict[str, Any] | None:
    if gen is not None:
        expected = generation_fingerprint(experiment, gen)
        if expected:
            git_sha = str(expected.get("git_sha") or git_sha or "")
    if not git_sha:
        git_sha = experiment_git_sha(experiment)
    for record in reversed(load_image_records(experiment)):
        if record.get("git_sha") == git_sha:
            return record
    return None


def _split_csv(values: list[str] | None) -> list[str]:
    out: list[str] = []
    for raw in values or []:
        out.extend(part.strip() for part in str(raw).split(",") if part.strip())
    return out


def load_capacity_config(path: str | Path | None) -> dict[str, Any]:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError("worker capacity config must contain a JSON object")
    return payload


def _config_regions(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    regions = config.get("regions") or {}
    if isinstance(regions, list):
        return {str(region): {} for region in regions}
    if isinstance(regions, dict):
        return {
            str(region): (settings if isinstance(settings, dict) else {})
            for region, settings in regions.items()
        }
    return {}


def _first(value: Any, default: Any = None) -> Any:
    if isinstance(value, list):
        return value[0] if value else default
    return value if value is not None else default


def _list_value(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x) for x in value if str(x)]
    return [str(value)]


def _instance_vcpus_from_name(instance_type: str) -> int:
    size = instance_type.rsplit(".", 1)[-1]
    table = {
        "nano": 1,
        "micro": 1,
        "small": 1,
        "medium": 1,
        "large": 2,
        "xlarge": 4,
        "2xlarge": 8,
        "3xlarge": 12,
        "4xlarge": 16,
        "6xlarge": 24,
        "8xlarge": 32,
        "9xlarge": 36,
        "10xlarge": 40,
        "12xlarge": 48,
        "16xlarge": 64,
        "18xlarge": 72,
        "24xlarge": 96,
        "32xlarge": 128,
        "48xlarge": 192,
        "metal": 96,
    }
    return table.get(size, 16)


def describe_instance_vcpus(instance_type: str, *, region: str) -> int:
    if os.environ.get("STS2_AWS_DESCRIBE_INSTANCE_TYPES") != "1":
        return _instance_vcpus_from_name(instance_type)
    try:
        payload = aws_json(
            ["ec2", "describe-instance-types", "--instance-types", instance_type],
            region=region,
        )
        items = payload.get("InstanceTypes") or []
        if items:
            return int(items[0]["VCpuInfo"]["DefaultVCpus"])
    except Exception:
        pass
    return _instance_vcpus_from_name(instance_type)


def _image_for_region(image: str, record: dict[str, Any] | None, region: str) -> str:
    if record:
        images_by_region = record.get("images_by_region") or {}
        if isinstance(images_by_region, dict):
            if region in images_by_region:
                return str(images_by_region[region])
            if "default" in images_by_region:
                return str(images_by_region["default"])
            if images_by_region:
                raise ValueError(
                    f"worker image for git {record.get('git_sha')} was not pushed "
                    f"for region {region}; rebuild with --region {region}"
                )
        if record.get("image"):
            image = str(record["image"])
    if ecr_parts(image.split(":", 1)[0]):
        repo, tag = image.rsplit(":", 1)
        return f"{ecr_repository_for_region(repo, region)}:{tag}"
    return image


def render_worker_user_data(
    *,
    experiment: str,
    coordinator_url: str,
    tailscale_auth_key: str,
    worker_image: str,
    worker_git_sha: str,
    aws_region: str,
    threads_per_worker: int,
    worker_count: int,
    worker_group: str = "ec2",
    lease_s: float = dist.DEFAULT_LEASE_S,
    idle_sleep_s: float = 5.0,
    template_path: Path | None = None,
) -> str:
    from .paths import SOLVER_ROOT

    template = template_path or (SOLVER_ROOT / "scripts" / "ec2_worker_cloud_init.sh")
    body = template.read_text(encoding="utf-8")
    if body.startswith("#!"):
        first, _, rest = body.partition("\n")
    else:
        first, rest = "#!/usr/bin/env bash", body
    env = {
        "TAILSCALE_AUTH_KEY": tailscale_auth_key,
        "COORDINATOR_URL": coordinator_url,
        "EXPERIMENT": experiment,
        "WORKER_GROUP": worker_group,
        "LEASE_S": str(lease_s),
        "IDLE_SLEEP_S": str(idle_sleep_s),
        "THREADS_PER_WORKER": str(int(threads_per_worker)),
        "WORKER_COUNT": str(int(worker_count)),
        "WORKER_IMAGE": worker_image,
        "WORKER_GIT_SHA": worker_git_sha,
        "AWS_REGION": aws_region,
        "ECR_LOGIN_REGISTRY": ecr_registry(worker_image.split(":", 1)[0])
        if ecr_parts(worker_image.split(":", 1)[0])
        else "",
    }
    exports = "\n".join(f"export {key}={shlex.quote(value)}" for key, value in env.items())
    return f"{first}\n{exports}\n\n{rest}"


def make_launch_plan(
    *,
    experiment: str,
    max_workers: int,
    config: dict[str, Any] | None = None,
    regions: list[str] | None = None,
    instance_types: list[str] | None = None,
    image: str | None = None,
    image_record: dict[str, Any] | None = None,
    coordinator_url: str | None = None,
    threads_per_worker: int = 1,
    worker_count: str | int = "auto",
    market: str = "spot",
) -> LaunchPlan:
    config = config or {}
    region_cfg = _config_regions(config)
    selected_regions = regions or list(region_cfg.keys()) or _split_csv([os.environ.get("STS2_EC2_REGIONS", "")])
    if not selected_regions:
        selected_regions = ["us-east-1"]

    selected_instance_types = instance_types or _list_value(config.get("instance_types")) or DEFAULT_INSTANCE_TYPES
    coordinator_url = coordinator_url or str(config.get("coordinator_url") or os.environ.get("STS2_COORDINATOR_URL") or "")
    if not coordinator_url:
        raise ValueError("coordinator URL is required")
    if not image:
        if image_record and image_record.get("image"):
            image = str(image_record["image"])
        else:
            raise ValueError("worker image is required; build one or pass --image")

    max_workers = int(max_workers)
    if max_workers < 1:
        raise ValueError("max_workers must be >= 1")
    threads_per_worker = max(1, int(threads_per_worker))
    explicit_worker_count = None if str(worker_count).lower() == "auto" else max(1, int(worker_count))

    units: list[LaunchUnit] = []
    remaining = max_workers
    index = 0
    combos = [(r, t) for r in selected_regions for t in selected_instance_types]
    if not combos:
        raise ValueError("at least one region and instance type is required")

    while remaining > 0:
        region, instance_type = combos[index % len(combos)]
        per_region = region_cfg.get(region, {})
        ami = str(per_region.get("ami") or config.get("ami") or os.environ.get("STS2_EC2_AMI") or "")
        if not ami:
            raise ValueError(f"AMI is required for region {region}")
        subnets = _list_value(per_region.get("subnet_ids") or per_region.get("subnet_id") or config.get("subnet_ids") or config.get("subnet_id") or os.environ.get("STS2_EC2_SUBNET_ID"))
        security_groups = _list_value(per_region.get("security_group_ids") or per_region.get("security_group_id") or config.get("security_group_ids") or config.get("security_group_id") or os.environ.get("STS2_EC2_SECURITY_GROUP_ID"))
        if not security_groups:
            raise ValueError(f"security group is required for region {region}")
        vcpus = describe_instance_vcpus(instance_type, region=region)
        auto_workers = max(1, vcpus // threads_per_worker)
        workers = explicit_worker_count or auto_workers
        workers = min(workers, remaining)
        subnet = subnets[index % len(subnets)] if subnets else None
        units.append(
            LaunchUnit(
                region=region,
                instance_type=instance_type,
                workers=workers,
                threads_per_worker=threads_per_worker,
                image=_image_for_region(image, image_record, region),
                subnet_id=subnet,
                security_group_ids=security_groups,
                ami=ami,
                iam_instance_profile=_first(per_region.get("iam_instance_profile") or config.get("iam_instance_profile") or os.environ.get("STS2_EC2_IAM_INSTANCE_PROFILE")),
                key_name=_first(per_region.get("key_name") or config.get("key_name") or os.environ.get("STS2_EC2_KEY_NAME")),
            )
        )
        remaining -= workers
        index += 1

    return LaunchPlan(
        experiment=experiment,
        max_workers=max_workers,
        git_sha=str((image_record or {}).get("git_sha") or experiment_git_sha(experiment)),
        coordinator_url=coordinator_url,
        market=market,
        units=units,
    )


def launch_ec2_workers(
    plan: LaunchPlan,
    *,
    tailscale_auth_key: str,
    worker_group: str = "ec2",
    lease_s: float = dist.DEFAULT_LEASE_S,
    idle_sleep_s: float = 5.0,
    dry_run: bool = False,
) -> list[dict[str, Any]]:
    responses: list[dict[str, Any]] = []
    for idx, unit in enumerate(plan.units):
        user_data = render_worker_user_data(
            experiment=plan.experiment,
            coordinator_url=plan.coordinator_url,
            tailscale_auth_key=tailscale_auth_key,
            worker_image=unit.image,
            worker_git_sha=plan.git_sha,
            aws_region=unit.region,
            threads_per_worker=unit.threads_per_worker,
            worker_count=unit.workers,
            worker_group=worker_group,
            lease_s=lease_s,
            idle_sleep_s=idle_sleep_s,
        )
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".sh", delete=False) as f:
            f.write(user_data)
            user_data_path = Path(f.name)
        cmd = [
            _aws_cli(),
            "--region",
            unit.region,
            "ec2",
            "run-instances",
            "--image-id",
            unit.ami,
            "--instance-type",
            unit.instance_type,
            "--count",
            "1",
            "--user-data",
            f"file://{user_data_path}",
            "--tag-specifications",
            (
                "ResourceType=instance,Tags=["
                f"{{Key=Name,Value=sts2-{plan.experiment}-worker}},"
                f"{{Key=STS2Experiment,Value={plan.experiment}}},"
                f"{{Key=STS2WorkerGroup,Value={worker_group}}},"
                f"{{Key=STS2GitSha,Value={plan.git_sha}}}"
                "]"
            ),
        ]
        if plan.market == "spot":
            cmd.extend([
                "--instance-market-options",
                "MarketType=spot,SpotOptions={SpotInstanceType=one-time,InstanceInterruptionBehavior=terminate}",
            ])
        if unit.subnet_id:
            cmd.extend(["--subnet-id", unit.subnet_id])
        if unit.security_group_ids:
            cmd.extend(["--security-group-ids", *unit.security_group_ids])
        if unit.iam_instance_profile:
            profile = unit.iam_instance_profile
            if profile.startswith("arn:"):
                cmd.extend(["--iam-instance-profile", f"Arn={profile}"])
            else:
                cmd.extend(["--iam-instance-profile", f"Name={profile}"])
        if unit.key_name:
            cmd.extend(["--key-name", unit.key_name])
        if dry_run:
            responses.append({"dry_run": True, "command": cmd, "workers": unit.workers})
        else:
            result = _run(cmd, capture=True)
            payload = json.loads(result.stdout) if result.stdout.strip() else {}
            payload["_planned_workers"] = unit.workers
            payload["_region"] = unit.region
            payload["_instance_type"] = unit.instance_type
            responses.append(payload)
        try:
            user_data_path.unlink()
        except OSError:
            pass
    return responses


def coordinator_record_path(experiment: str) -> Path:
    return Experiment(experiment).dir / WORKER_COORDINATOR_RECORD


def coordinator_health(url: str, *, timeout: float = 3.0) -> dict[str, Any]:
    req = urllib.request.Request(url.rstrip("/") + "/api/health", headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def start_coordinator(
    *,
    experiment: str,
    host: str = "0.0.0.0",
    port: int = 8765,
    static: str | None = None,
) -> dict[str, Any]:
    exp = Experiment(experiment)
    log_dir = exp.dir / "logs"
    log_dir.mkdir(exist_ok=True)
    log_path = log_dir / f"companion-api-{port}.log"
    err_path = log_dir / f"companion-api-{port}.err.log"
    cmd = [sys.executable, "-m", "sts2_solver.companion", "--host", host, "--port", str(port)]
    if static:
        cmd.extend(["--static", static])
    out = open(log_path, "a", encoding="utf-8")
    err = open(err_path, "a", encoding="utf-8")
    kwargs: dict[str, Any] = {
        "cwd": str(experiment_solver_root(experiment)),
        "stdout": out,
        "stderr": err,
    }
    if os.name == "nt":
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
    proc = subprocess.Popen(cmd, **kwargs)
    record = {
        "pid": proc.pid,
        "host": host,
        "port": int(port),
        "url": f"http://127.0.0.1:{port}",
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "log": str(log_path),
        "err_log": str(err_path),
    }
    dist.atomic_write_json(coordinator_record_path(experiment), record)
    return record


def stop_coordinator(experiment: str) -> dict[str, Any]:
    record = dist.read_json(coordinator_record_path(experiment))
    pid = record.get("pid")
    if not pid:
        raise ValueError(f"no coordinator record for {experiment}")
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(int(pid)), "/T", "/F"], capture_output=True)
    else:
        os.kill(int(pid), 15)
    record["stopped_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    dist.atomic_write_json(coordinator_record_path(experiment), record)
    return record


def coordinator_status(experiment: str, url: str | None = None) -> dict[str, Any]:
    record = dist.read_json(coordinator_record_path(experiment))
    target = url or record.get("url") or "http://127.0.0.1:8765"
    try:
        health = coordinator_health(str(target))
        return {"url": target, "healthy": True, "health": health, "record": record}
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return {"url": target, "healthy": False, "error": str(exc), "record": record}
