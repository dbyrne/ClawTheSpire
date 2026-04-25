# Distributed Workers

The trainer/coordinator runs on the local machine and exposes the companion API.
Workers are CPU-bound self-play shards that poll the coordinator, run shards
against the leased checkpoint, and upload compressed shard results. Workers can
run on the same host as the coordinator, on EC2, or on both at once.

All worker hosts run the same `sts2-solver/Dockerfile.worker` image driven by
`sts2-solver/scripts/docker_worker_entrypoint.py`, so the only thing that varies
between deployments is the launcher.

## Experiment CLI Control Plane

`sts2-experiment` is the preferred control plane for cloud workers. It knows
which worktree owns an experiment, records images against the experiment, and
can plan EC2 capacity from the same fingerprint that distributed workers must
send to the coordinator.

Start or check the coordinator:

```powershell
sts2-experiment coordinator start encoder-v2-cpuct3-dist-pilot --port 8765
sts2-experiment coordinator status encoder-v2-cpuct3-dist-pilot
```

Build and push one immutable image for the current experiment commit:

```powershell
sts2-experiment worker-image build encoder-v2-cpuct3-dist-pilot `
  --repository 700694289572.dkr.ecr.us-east-1.amazonaws.com/sts2-worker `
  --region us-east-1 `
  --region us-west-2 `
  --ensure-repository `
  --ecr-login `
  --push
```

If resuming an already scheduled generation, add `--gen N`. The CLI refuses to
build from the current worktree if that generation was scheduled with a
different worker fingerprint.

Plan or launch EC2 workers from the latest recorded compatible image:

```powershell
sts2-experiment workers plan encoder-v2-cpuct3-dist-pilot `
  --max-workers 96 `
  --config .\worker-capacity.json `
  --coordinator http://100.100.101.1:8765 `
  --region us-east-1 `
  --region us-west-2 `
  --instance-type c7i.4xlarge `
  --instance-type c7a.4xlarge

sts2-experiment workers launch encoder-v2-cpuct3-dist-pilot `
  --max-workers 96 `
  --config .\worker-capacity.json `
  --coordinator http://100.100.101.1:8765 `
  --region us-east-1 `
  --region us-west-2
```

`--max-workers` caps worker containers, not instances. For shard-size-1 runs,
keep `--threads-per-worker 1 --worker-count auto`; the planner will set the
last instance's `WORKER_COUNT` lower when needed so the launch does not exceed
the requested worker count.

Example capacity config:

```json
{
  "coordinator_url": "http://100.100.101.1:8765",
  "ami": "ami-05cf1e9f73fbad2e2",
  "security_group_id": "sg-0ab2f9f94bdd40c0d",
  "iam_instance_profile": "sts2-ec2-worker-profile",
  "instance_types": ["c7i.4xlarge", "c7a.4xlarge", "m7i.4xlarge"],
  "regions": {
    "us-east-1": {
      "subnet_ids": ["subnet-0020689fdecb2b4f8"]
    },
    "us-west-2": {
      "ami": "ami-for-us-west-2",
      "subnet_ids": ["subnet-for-us-west-2"],
      "security_group_id": "sg-for-us-west-2"
    }
  }
}
```

## Manual Image

```bash
docker build -f sts2-solver/Dockerfile.worker -t sts2-worker:latest .
```

The Dockerfile bundles the Python solver and the Rust `sts2_engine` extension,
so the image is the same artifact regardless of where it runs. Each container
is stamped with `STS2_GIT_SHA` and tagged with `STS2_WORKER_GROUP` (e.g.
`local`, `ec2`) so companion metrics can be sliced by source.

For EC2 Spot capacity, prefer prebuilding one image per experiment commit with
the CLI and pulling it on boot. That avoids spending several minutes compiling
Rust on instances that may be reclaimed. The lower-level PowerShell helper is
still available when you need to build outside the experiment CLI:

```powershell
.\sts2-solver\scripts\build_worker_image.ps1 `
  -ImageRepository 700694289572.dkr.ecr.us-east-1.amazonaws.com/sts2-worker `
  -TagPrefix encoder-v2-cpuct3-dist-pilot `
  -EcrLogin `
  -Push
```

The image tag includes the full git SHA, and the image is labeled with the same
SHA. EC2 workers use that label as `STS2_GIT_SHA`, so the distributed
fingerprint check still rejects workers built from the wrong commit.

## Local Workers

Run alongside the coordinator on the same machine. The launcher is
`sts2-solver/scripts/run_local_workers.sh`, which mirrors the EC2 cloud-init
loop without the Tailscale / IMDS / git-clone steps.

```bash
./sts2-solver/scripts/run_local_workers.sh                    # auto vCPU count
WORKER_COUNT=2 THREADS_PER_WORKER=6 ./sts2-solver/scripts/run_local_workers.sh
BUILD=force ./sts2-solver/scripts/run_local_workers.sh        # after code changes
```

Defaults that differ from the EC2 path:

- `COORDINATOR_URL=http://localhost:8765` (uses `--network host`).
- `STS2_WORKER_GROUP=local`.
- `CACHE_DIR=${HOME}/.cache/sts2-worker`, host-mounted so checkpoints survive
  restarts and are shared across local containers.
- `NAME_PREFIX=sts2-worker-local`, distinct from the EC2 `sts2-worker-N` names.
- `BUILD=auto` builds the image only if `sts2-worker:latest` is missing. Use
  `BUILD=force` after code changes, `BUILD=skip` to reuse whatever is present.

## EC2 Workers

EC2 instances connect to the companion API over Tailscale and run the same
image. Prerequisites:

- Companion API reachable from Tailscale, for example `http://100.100.101.1:8765`.
- A reusable or ephemeral Tailscale auth key.
- An Ubuntu EC2 AMI with outbound internet access. No inbound security-group
  rule is required for the worker path.

Manual launch path:

1. Copy `sts2-solver/scripts/ec2_worker_cloud_init.sh`.
2. Fill in `TAILSCALE_AUTH_KEY`.
3. Set `COORDINATOR_URL`, `EXPERIMENT`, and `BRANCH` if the defaults are not the
   desired target.
4. For a prebuilt image, set `WORKER_IMAGE` to the pushed image URI. If the image
   is in private ECR, the script logs in with the instance profile before
   pulling. Set `AWS_REGION` if the instance metadata region is not the ECR
   region.
5. Paste the edited script as EC2 user data.

The script installs Docker and Tailscale, joins the tailnet, clones the repo,
then either pulls `WORKER_IMAGE` or builds `sts2-solver/Dockerfile.worker`, and
launches one or more worker containers tagged `STS2_WORKER_GROUP=ec2`. The
experiment CLI generates the same user-data script automatically when using
`sts2-experiment workers launch`.

Container logs are capped at three 50 MB json-file logs per worker.

## Sizing

`THREADS_PER_WORKER` defaults to `8`, which matches larger shard sizes well.
For shard-size-1 distributed runs, use `THREADS_PER_WORKER=1` and
`WORKER_COUNT=auto`, which starts one container per vCPU. If the host has fewer
vCPUs than `THREADS_PER_WORKER`, the launchers clamp `THREADS_PER_WORKER` down
to the host vCPU count.

Tune the host shape by watching the companion app worker metrics:

- `cpu_pct` should be high once workers are past their first heartbeat.
- `load_per_cpu` near `1.0` means the host is saturated.
- If `cpu_pct` is low and shards are available, add containers or reduce
  `THREADS_PER_WORKER`.
- If `load_per_cpu` is well above `1.0`, reduce containers or increase host
  size.

CPU percent, load, load per CPU, and RSS metrics are emitted by Linux workers
(both local and EC2). Windows laptop workers still report identity metadata,
but those utilization fields are currently blank.

## Useful Commands

```bash
docker ps --filter "name=sts2-worker"            # all workers on this host
docker ps --filter "name=sts2-worker-local"      # local-only
docker logs -f sts2-worker-local-1
docker restart sts2-worker-local-1
docker rm -f sts2-worker-local-1
```

To change worker count on a running host, rerun the launcher with the new
`WORKER_COUNT` / `THREADS_PER_WORKER` values. The launchers `docker rm -f` any
existing container with the same name before starting, so reruns are
idempotent.
