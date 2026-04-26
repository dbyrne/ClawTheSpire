# Distributed EC2 Workers

The trainer/coordinator still runs on the local machine. EC2 instances only run
CPU-bound self-play shards, connect to the companion API over Tailscale, and
upload compressed shard results.

## Prerequisites

- Companion API reachable from Tailscale, for example `http://100.100.101.1:8765`.
- A reusable or ephemeral Tailscale auth key.
- An Ubuntu EC2 AMI with outbound internet access. No inbound security-group rule
  is required for the worker path.

## Launch Path

For Spot workers, prefer prebuilding one image per experiment commit and
pulling it on boot. That avoids spending several minutes compiling Rust on
instances that may be reclaimed.

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
launches one or more worker containers.

Container logs are capped at three 50 MB json-file logs per worker.

## Sizing

`THREADS_PER_WORKER` defaults to `8`, which matches larger shard sizes well.
For shard-size-1 distributed runs, use `THREADS_PER_WORKER=1` and
`WORKER_COUNT=auto`, which starts one container per vCPU. If the instance has
fewer vCPUs than `THREADS_PER_WORKER`, the bootstrap clamps `THREADS_PER_WORKER`
down to the instance vCPU count.

Tune the instance shape by watching the companion app worker metrics:

- `cpu_pct` should be high once workers are past their first heartbeat.
- `load_per_cpu` near `1.0` means the instance is saturated.
- If `cpu_pct` is low and shards are available, add containers or reduce
  `THREADS_PER_WORKER`.
- If `load_per_cpu` is well above `1.0`, reduce containers or increase instance
  size.

CPU percent, load, load per CPU, and RSS metrics are emitted by Linux workers.
Windows laptop workers still report identity metadata, but those utilization
fields are currently blank.

## Useful Commands

```bash
docker ps --filter "name=sts2-worker"
docker logs -f sts2-worker-1
docker restart sts2-worker-1
docker rm -f sts2-worker-1
```

To change worker count on a running instance, rerun the `docker run` loop from
the cloud-init script with the new `WORKER_COUNT` / `THREADS_PER_WORKER` values.
