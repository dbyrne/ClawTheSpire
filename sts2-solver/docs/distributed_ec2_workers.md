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

1. Copy `sts2-solver/scripts/ec2_worker_cloud_init.sh`.
2. Fill in `TAILSCALE_AUTH_KEY`.
3. Set `COORDINATOR_URL`, `EXPERIMENT`, and `BRANCH` if the defaults are not the
   desired target.
4. Paste the edited script as EC2 user data.

The script installs Docker and Tailscale, joins the tailnet, clones the repo,
builds `sts2-solver/Dockerfile.worker`, and launches one or more worker
containers.

## Sizing

`THREADS_PER_WORKER` defaults to `8`, which matches the current shard size well.
`WORKER_COUNT=auto` starts `floor(vCPU / THREADS_PER_WORKER)` containers. For a
16-vCPU instance this starts two containers at eight Rayon threads each.

Tune the instance shape by watching the companion app worker metrics:

- `cpu_pct` should be high once workers are past their first heartbeat.
- `load_per_cpu` near `1.0` means the instance is saturated.
- If `cpu_pct` is low and shards are available, add containers or reduce
  `THREADS_PER_WORKER`.
- If `load_per_cpu` is well above `1.0`, reduce containers or increase instance
  size.

## Useful Commands

```bash
docker ps --filter "name=sts2-worker"
docker logs -f sts2-worker-1
docker restart sts2-worker-1
docker rm -f sts2-worker-1
```

To change worker count on a running instance, rerun the `docker run` loop from
the cloud-init script with the new `WORKER_COUNT` / `THREADS_PER_WORKER` values.
