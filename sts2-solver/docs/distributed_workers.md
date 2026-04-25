# Distributed Workers

The trainer/coordinator runs on the local machine and exposes the companion API.
Workers are CPU-bound self-play shards that poll the coordinator, run shards
against the leased checkpoint, and upload compressed shard results. Workers can
run on the same host as the coordinator, on EC2, or on both at once.

All worker hosts run the same `sts2-solver/Dockerfile.worker` image driven by
`sts2-solver/scripts/docker_worker_entrypoint.py`, so the only thing that varies
between deployments is the launcher.

## Image

```bash
docker build -f sts2-solver/Dockerfile.worker -t sts2-worker:latest .
```

The Dockerfile bundles the Python solver and the Rust `sts2_engine` extension,
so the image is the same artifact regardless of where it runs. Each container
is stamped with `STS2_GIT_SHA` and tagged with `STS2_WORKER_GROUP` (e.g.
`local`, `ec2`) so companion metrics can be sliced by source.

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

Launch path:

1. Copy `sts2-solver/scripts/ec2_worker_cloud_init.sh`.
2. Fill in `TAILSCALE_AUTH_KEY`.
3. Set `COORDINATOR_URL`, `EXPERIMENT`, and `BRANCH` if the defaults are not the
   desired target.
4. Paste the edited script as EC2 user data.

The script installs Docker and Tailscale, joins the tailnet, clones the repo,
builds `sts2-solver/Dockerfile.worker`, and launches one or more worker
containers tagged `STS2_WORKER_GROUP=ec2`.

Container logs are capped at three 50 MB json-file logs per worker.

## Sizing

`THREADS_PER_WORKER` defaults to `8`, which matches the current shard size well.
`WORKER_COUNT=auto` starts `floor(vCPU / THREADS_PER_WORKER)` containers. For a
16-vCPU host this starts two containers at eight Rayon threads each. If the
host has fewer vCPUs than `THREADS_PER_WORKER`, the launchers clamp
`THREADS_PER_WORKER` down to the host vCPU count.

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
