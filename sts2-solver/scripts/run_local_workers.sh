#!/usr/bin/env bash
set -euo pipefail

# Launch one or more local distributed self-play workers in Docker, mirroring
# the EC2 cloud-init layout so local and cloud workers stay in lockstep.

COORDINATOR_URL="${COORDINATOR_URL:-http://localhost:8765}"
EXPERIMENT="${EXPERIMENT:-encoder-v2-cpuct3-dist-pilot}"
WORKER_GROUP="${WORKER_GROUP:-local}"
LEASE_S="${LEASE_S:-240}"
IDLE_SLEEP_S="${IDLE_SLEEP_S:-5}"
THREADS_PER_WORKER="${THREADS_PER_WORKER:-8}"
WORKER_COUNT="${WORKER_COUNT:-auto}"
CACHE_DIR="${CACHE_DIR:-${HOME}/.cache/sts2-worker}"
IMAGE_TAG="${IMAGE_TAG:-sts2-worker:latest}"
BUILD="${BUILD:-auto}"
NAME_PREFIX="${NAME_PREFIX:-sts2-worker-local}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

GIT_SHA="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
HOST_ID="$(hostname -s 2>/dev/null || hostname)"

case "$BUILD" in
  force)
    docker build -f sts2-solver/Dockerfile.worker -t "$IMAGE_TAG" .
    ;;
  auto)
    if ! docker image inspect "$IMAGE_TAG" >/dev/null 2>&1; then
      docker build -f sts2-solver/Dockerfile.worker -t "$IMAGE_TAG" .
    fi
    ;;
  skip|0|"")
    ;;
  *)
    echo "Unknown BUILD mode: $BUILD (expected force|auto|skip)" >&2
    exit 1
    ;;
esac

vcpus="$(nproc)"
if (( THREADS_PER_WORKER < 1 )); then
  THREADS_PER_WORKER=1
fi
if (( THREADS_PER_WORKER > vcpus )); then
  THREADS_PER_WORKER="$vcpus"
fi
if [[ "$WORKER_COUNT" == "auto" ]]; then
  WORKER_COUNT="$(( vcpus / THREADS_PER_WORKER ))"
  if (( WORKER_COUNT < 1 )); then
    WORKER_COUNT=1
  fi
fi

mkdir -p "$CACHE_DIR"

for i in $(seq 1 "$WORKER_COUNT"); do
  name="${NAME_PREFIX}-${i}"
  docker rm -f "$name" >/dev/null 2>&1 || true
  docker run -d \
    --restart unless-stopped \
    --name "$name" \
    --network host \
    --cpus="$THREADS_PER_WORKER" \
    --log-opt max-size=50m \
    --log-opt max-file=3 \
    -v "${CACHE_DIR}:/cache" \
    -e COORDINATOR_URL="$COORDINATOR_URL" \
    -e EXPERIMENT="$EXPERIMENT" \
    -e WORKER_ID="${HOST_ID}-w${i}" \
    -e CACHE_DIR="/cache" \
    -e LEASE_S="$LEASE_S" \
    -e IDLE_SLEEP_S="$IDLE_SLEEP_S" \
    -e RAYON_NUM_THREADS="$THREADS_PER_WORKER" \
    -e STS2_WORKER_GROUP="$WORKER_GROUP" \
    -e STS2_GIT_SHA="$GIT_SHA" \
    "$IMAGE_TAG"
done

docker ps --filter "name=${NAME_PREFIX}"
