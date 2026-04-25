#!/usr/bin/env bash
set -euo pipefail

# Edit these values before pasting this file as EC2 user data.
TAILSCALE_AUTH_KEY="${TAILSCALE_AUTH_KEY:-}"
COORDINATOR_URL="${COORDINATOR_URL:-http://100.100.101.1:8765}"
EXPERIMENT="${EXPERIMENT:-encoder-v2-cpuct3-dist-pilot}"
REPO_URL="${REPO_URL:-https://github.com/dbyrne/ClawTheSpire.git}"
BRANCH="${BRANCH:-main}"
WORKER_GROUP="${WORKER_GROUP:-ec2}"
LEASE_S="${LEASE_S:-240}"
IDLE_SLEEP_S="${IDLE_SLEEP_S:-5}"
THREADS_PER_WORKER="${THREADS_PER_WORKER:-8}"
WORKER_COUNT="${WORKER_COUNT:-auto}"
WORKER_IMAGE="${WORKER_IMAGE:-}"
WORKER_GIT_SHA="${WORKER_GIT_SHA:-}"
AWS_REGION="${AWS_REGION:-}"
ECR_LOGIN_REGISTRY="${ECR_LOGIN_REGISTRY:-}"

if [[ -z "$TAILSCALE_AUTH_KEY" ]]; then
  echo "TAILSCALE_AUTH_KEY is required" >&2
  exit 1
fi

export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-install-recommends ca-certificates curl docker.io git unzip
systemctl enable --now docker

install_aws_cli() {
  if command -v aws >/dev/null 2>&1; then
    return
  fi
  tmpdir="$(mktemp -d)"
  curl -fsSL "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" \
    -o "${tmpdir}/awscliv2.zip"
  unzip -q "${tmpdir}/awscliv2.zip" -d "$tmpdir"
  "${tmpdir}/aws/install" --update
  rm -rf "$tmpdir"
}

if ! command -v tailscale >/dev/null 2>&1; then
  curl -fsSL https://tailscale.com/install.sh | sh
fi

metadata_token="$(curl -fsS -m 2 -X PUT \
  "http://169.254.169.254/latest/api/token" \
  -H "X-aws-ec2-metadata-token-ttl-seconds: 21600" || true)"

metadata() {
  local path="$1"
  if [[ -n "$metadata_token" ]]; then
    curl -fsS -m 2 -H "X-aws-ec2-metadata-token: $metadata_token" \
      "http://169.254.169.254/latest/meta-data/${path}" || true
  else
    curl -fsS -m 2 "http://169.254.169.254/latest/meta-data/${path}" || true
  fi
}

INSTANCE_ID="$(metadata instance-id)"
INSTANCE_TYPE="$(metadata instance-type)"
if [[ -z "$AWS_REGION" ]]; then
  AWS_REGION="$(metadata placement/region)"
fi
INSTANCE_ID="${INSTANCE_ID:-$(hostname)}"
INSTANCE_TYPE="${INSTANCE_TYPE:-unknown}"
AWS_REGION="${AWS_REGION:-us-east-1}"

tailscale up \
  --auth-key="$TAILSCALE_AUTH_KEY" \
  --hostname="sts2-${INSTANCE_ID}" \
  --accept-routes=false

if [[ -n "$WORKER_IMAGE" ]]; then
  if [[ "$WORKER_IMAGE" == *".dkr.ecr."*".amazonaws.com"* ]]; then
    install_aws_cli
    registry="${ECR_LOGIN_REGISTRY:-${WORKER_IMAGE%%/*}}"
    aws ecr get-login-password --region "$AWS_REGION" \
      | docker login --username AWS --password-stdin "$registry"
  fi
  docker pull "$WORKER_IMAGE"
  docker tag "$WORKER_IMAGE" sts2-worker:latest
  GIT_SHA="${WORKER_GIT_SHA:-$(docker image inspect "$WORKER_IMAGE" \
    --format '{{ index .Config.Labels "org.opencontainers.image.revision" }}' 2>/dev/null || true)}"
  if [[ -z "$GIT_SHA" || "$GIT_SHA" == "<no value>" || "$GIT_SHA" == "unknown" ]]; then
    echo "WORKER_GIT_SHA is required when WORKER_IMAGE has no revision label" >&2
    exit 1
  fi
else
  rm -rf /opt/sts2
  git clone --depth 1 --branch "$BRANCH" "$REPO_URL" /opt/sts2
  cd /opt/sts2
  GIT_SHA="$(git rev-parse HEAD)"

  docker build \
    --build-arg STS2_GIT_SHA="$GIT_SHA" \
    --build-arg STS2_IMAGE_SOURCE="${REPO_URL}#${BRANCH}" \
    -f sts2-solver/Dockerfile.worker \
    -t sts2-worker:latest \
    .
fi

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

mkdir -p /var/cache/sts2-worker

for i in $(seq 1 "$WORKER_COUNT"); do
  name="sts2-worker-${i}"
  docker rm -f "$name" >/dev/null 2>&1 || true
  docker run -d \
    --restart unless-stopped \
    --name "$name" \
    --network host \
    --cpus="$THREADS_PER_WORKER" \
    --log-opt max-size=50m \
    --log-opt max-file=3 \
    -v "/var/cache/sts2-worker:/cache" \
    -e COORDINATOR_URL="$COORDINATOR_URL" \
    -e EXPERIMENT="$EXPERIMENT" \
    -e WORKER_ID="${INSTANCE_ID}-w${i}" \
    -e CACHE_DIR="/cache" \
    -e LEASE_S="$LEASE_S" \
    -e IDLE_SLEEP_S="$IDLE_SLEEP_S" \
    -e RAYON_NUM_THREADS="$THREADS_PER_WORKER" \
    -e AWS_INSTANCE_ID="$INSTANCE_ID" \
    -e AWS_INSTANCE_TYPE="$INSTANCE_TYPE" \
    -e STS2_WORKER_GROUP="$WORKER_GROUP" \
    -e STS2_GIT_SHA="$GIT_SHA" \
    sts2-worker:latest
done

docker ps --filter "name=sts2-worker"
