#!/usr/bin/env bash
# Run scripts/gate.sh in python:3.11-slim. This is the laptop route: pyenv cannot
# build 3.11 against this machine's OpenSSL 4.1.
#
# Usage: bash scripts/gate-docker.sh
#
# The repository is mounted read-only, so a passing run proves the gate makes no
# changes to tracked files. Two writable places are added instead:
#   * a tmpfs over .runtime, where prepare_source.build_runtime() writes the
#     runnable copy of the DRAG sources;
#   * a named volume for the pip cache, so requirements reinstall quickly.
# Dependencies go into /tmp/venv inside the container, never into the repository.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

VOLUME="${GATE_PIP_VOLUME:-drag-gate-pip-cache}"

if ! command -v docker >/dev/null 2>&1; then
  echo "GATE FAILED: docker is not installed; use scripts/gate.sh on a Python 3.11 host" >&2
  exit 1
fi

# A git worktree keeps its metadata outside the repository, so mount the real
# directories read-only at their own absolute paths; otherwise git cannot resolve
# the guards' comparison base.
GIT_DIR="$(git rev-parse --absolute-git-dir)"
GIT_COMMON="$(cd "$(git rev-parse --git-common-dir)" && pwd -P)"

# The tmpfs below is mounted over .runtime, which must exist on the host first:
# a fresh clone has no .runtime, and Docker's read-only bind refuses to create a
# mount point inside it. Remove the directory again if this script created it.
RUNTIME="$ROOT/.runtime"
CREATED_RUNTIME=0
if [ ! -d "$RUNTIME" ]; then
  mkdir -p "$RUNTIME"
  CREATED_RUNTIME=1
fi

cleanup() {
  if [ "$CREATED_RUNTIME" = 1 ]; then
    rmdir "$RUNTIME" 2>/dev/null || true
  fi
}
trap cleanup EXIT

echo "==> install git in a throwaway image layer"
# --network host is required here too: bridge networking is broken on the laptop
# (veth pairs cannot be created) until a reboot, so the default build network fails.
docker build --network host -q -t drag-gate:3.11 - <<'DOCKERFILE' >/dev/null
FROM python:3.11-slim
RUN apt-get update -qq && apt-get install -y -qq --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*
DOCKERFILE

echo "==> gate (repo mounted read-only)"
docker run --rm \
  --network host \
  -u "$(id -u):$(id -g)" \
  -e HOME=/tmp \
  -e PYTHONDONTWRITEBYTECODE=1 \
  -e GIT_OPTIONAL_LOCKS=0 \
  -v "$ROOT":/work:ro \
  -v "$GIT_DIR":"$GIT_DIR":ro \
  -v "$GIT_COMMON":"$GIT_COMMON":ro \
  -v "$VOLUME":/tmp/pip-cache \
  --tmpfs "/work/.runtime:rw,exec,size=256m,uid=$(id -u),gid=$(id -g),mode=0755" \
  -w /work \
  drag-gate:3.11 \
  bash -c '
    set -euo pipefail
    export PIP_CACHE_DIR=/tmp/pip-cache
    python -m venv /tmp/venv
    /tmp/venv/bin/pip install -q -r requirements-ci.txt
    export PATH=/tmp/venv/bin:$PATH
    exec bash scripts/gate.sh
  '
