#!/usr/bin/env bash
# Start the local live viewer in the fvv container, published on the host's port (default 8080).
#   scripts/serve.sh data/panoptic/170221_haggling_b1
# Stop with: docker stop fvv-serve
set -euo pipefail
cd "$(dirname "$0")/.."
PORT="${FVV_PORT:-8080}"
mkdir -p .cache/home .cache/torch
docker rm -f fvv-serve >/dev/null 2>&1 || true
exec docker run -d --name fvv-serve --gpus all --ipc=host \
    --user "$(id -u):$(id -g)" \
    -e HOME=/workspace/.cache/home -e USER="$(id -un)" \
    -e TORCH_HOME=/workspace/.cache/torch -e PYTHONPATH=/workspace -e PYTHONUNBUFFERED=1 \
    -p "${PORT}:${PORT}" \
    -v "$PWD":/workspace -w /workspace \
    "${FVV_IMAGE:-fvv:latest}" python scripts/serve.py "$@" --port "$PORT"
