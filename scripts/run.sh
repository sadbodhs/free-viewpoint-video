#!/usr/bin/env bash
# Run a command inside the fvv container with the project mounted at /workspace.
#   scripts/run.sh python scripts/download_panoptic.py ...
set -euo pipefail
cd "$(dirname "$0")/.."

TTY=()
[ -t 0 ] && [ -t 1 ] && TTY=(-it)

mkdir -p .cache/home .cache/torch
exec docker run --rm "${TTY[@]}" --gpus all --ipc=host \
    --user "$(id -u):$(id -g)" \
    -e HOME=/workspace/.cache/home \
    -e USER="$(id -un)" \
    -e TORCH_HOME=/workspace/.cache/torch \
    -e PYTHONPATH=/workspace \
    -v "$PWD":/workspace -w /workspace \
    "${FVV_IMAGE:-fvv:latest}" "$@"
