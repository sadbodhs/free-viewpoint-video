#!/usr/bin/env bash
# Build the fvv image and record the resolved package versions.
set -euo pipefail
cd "$(dirname "$0")/.."
docker build -f docker/Dockerfile -t "${FVV_IMAGE:-fvv:latest}" .
docker run --rm "${FVV_IMAGE:-fvv:latest}" uv pip freeze > docker/requirements.lock
echo "wrote docker/requirements.lock"
