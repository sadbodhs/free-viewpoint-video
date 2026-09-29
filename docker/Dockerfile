# Free-viewpoint video environment (RTX 3090 / sm_86).
# devel image so CUDA extensions (gsplat, custom kernels) can compile later.
FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    TORCH_CUDA_ARCH_LIST="8.6" \
    UV_LINK_MODE=copy \
    UV_NO_CACHE=1 \
    UV_PYTHON_INSTALL_DIR=/opt/python \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH

RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg git curl ca-certificates build-essential ninja-build \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.8 /uv /usr/local/bin/uv
RUN uv venv --python 3.11 /opt/venv && chmod -R a+rX /opt/python /opt/venv

RUN uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
COPY docker/requirements.txt /tmp/requirements.txt
RUN uv pip install -r /tmp/requirements.txt

WORKDIR /workspace
