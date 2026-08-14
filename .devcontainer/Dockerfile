# syntax=docker/dockerfile:1
# Dev container for Muse Glimmer 30B long-context work (training/eval side).
#
# Base: NVIDIA NGC PyTorch container. Verified on DGX Spark (GB10, aarch64, sm_121) with
# BOTH 26.07-py3 (torch 2.13.0a0+nv26.07, CUDA 13.3 — default) and 25.10-py3
# (torch 2.9.0a0+nv25.10, CUDA 13.0). Both are multi-arch (aarch64 + x86_64), so the
# same image also runs on the RTX 5090 deployment box.
#
# Build args:
#   NGC_TAG  — NGC pytorch container tag (default: newest verified in this repo)
ARG NGC_TAG=26.07-py3
FROM nvcr.io/nvidia/pytorch:${NGC_TAG}

# --- uv for fast, reproducible dependency installation -------------------------
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
ENV UV_LINK_MODE=copy \
    UV_BREAK_SYSTEM_PACKAGES=1 \
    PYTHONUNBUFFERED=1

# --- project dependencies -------------------------------------------------------
# torch comes from the base image (NGC build with GB10/Blackwell kernels + flash-attn).
# `--excludes torch` removes torch from resolution entirely so the NGC build is never
# replaced. (Alternatives that do NOT work: an exact `torch==<ngc-local-version>`
# constraint — uv finds no index candidate and resolution fails on 26.07; and
# `--no-install-package` — not available on `uv pip install` in 0.12.x. Both verified.)
COPY pyproject.toml /tmp/pyproject.toml
RUN printf 'torch\n' > /tmp/excludes.txt \
    && uv pip install --system --break-system-packages --excludes /tmp/excludes.txt -r /tmp/pyproject.toml \
    && rm /tmp/pyproject.toml /tmp/excludes.txt

# --- caches and workspace -------------------------------------------------------
# Shared with the vllm/llamacpp compose services via bind mounts (see docker-compose.yml).
ENV HF_HOME=/cache/huggingface \
    HF_XET_HIGH_PERFORMANCE=1 \
    TORCH_HOME=/cache/torch \
    TRITON_CACHE_DIR=/cache/triton
RUN mkdir -p /cache/huggingface /cache/torch /cache/triton /cache/weights

WORKDIR /workspaces/muse-glimmer-long-ctx
