# syntax=docker/dockerfile:1
# llama.cpp deployment engine — pinned release, CUDA 12.8 toolchain, Blackwell archs.
#
# Why CUDA 12.9 (not 12.8, not 13.x): sm_121 (GB10/DGX Spark) is NOT supported by nvcc
# in CUDA 12.8 (verified: `nvcc --list-gpu-arch` tops out at compute_120) — first added in
# 12.9. And the TurboQuant KV forks are validated on CUDA 12.8/12.9 with a known MMQ
# segfault on 13.1, so staying on 12.9 keeps both the main engine and the future turbo
# build on one toolchain. CUDA 13.0.0 also lists compute_121 if you ever need to match
# the DGX Spark host toolkit exactly.
#
# Build args:
#   LLAMA_TAG  — llama.cpp release tag (must be >= b10353 for Muse Glimmer)
#   CUDA_ARCHES — sm_120 (RTX 5090) + sm_121 (GB10) by default
#   CUDA_TAG   — base image tag (12.9.1-devel-ubuntu24.04; multi-arch arm64/amd64, verified)
ARG CUDA_TAG=12.9.1-devel-ubuntu24.04
FROM nvidia/cuda:${CUDA_TAG}

ARG LLAMA_TAG=b10428
ARG CUDA_ARCHES=120;121

RUN apt-get update \
    && apt-get install -y --no-install-recommends git cmake build-essential ccache libcurl4-openssl-dev \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 --branch "${LLAMA_TAG}" https://github.com/ggml-org/llama.cpp /src/llama.cpp
WORKDIR /src/llama.cpp

# nvidia/cuda devel images ship a link-time *stub* for the CUDA driver API at
# /usr/local/cuda/lib64/stubs/libcuda.so — but ld looks for the SONAME libcuda.so.1.
# Expose the stub under that name. Resolution of the shlib's NEEDED entry requires
# -rpath-link (plain -L / LIBRARY_PATH is NOT consulted for transitive deps — verified
# the hard way). At RUNTIME the real host driver lib is bind-mounted by docker-compose.
RUN ln -sf /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1
ENV STUBS_DIR=/usr/local/cuda/lib64/stubs

# GGML_CUDA_FA_ALL_QUANTS: FlashAttention for all quant/KV-type combos (incl. q8 KV at long ctx)
RUN cmake -B build \
      -DGGML_CUDA=ON \
      -DGGML_CUDA_FA_ALL_QUANTS=ON \
      -DGGML_CUDA_FORCE_MMQ=ON \
      -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHES}" \
      -DCMAKE_EXE_LINKER_FLAGS="-L${STUBS_DIR} -Wl,-rpath-link,${STUBS_DIR}" \
      -DLLAMA_CURL=ON \
      -DCMAKE_BUILD_TYPE=Release \
    && cmake --build build --config Release -j"$(nproc)" \
        --target llama-server llama-cli llama-bench llama-quantize llama-gguf-split

ENV PATH="/src/llama.cpp/build/bin:${PATH}"
# sanity: binaries exist and load (driver stub satisfies libcuda.so.1 at build time;
# the real driver is bind-mounted at runtime by docker-compose)
RUN LD_LIBRARY_PATH="${STUBS_DIR}" llama-server --version
