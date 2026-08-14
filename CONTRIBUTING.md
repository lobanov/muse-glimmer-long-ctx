# CONTRIBUTING — Development Environment

All GPU work runs in Docker containers managed by Compose; the host needs only Docker.
Humans can use VS Code's devcontainer or plain `docker compose`; coding agents work from
the host via `docker exec` / `docker compose` (§4). No host-side Python, no sudo.

Verified end-to-end on the DGX Spark (GB10, aarch64, sm_121) — see §7 for measurements.
Related: `GOAL.md`, `PLAN.md`, `MODEL.md`.

## 1. What runs where

| Service | Image (current pin) | Purpose |
|---|---|---|
| `dev` | `muse-glimmer-dev:26.07-py3` — NGC PyTorch 26.07 (torch 2.13.0a0+nv26.07, CUDA 13.3, flash-attn) + uv-managed HF stack | all training/eval code; IDEs attach here |
| `vllm` | `vllm/vllm-openai:muse-glimmer` (vLLM 0.26.1rc1) | research inference sidecar, `:8000/v1` |
| `llamacpp` | `muse-glimmer-llamacpp:b10428` — CUDA 12.9.1, sm_120+sm_121 | deployment engine |

Files: `.devcontainer/{devcontainer.json, docker-compose.yml, Dockerfile, llamacpp.Dockerfile}`,
`scripts/{detect-host-gpu.sh, verify-env.sh}`, `cache/` (gitignored; HF + weights caches,
shared by all services), `logs/` (gitignored; background process output).

GPU passthrough needs no nvidia runtime and no sudo: compose passes `/dev/nvidia*` and
bind-mounts the host driver libs at their SONAME paths, version-resolved from `.env` by
`detect-host-gpu.sh`. On a host that *does* have the nvidia runtime, `gpus: all` works too.

## 2. One-time setup

```bash
./scripts/detect-host-gpu.sh     # writes .devcontainer/.env (driver ver, arch) + creates cache/ as your user

# optional but recommended — authenticated HF downloads (~68 MB/s vs ~3 MB/s):
echo "HF_TOKEN=$HF_TOKEN" >> .devcontainer/.env && chmod 600 .devcontainer/.env
```

Then either open the repo in VS Code/Cursor → "Reopen in Container", or:

```bash
cd .devcontainer && docker compose up -d dev
docker compose exec dev bash scripts/verify-env.sh   # health gate → expect: ALL CHECKS PASSED
```

First build pulls the ~25 GB NGC base (~30 min). Rebuilds are only needed when
`pyproject.toml`, the Dockerfiles, or compose change (`docker compose build dev`) — rerun
`verify-env.sh` after any rebuild.

### devcontainer CLI (optional convenience layer)

The spec's reference implementation (`@devcontainers/cli`, tested v0.88.0) wraps the same
compose stack with spec lifecycle semantics. Use it if you prefer one-command bring-up:

```bash
npx -y @devcontainers/cli up --workspace-folder .      # build + start + postCreate (verify-env)
npx -y @devcontainers/cli exec --workspace-folder . <cmd>   # name-independent exec
```

Empirically verified: it adopts our compose project as-is (same container names, profile
services untouched) and runs `postCreateCommand` automatically. Two caveats:

- `exec` only finds containers **created by the CLI or VS Code** (they carry `devcontainer.*`
  labels). If the stack was started with plain `docker compose up`, `exec` fails with
  "Dev container not found" until the `dev` container is recreated via `devcontainer up`
  (`docker compose rm -f dev` first, if needed). `docker compose` commands always work.
- `npx` adds ~5–10 s per call — agents should keep using `docker exec` (§4).

Verdict: compose remains the source of truth; the CLI is a human/portability convenience
(also the path to `devcontainer build --push` prebuilt images for the 5090 box later).

## 3. Daily usage

```bash
cd .devcontainer   # compose lives here

# run code / shells in the dev container
docker compose exec dev python3 …
docker compose exec dev bash

# vLLM sidecar (~10 min cold start; 63 GB BF16 weights download once into cache/huggingface)
docker compose --profile inference up -d vllm
curl -s localhost:8000/v1/models | jq '.data[0].id'   # → "muse-glimmer"

# llama.cpp server (weights in cache/weights/; full flag set in compose comments)
docker compose --profile llamacpp run --service-ports llamacpp \
  llama-server -m /cache/weights/Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf \
    --mmproj /cache/weights/mmproj-Muse-Glimmer-30B-Q4_K_M.gguf -ngl 99 -c 524288 --jinja

# CUDA smoke test of the engine image (15 MB model in cache/weights/)
docker compose --profile llamacpp run --rm llamacpp \
  llama-bench -m /cache/weights/stories15M-q8_0.gguf -ngl 99 -p 128 -n 32
```

Downloading weights (host or in `dev`; authenticated if HF_TOKEN is set):

```bash
hf download meta-models/Muse-Glimmer-30B                    # BF16 → cache/huggingface (vllm)
hf download meta-models/Muse-Glimmer-30B-GGUF --local-dir cache/weights \
  --include "*KQuant-17GB*" --include "mmproj-*" --include "dflash-*"
```

Model-usage notes that affect any client (eval harness included):

- Glimmer is a reasoning model: thinking lands in `message.reasoning`, answer in
  `message.content`. With small `max_tokens` the budget can be consumed *inside* reasoning →
  `content: null`. Always pass `"chat_template_kwargs": {"reasoning_strength": "low"}` and a
  generous `max_tokens` (this is the mechanism behind PLAN.md §2's requirement).
- Sampling defaults per Meta: temp 1.0, top-p 0.95, top-k 64.
- GB10's `nvidia-smi` reports `[N/A]` for memory — use vLLM's startup log or `tegrastats`.

## 4. Working with a coding agent

Coding agents (pi, Claude Code, …) run **on the host**, not inside the container. Docker is
their only GPU gateway. This section is mirrored in `AGENTS.md`, which agents load
automatically at session start — keep the two in sync when changing the contract below.

- **Live filesystem**: the repo is bind-mounted at `/workspaces/muse-glimmer-long-ctx` in
  `dev`. Agent edits on the host are immediately visible in the container; anything the
  container writes under the repo or `cache/` is visible on the host. No sync step.
- **Execution**: one-off commands via `docker compose exec dev <cmd>` (from `.devcontainer/`)
  or `docker exec muse-glimmer-long-ctx-dev-1 <cmd>` (from anywhere). Services via
  `docker compose --profile …`. Long jobs: `docker exec -d … '… > logs/<name>.log 2>&1'`
  so output stays host-visible under `logs/`.
- **Shared caches**: weights downloaded once (host or `dev`) are available to every service;
  never re-download inside a service that could reuse `cache/`.
- **Health gate**: after any container rebuild/recreate, `scripts/verify-env.sh` must print
  `ALL CHECKS PASSED` before proceeding — agents should treat this as a precondition.
- **Hygiene**: experiment outputs → host paths (bind-mounted), not container-internal paths;
  background-process logs → `logs/`; nothing root-owned in `cache/` (see §6).

## 5. Gotchas — encoded in the infra, don't undo them

Each of these was verified the hard way; the file that encodes the fix is listed.

- **torch must never be pip-installed** — the NGC build carries GB10 kernels + flash-attn.
  uv installs with `--excludes torch`; an exact torch pin breaks resolution (no index
  candidate for `+nv26.7` local versions). → `.devcontainer/Dockerfile`
- **NGC tag, not pyproject, anchors torch** — hence bounded version ranges in
  `pyproject.toml` (floors = validated set; NGC's newer bundled versions win). → `pyproject.toml`
- **CUDA 12.8 has no sm_121** — `nvcc fatal: Unsupported gpu architecture 'compute_121'`;
  sm_121 first ships in CUDA 12.9. Engine image uses 12.9.1 with `120;121` (also serves the
  RTX 5090 box; 12.9 stays on the TurboQuant-safe toolchain line). → `llamacpp.Dockerfile`
- **Linking in bare CUDA devel images needs the driver stub** — stub symlinked to the
  SONAME + `-Wl,-rpath-link`; `-L`/`LIBRARY_PATH` alone do not resolve transitive
  `DT_NEEDED libcuda.so.1`. Runtime uses the real host lib via compose mounts.
  → `llamacpp.Dockerfile`
- **vLLM service needs `LD_LIBRARY_PATH`** — triton's `libcuda_dirs()` otherwise asserts
  (`libcuda.so cannot found!`); the NGC dev image doesn't need this. → `docker-compose.yml`
- **No `hf-transfer`** — deprecated with huggingface_hub ≥ 1.x; Xet is on via
  `HF_XET_HIGH_PERFORMANCE=1`. → `.devcontainer/Dockerfile`
- **transformers ≥ 5.15** is the floor: 5.15.0 is the release that added Muse Glimmer.
  → `pyproject.toml`
- **bitsandbytes** on NGC 26.07 logs "No prebuilt binary for CUDA 13.3, loading CUDA 13.2" —
  benign. → set `BNB_CUDA_VERSION` only if an actual kernel failure appears.

## 6. Troubleshooting

| Symptom | Fix |
|---|---|
| `torch.cuda.is_available() == False` / no `nvidia-smi` in container | rerun `scripts/detect-host-gpu.sh`; confirm `.env` driver version matches host (`readlink -f /usr/lib/$(dpkg-architecture -qDEB_HOST_MULTIARCH 2>/dev/null || echo aarch64-linux-gnu)/libcuda.so.1`) |
| uv resolution error mentioning `torch==…nv…` | a torch constraint crept back in — restore `--excludes` pattern (§5) |
| `nvcc fatal: compute_121` / linker `cuMem*` undefined refs | toolkit < 12.9 / missing stub flags — both handled in `llamacpp.Dockerfile`; don't downgrade the base |
| Host can't write into `cache/` | compose created dirs root-owned before the detect script ran: `docker run --rm --platform linux/arm64 -v "$PWD/cache:/c" alpine chown -R $(id -u):$(id -g) /c` |
| vLLM `AssertionError: libcuda.so cannot found!` | `LD_LIBRARY_PATH` missing on the service (§5) |
| vLLM OOM at startup | lower `--gpu-memory-utilization` / `--max-model-len` in compose |
| Unauthenticated-crawl download speeds | `HF_TOKEN` missing from `.devcontainer/.env`, then recreate the service |

## 7. Reference measurements (DGX Spark, 2026-08-14)

| Item | Value |
|---|---|
| dev image build (after ~30 min NGC pull) | ~3 min |
| llamacpp image build | ~5 min (9.68 GB) |
| llama-bench smoke (stories15M-q8_0) | pp128 ≈ 119,700 t/s · tg32 ≈ 4,320 t/s |
| Glimmer BF16 download | 63 GB @ ~68 MB/s (~21 min), resumable |
| vLLM cold start | ~9.5 min; 55.8 GiB weights, 51.5 GiB KV = 3.64M tokens (27.8 × 131k concurrency) |
| vLLM functional | NIAH exact retrieval at 35k (10% depth) and 60k (90% depth); ~920 tok/s e2e incl. prefill |

## 8. Portability (RTX 5090 box)

Same files, multi-arch images: rerun `detect-host-gpu.sh` (writes `x86_64` + that host's
driver), rebuild both images (layers are arch-specific). `CUDA_ARCHES=120` alone is fine
there; `gpus: all` can replace the manual mounts if the nvidia runtime is configured.
