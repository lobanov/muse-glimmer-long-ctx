# AGENTS.md — Agent Operating Instructions

This file is loaded automatically by coding agents (pi, Claude Code, …) at session start.
Human-oriented docs: `CONTRIBUTING.md` (environment), `GOAL.md` → `PLAN.md` → `MODEL.md`
(project). This file repeats only what an agent needs to act correctly in this repo.

## Project in one paragraph

Adapting **Muse Glimmer 30B** (hybrid SWA-RoPE + NoPE-global VLM, 131k native context) to a
reliably usable **512k context** on a 32 GB RTX 5090, training via QLoRA on the DGX Spark.
Current phase: environment + PLAN §0 (baselines/spikes). Inference engines: vLLM (research,
sidecar) and llama.cpp (deployment). Eval harness and training code do not exist yet —
you will be writing them.

## Environment contract (how to run anything)

You run on the **host**. Docker is the only GPU gateway — never install Python packages on
the host, never use sudo. Everything GPU-related goes through the `dev` container.

```bash
# one-off commands (from repo root — note the container name):
docker exec muse-glimmer-long-ctx-dev-1 python3 <script.py>
docker exec muse-glimmer-long-ctx-dev-1 bash -c '<command>'

# equivalent, from .devcontainer/:
docker compose exec dev <command>

# services:
cd .devcontainer
docker compose --profile inference up -d vllm      # research sidecar → localhost:8000/v1
docker compose --profile llamacpp run --rm llamacpp <llama-* binary>   # deployment engine

# health gate — MUST pass before/after any container rebuild or recreate:
docker exec muse-glimmer-long-ctx-dev-1 bash scripts/verify-env.sh    # → ALL CHECKS PASSED
# (a "[warn] GPU busy" line is acceptable — it means the vLLM sidecar holds the memory)

# rebuild after touching pyproject.toml / Dockerfiles / compose:
cd .devcontainer && docker compose build dev
```

Note: `npx -y @devcontainers/cli up/exec` also works (human convenience; see CONTRIBUTING.md §2)
but stick to `docker exec`/`docker compose` — faster, and unaffected by devcontainer-label state.

Rules of the road:

- **Filesystem is live**: repo is bind-mounted at `/workspaces/muse-glimmer-long-ctx` in `dev`.
  Edit on host, run in container — no sync. Write experiment outputs to host-visible paths
  (repo dirs or `cache/`), never container-internal-only paths.
- **Long jobs**: launch detached with output under `logs/` (host-visible, gitignored):
  `docker exec -d muse-glimmer-long-ctx-dev-1 bash -c '… > /workspaces/muse-glimmer-long-ctx/logs/<name>.log 2>&1'`
  — then poll with `tail`.
- **Shared caches**: `cache/huggingface` (HF hub cache, used by dev + vllm), `cache/weights`
  (GGUFs for llamacpp), `cache/torch`, `cache/triton`. Download once, reuse across services.
  Downloads: `docker exec muse-glimmer-long-ctx-dev-1 hf download <repo>` (auth comes from
  `.devcontainer/.env` via compose).
- **Cold-start expectations**: vLLM sidecar ~10 min to serve (60 GB BF16); don't poll faster
  than ~30 s. llama-server at 512k context takes minutes of prefill — that's normal.

## Key files and directories

| Path | What it is |
|---|---|
| `GOAL.md` | project objective + success criteria |
| `PLAN.md` | experiment plan (§0 = current phase: compat/memory spikes, baselines) |
| `MODEL.md` | Glimmer architecture reference (layers, tensors, KV math, engines) |
| `CONTRIBUTING.md` | full environment doc, gotchas, troubleshooting, measurements |
| `AGENTS.md` | this file |
| `.devcontainer/docker-compose.yml` | 3 services (dev/vllm/llamacpp), GPU passthrough, shared caches; profiles: `inference`, `llamacpp` |
| `.devcontainer/Dockerfile` | dev image: NGC PyTorch base + uv-installed HF stack (`--excludes torch`) |
| `.devcontainer/llamacpp.Dockerfile` | llama.cpp b10428, CUDA 12.9.1, sm_120+sm_121 |
| `.devcontainer/devcontainer.json` | VS Code/Cursor entry point |
| `.devcontainer/.env` | **generated, gitignored, contains HF_TOKEN** — never commit, never print |
| `scripts/detect-host-gpu.sh` | regenerates `.env` (driver version, arch) + pre-creates `cache/` |
| `scripts/verify-env.sh` | health gate (GPU, NGC torch, transformers/muse_glimmer, caches) |
| `pyproject.toml` | dependency floors (bounded ranges; NGC tag anchors torch) |
| `cache/` | gitignored model/dependency caches (see above) |
| `logs/` | gitignored background-job output — check here before assuming a job died |
| `src/`, `evals/`, `data/` | **do not exist yet** — create per PLAN.md when the work starts |

## Hard rules (violating these breaks the environment)

1. **Never install/upgrade torch, flash-attn, or huggingface libs ad hoc.** torch comes from
   the NGC base image only; a pip-installed torch loses GB10 kernels. Dependency changes go
   through `pyproject.toml` (bounded ranges, never torch/flash-attn/hf-transfer) +
   `docker compose build dev` + verify-env.
2. **Never edit the driver-lib bind mounts or CUDA flags** in the Dockerfiles/compose — they
   encode verified fixes (stub linking, `LD_LIBRARY_PATH` for vLLM's triton, CUDA 12.9 for
   sm_121). Rationale in CONTRIBUTING.md §5 if you need it.
3. **Never commit** `.devcontainer/.env`, anything in `cache/` or `logs/`, or model files
   (`*.gguf`, `*.safetensors`) — all gitignored; keep it that way.
4. **After any container rebuild/recreate**: `scripts/verify-env.sh` must print
   `ALL CHECKS PASSED` before doing anything else.
5. **vLLM client calls** (eval harness!): always `"chat_template_kwargs": {"reasoning_strength": "low"}`
   and generous `max_tokens` — Glimmer is a reasoning model and can burn the whole budget in
   the reasoning channel and return `content: null`. Answers are in `message.content`,
   thinking in `message.reasoning`. Sampling defaults: temp 1.0 / top-p 0.95 / top-k 64.
6. **GB10 monitoring**: host `nvidia-smi` shows `[N/A]` for memory — get memory numbers from
   vLLM logs or `tegrastats`, not nvidia-smi.

## Conventions

- Experiment scripts → `scripts/` (infra) or future `src/`/`evals/` per PLAN.md; results →
  Parquet under a host-visible output dir (PLAN.md §2 schema).
- Background jobs always get a named log in `logs/`; reference the log path in any summary.
- When measurements are taken, append them to CONTRIBUTING.md §7 so they accumulate.
- If something breaks in a way §rules here don't cover: check CONTRIBUTING.md §6
  (troubleshooting) before inventing a workaround, and record new gotchas in both files.
