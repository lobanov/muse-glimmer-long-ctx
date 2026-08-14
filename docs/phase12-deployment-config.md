# Phase 12 — RTX 5090 Deployment Configuration (32 GB / 512k)

Status: **CONFIG COMPLETE — awaiting hardware.** Every command below is final except
on-device numbers. Analytic basis: phase-0 GB10-measured components (hardware-independent
for weights/KV/compute buffers; see `docs/phase0-compat-memory-spike.md`).

## Artifact

`<model>-Q4_K_M.gguf` (~15 GiB weights) + official `mmproj` projector + stock `dflash`
drafter (§11 pipeline output: `outputs/gguf/<name>/<name>-Q4_K_M.gguf`).

## Memory budget (measured components → 5090 fit)

| ctx | KV dtype | weights | KV | compute | total | margin vs ~31 GB |
|---:|---|---:|---:|---:|---:|---|
| 128k | F16 | 15,246 MiB | 1,762 MiB | 274 MiB | **16.88 GB** | ~14 GB |
| 256k | F16 | 15,246 MiB | 3,426 MiB | 402 MiB | **18.63 GB** | ~12 GB |
| 384k | F16 | 15,246 MiB | 5,090 MiB | 530 MiB | **20.38 GB** | ~11 GB |
| 512k | F16 | 15,246 MiB | 6,754 MiB | 658 MiB | **22.13 GB** | ~9 GB |
| 512k | Q8_0 | 15,246 MiB | 3,588 MiB | 1,069 MiB | **19.44 GB** | ~12 GB |

DFlash drafter adds weights + its own KV at full context — measure before enabling at 512k
(plan: `-c 524288` forces a large drafter KV; if it pushes past budget, cap spec-decode
to ≤256k or use Q8_0 KV at 512k).

## Launch configurations (in order of preference)

```bash
# 1) PRIMARY — F16 KV @ 512k, iSWA default (never --swa-full)
llama-server -m <model>-Q4_K_M.gguf \
  --mmproj mmproj-Muse-Glimmer-30B-Q4_K_M.gguf \
  -ngl 99 -c 524288 --host 0.0.0.0 --port 8080 \
  --jinja --temp 1.0 --top-p 0.95 --top-k 64

# 2) FALLBACK — Q8_0 KV @ 512k (zero forks; use if 1) tightens with drafter/UI overhead)
llama-server ... -c 524288 --cache-type-k q8_0 --cache-type-v q8_0 ...

# 3) DFlash speculative — measure acceptance + memory first (see below)
llama-server ... -c 524288 --spec-type draft-dflash \
  --draft-model dflash-Muse-Glimmer-30B-Q4_K_M.gguf ...
```

> Flag spellings for the drafter (`--spec-type` / `--draft-model`) follow PLAN §11's
> citation of the official GGUF docs; they could not be re-verified on-target because
> the GPU is fully held by the eval sidecar (llama-server aborts at CUDA init).
> Confirm with `llama-server --help` during qualification step 1 — it is part of the
> checklist below.

Client contract (harness already enforces): `chat_template_kwargs.reasoning_strength=low`,
generous `max_tokens`, score `message.content`; sampling temp 1.0 / top-p 0.95 / top-k 64.

## On-device qualification checklist (runs when the 5090 arrives)

1. `docker exec dev bash scripts/verify-env.sh` → ALL CHECKS PASSED (after
   `detect-host-gpu.sh` + rebuilds on the x86 host — CONTRIBUTING §8). Also confirm
   `llama-server --help` flag spellings (`--cache-type-k/v`, `--spec-type`,
   `--draft-model`) on the 5090 build.
2. Buffer-report check at each ctx ∈ {128k, 256k, 384k, 512k}: CUDA0 model/KV/compute
   totals vs the table above (script: `scripts/measure_kv_memory.sh`).
3. iSWA engagement in logs: 39 SWA layers at **window-sized cache (2,560 cells /
   97.5 MiB F16)**, NOT full-context (`--swa-full` absent).
4. Accuracy: 512k NIAH grid via harness (engine `llamacpp`, config_label `kquant-5090`)
   — compare against stock@512k and the §11 quant-parity mini-suite.
5. Throughput: pp512/td512 via `llama-bench -m <gguf> -p 512000 -n 512` with and without
   DFlash; record drafter acceptance rate; prefill wall-clock at 512k (agentic-relevant).
6. nvidia-smi is valid on x86 (unlike GB10 `[N/A]`) — cross-check totals, but treat
   buffer reports as ground truth.

## Deferred to §13 (only after 512k is robust)

1M probe: analytic F16 ≈ 30 GB (borderline), Q8_0 ≈ 23 GB (comfortable), turbo3 ≈ 2.9 GB —
compute in the 13 global layers is the real limit, not VRAM.
