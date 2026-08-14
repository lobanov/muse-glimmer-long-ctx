# Phase 0 Report — Compatibility & Memory Spike (PLAN §0)

Status: **COMPLETE** (commit `d8250c3`; gates re-verified 2026-08-14, this report added after).
Raw data: `outputs/parity_spike/*.jsonl`, `logs/mem-*.log`. Tools: `scripts/measure_kv_memory.sh`,
`scripts/summarize_memory.py`, `evals/parity_spike.py`.

## Outcome

All three go/no-go gates **PASSED**. The 32 GB RTX 5090 fit for F16-KV @ 512k is decided
analytically from GB10-measured components: **22.1 GB device total, ~9 GB margin**.

## Gate 1 — Parity (BF16/vLLM vs K-Quant GGUF/llama.cpp), PASSED with caveat

NIAH parity set: {32k, 64k, 128k} × {10%, 50%, 90%} depth × 3 reps, greedy (temp 0), identical
prompts on both engines.

| Engine | Score | Misses |
|---|---|---|
| BF16 (vLLM, transformers impl) | **27/27** | — |
| K-Quant Q4_K_M 17 GB (llama.cpp) | **26/27** | 128k @ 90% depth, rep 0 |

The single miss is **not a retrieval failure**: the quantized model exhausted the 1,024-token
budget inside its reasoning channel (`finish_reason: length`, empty `content`) where BF16
answered tersely. → **Action item for §2 harness** (closed there): pin
`chat_template_kwargs.reasoning_strength: "low"`, generous `max_tokens`, score
`message.content` only; re-run this cell class under those settings.

Wall-clock (median/max, greedy, includes prefill):

| ctx | vLLM | llama.cpp | prompt tokens (actual) |
|---:|---|---|---:|
| 32k | 65 / 105 s | 48 / 56 s | ~28.6k |
| 64k | 84 / 144 s | 87 / 94 s | ~57.4k |
| 128k | 112 / 185 s | 177 / 248 s | ~115.5k |

## Gate 2 — Memory model (measured GB10 components → analytic 5090 fit), PASSED

llama.cpp buffer reports at `-c` ∈ {128k, 256k, 384k, 512k}, F16 and Q8_0 KV
(`logs/mem-f16-*.log`, `logs/mem-q8_0-524288.log`):

| ctx | KV dtype | weights | KV cache | compute | device total | fits ~31 GB? |
|---:|---|---:|---:|---:|---:|---|
| 128k | F16 | 15,246 MiB | 1,762 MiB | 274 MiB | **16.88 GB** | yes |
| 256k | F16 | 15,246 MiB | 3,426 MiB | 402 MiB | **18.63 GB** | yes |
| 384k | F16 | 15,246 MiB | 5,090 MiB | 530 MiB | **20.38 GB** | yes |
| 512k | F16 | 15,246 MiB | 6,754 MiB | 658 MiB | **22.13 GB** | yes (~9 GB margin) |
| 512k | Q8_0 | 15,246 MiB | 3,588 MiB | 1,069 MiB | **19.44 GB** | yes |

Global-layer KV scales at exactly 1,024 B/token (F16) per token across K+V for the 13
non-SWA layers — matches the theoretical 13,312 B/token. `CUDA_Host` buffers are pinned host
RAM and excluded. Extrapolation: 1M F16 ≈ 30 GB (borderline), 1M Q8_0 ≈ 23 GB (comfortable).

## Gate 3 — iSWA verification, PASSED

llama.cpp logs (`--swa-full` NOT set) show the 39 sliding-window layers getting a
window-sized cache — **2,560 cells / 97.50 MiB F16 (51.80 MiB Q8_0)** — constant across all
context sizes; only the 13 global layers scale with context. Model metadata confirmed:
`sliding_window = 2048`, `sliding_window_pattern` alternating 2 SWA : 1 global across 52
layers, `freq_base_swa = 500000` (RoPE only on SWA layers).

## Deferred (awaits RTX 5090 hardware — moved to PLAN §12)

- On-device 32 GB fit validation (predicted: fits, 22.1 GB @ 512k F16)
- On-device throughput/latency qualification at 128k–512k
- TurboQuant sm_120 fork validation on-target

Spark throughput numbers above are lower bounds (GB10 has ~6× less bandwidth than a 5090).

## Engine compatibility facts established

- transformers 5.15.0 loads the model; `muse_glimmer` modeling code present in-tree.
- vLLM serves it via `--model-impl transformers --tool-call-parser muse_glimmer
  --reasoning-parser muse_glimmer` (~9.5 min cold start, 55.8 GiB weights).
- llama.cpp **b10428**, CUDA 12.9.1, `sm_120`+`sm_121` — official 17 GB K-Quant GGUF loads
  and serves through 512k context.
