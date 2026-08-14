# Phase 2 Report — Evaluation Harness (PLAN §2)

Status: **COMPLETE (core harness)**. Code: `evals/harness/` (runner, client, 6 synthetic
tasks, Parquet writer, summarizer). Smoke + §0-caveat control run on vLLM (BF16) 2026-08-14.
Raw data: `outputs/eval/*.jsonl`, `*.parquet` (gitignored by policy; results live in docs).

## Components

| File | Role |
|---|---|
| `evals/harness/client.py` | OpenAI-compatible streaming chat client; PLAN §2 contract baked in: `chat_template_kwargs.reasoning_strength: "low"` on every call; capability mode temp 1.0 / top-p 0.95 / top-k 64; parity mode greedy; captures TTFT (prefill proxy), usage, finish_reason, `content` vs thinking |
| `evals/harness/tasks.py` | 6 machine-scored synthetic tasks (below); depth-controllable evidence; resample seeds via sha1 cell hashing |
| `evals/harness/run_eval.py` | Grid runner: task × ctx × depth × rep; resumable JSONL (skips done cells, retries error cells); `--config-label` for zero-shot arms (stock / qk4.x / yarn4 / …) |
| `evals/harness/to_parquet.py` | Common typed Parquet schema (24 cols) — all configs comparable |
| `evals/harness/summarize.py` | Mean score ± 95% CI (t-dist, n≥3 resamples), TTFT/wall medians, length-truncation counts |

Default grids = PLAN §2 spec: ctx {32k, 64k, 128k, 192k, 256k, 384k, 512k}, depths
{0%, 10%, 25%, 50%, 75%, 90%, 100%}, ≥3 resamples.

## Task suite v1 (synthetic; no benchmark data touched — keeps the training-data
exclusion guarantee by construction; community suites integrate behind the same runner)

| Task | Measures | Scoring |
|---|---|---|
| `niah` | single-needle retrieval (variable tracking) | exact code in content |
| `niah_multi` | 4 needles spread ±0.30 around anchor depth | partial credit 0..1 |
| `multihop` | 2-hop chain: pointer needle at depth d, fact needle at 1−d | exact code |
| `counting` | aggregation: k=5..12 marker occurrences spread evenly | exact integer |
| `semantic` | NoLiMa-style retrieval, near-zero lexical overlap question↔evidence | exact place |
| `abstain` | needle absent → "I don't know" (no fabrication) | 1 acknowledged / 0 fabricated / 0.5 silent |

## Smoke run (vLLM, BF16, capability mode, 2026-08-14)

`outputs/eval/smoke_vllm.{jsonl,parquet}`: niah 2/2 hit (32k, depths 0%/90%), abstain
2/2 acknowledged; TTFT 10–48 s @ 28.5k prompt tokens (prefill ≈ 600–2700 tok/s with prefix
caching); Parquet + summarize verified.

## §0 parity-caveat control (action item from docs/phase0-compat-memory-spike.md)

Cell class 128k @ 90% depth × 3 reps, greedy + `reasoning_strength: low` + `max_tokens` 4096:
**vLLM/BF16 3/3 hit** (`outputs/eval/parity_caveat_vllm.jsonl`) — consistent with phase-0
(BF16 never failed). The original miss was llama.cpp/K-Quant: its re-run needs llama-server
up (deferred until the §3 vLLM grid finishes — GPU is serial; tracked in PLAN §3 notes).

## Notes / limitations

- vLLM prefix caching makes later reps of a cell much faster (shared deterministic haystack
  prefix) — wall/TTFT medians within a cell underestimate cold prefill; Spark latencies are
  lower bounds for the 5090 anyway and are recorded, not gated, per PLAN §0.
- Community suites (RULER proper, NoLiMa proper, LongBench v2, ∞Bench, HELMET,
  LongCodeBench/LongSWE) are not yet wired: they need dataset downloads + licence checks
  and will be integrated as additional task modules against the same schema when the §3
  baseline establishes where synthetic tasks saturate.
