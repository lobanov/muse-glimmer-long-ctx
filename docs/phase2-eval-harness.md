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
| `conflicts` | conflicting facts: same key recorded twice; recency rule stated; must report the superseding value (and not the stale one) | 1 correct / 0.5 both / 0 stale |
| `set_intersect` | two distant lists → items in BOTH | IoU over reported set |
| `chronology` | 5 scattered timestamped events → 3 earliest in order | in-order hits /3 |

All tasks ship with a module selftest (`python3 evals/harness/tasks.py`) asserting builder
invariants and scorer edge cases (added after a real precedence bug was caught in `conflicts`).
Note: `conflicts`/`set_intersect`/`chronology` joined after the §3 ≤128k grid started; they
run as a follow-up grid and merge into the same Parquet (identical schema, label `stock`).

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
## Community-suite integration

**NoLiMa: integrated** (`evals/harness/nolima.py`, plugin via `--plugin nolima`). Official
needle set (`amodaresi/NoLiMa`, 58 instances: 10 tasks × tests × {onehop, twohop}), official
word-shuffled haystacks (10 books, ≈1.9M tokens — covers 512k targets), official
"contains" metric and task templates; needle inserted at controllable depth; corpus
token-fill calibrated against the Glimmer tokenizer (cached, `outputs/eval/nolima_calib.json`,
verified 89.8–89.9% fill at 32k/256k). Canonical lengths 250–32K; ours additionally supports
64k–512k as a labelled extension. License: Adobe Research (non-commercial research), eval-only.
RULER / ∞Bench / HELMET / LongSWE integration: pending (LongSWE needs a test-execution
harness — sequenced deliberately after the first end-to-end result; dataset downloads +
licence checks for the rest).

**LongCodeQA (LCB @1M suite): integrated** (`evals/harness/longcodeqa.py`, plugin via
`--plugin longcodeqa`). All 443 official MC instances across 6 buckets (32K/64K/128K/
256K/512K/1M — counts match the paper table exactly: 113/76/92/65/47/50), official prompt
verbatim, `correct_letter` scoring; bucket selection maps to the largest bucket ≤ target.
All sampled repos verified present in `data/exclusions/eval_repos.json`. MIT license.

**LongBench v2: integrated** (`evals/harness/longbench_v2.py`, plugin via `--plugin
longbench_v2`). 503 official MC instances (contexts 10k–4.3M Glimmer tokens, median 97k —
measured & cached), official "answer with the option's letter directly" protocol, first
standalone-letter scoring. Instance selection filters to [0.5, 0.92] × target (usable pools:
79 @ 32k, 104 @ 128k, 90 @ 256k, 44 @ 512k; none below 32k — by dataset design). `depth` is
ignored per-instance (fixed by the original task) and recorded in meta. License: CC-BY-NC,
research/eval-only.
