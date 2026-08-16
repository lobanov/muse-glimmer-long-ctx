# Phase 3 Report — Stock Glimmer Baseline (PLAN §3)

Status: **≤128k COMPLETE · >128k RUNNING** (grid `outputs/eval/stock_vllm_gt128k.jsonl`,
log `logs/eval-stock-gt128k.log`; vLLM on `/arms/stock-524k` @524,288). This report
records the completed ≤128k half; §">128k" below is pending and must be filled from the
finished grid before any §3 conclusions are final. Raw: `outputs/eval/stock_vllm_le128k.{jsonl,parquet}`.

## Setup

- Engine: vLLM (`--model-impl transformers`), BF16, `max_model_len` 131,072 for this grid.
- Contract: capability mode (temp 1.0 / top-p 0.95 / top-k 64), `reasoning_strength=low`,
  `max_tokens` 4096, score on `message.content` (PLAN §2 controls; per-row `sampling` logged).
- Grid: 6 tasks × {32k, 64k, 128k} × 7 depths {0,10,25,50,75,90,100%} × 3 resamples = 378 cells,
  ~90% context fill. Zero transport errors; zero length-truncations (`finish_reason=length` = 0).

## Results (mean ± 95% CI, n=21 per task×ctx)

| task | 32k | 64k | 128k |
|---|---|---|---|
| niah (single needle) | **1.000 ± 0.000** | **1.000** | **1.000** |
| niah_multi (4 needles, ±30% spread) | **1.000** | **1.000** | **1.000** |
| multihop (2-hop, mirrored depths) | **1.000** | **1.000** | **1.000** |
| semantic (NoLiMa-style, low lexical overlap) | **1.000** | **1.000** | **1.000** |
| abstain (needle absent) | **1.000** | **1.000** | **1.000** |
| counting (k=5–12 spread markers) | 0.952 ± 0.093 | 0.762 ± 0.187 | 0.476 ± 0.219 |

- **Position insensitivity**: perfect scores at depths 0% and 100% on every retrieval task —
  no edge-of-context effect at ≤128k (GOAL criterion 3 is not the binding constraint here).
- **Counting error anatomy** (see snapshot): **ERRATUM (2026-08-15, post adversarial
  review)** — the original claim "every miss an exact off-by-one undercount, never
  over-count" was based on the first 4 misses mid-grid and is **false at full scale**:
  12/17 misses are k−1, one is an **over-count (13 vs 12)**, four under-count by 2–3.
  Undercount-biased but not perfectly systematic. The snapshot's anatomy table is
  computed from data and was always correct. **Mechanism not established**: attention
  dilution remains one live hypothesis; enumeration/arithmetic slips in the decode
  strategy (reasoning traces show entry-number enumeration with self-corrections) are
  an unexcluded alternative — E2 forensics (greedy ± enumeration-hint re-run of the
  exact miss instances) is running to split them; §4a's causal framing should be read
  as hypothesis-testing, not established mechanism.
- Scoring note: 63 semantic cells were initially mis-scored (scorer required the leading
  article; the model correctly drops it). Re-scored from stored responses with `rescored`
  flags; fix committed before the >128k grid imported the module (`e4de1d9`).

## Interpretation vs the plan's expectations

1. **Stock Glimmer is near-perfect ≤128k on retrieval + abstention + 2-hop reasoning.**
   Community reports of strong zero-shot long-context behavior are consistent with this.
2. **The first degradation axis is aggregation-under-distractor** (counting: 0.95 → 0.48),
   with an undercount-biased (not perfectly systematic) signature. PLAN §10's mode B
   (global retrieval/selectivity) is the leading interpretation — **one live hypothesis,
   not established** (E2 forensics pending; enumeration-slip alternative unexcluded).
   The §4 sweep tests the attention-temperature lever causally; it does not presuppose
   the mechanism.
3. Per PLAN §3's threshold rule, the "training optional" call (≥85% relative retention at
   256k+ on retrieval tasks) **awaits the >128k grid** — ≤128k data alone cannot decide it.

## Latency (median wall, includes prefill; GB10 lower bound)

| ctx | niah | counting |
|---|---|---|
| 32k | 30 s | 83 s |
| 64k | 43 s | 106 s |
| 128k | 80 s | 164 s |

Roughly linear in context with ~600–2700 tok/s effective prefill (prefix caching active
across resamples; see phase-2 notes on interpreting within-cell medians).

## §0 caveat — CLOSED (was recorded in docs/phase0)

K-Quant GGUF via llama.cpp re-ran the failing cell class (128k @ 90% depth × 3, greedy,
`reasoning_strength=low`, `max_tokens` 4096): **3/3 hit** (`parity_caveat_llamacpp.jsonl`).
The original miss was the reasoning-budget artifact, not retrieval. Quant artifact remains
a valid baseline for §11 comparisons.

## >128k (192k / 256k / 384k / 512k) — COMPLETE (2026-08-16 07:22, 216/216, n=9/cell)

| task | 192k | 256k | 384k | 512k |
|---|---|---|---|---|
| niah | 1.000 | 1.000 | 1.000 | 1.000 |
| niah_multi | 1.000 | 1.000 | 1.000 | 1.000 |
| multihop (2-hop) | 1.000 | 1.000 | 1.000 | 1.000 |
| semantic (NoLiMa-style) | 1.000 | 1.000 | 1.000 | 1.000 |
| abstain | 1.000 | 1.000 | 1.000 | 1.000 |
| counting | 0.667 | 0.667 | 0.667 | 0.222 |

### Decision-rule verdict (PLAN §3: ≥85% relative retention on retrieval tasks)

Every retrieval/reasoning task retains **100.0%** of its 128k reference at every length
through 512k (all depths incl. 0%/100%). The ≥85% threshold is met with maximal margin:
**the project formally takes the strengthen-qualify-deploy branch** (training optional/
targeted; recorded in PLAN §4 revision). Stock Glimmer @512k (mechanical window only)
meets GOAL criteria 3 and 4 outright.

### Counting beyond 128k — final picture (with the audit's corrections)

Non-monotone at 0.667 across 192k–384k (vs 0.476 @128k), then **0.222 @512k** — the only
substantial drop. Interpretation per the audit reframing: aggregation fragility that is
k-difficulty-dependent and partly stochastic (E2: ~30% of capability misses flip under
greedy; 11/17 enumeration-resistant), with a genuine length component emerging only at
4× nominal. The k-matched grid (strata k=6/11, capability+greedy; running) is the clean
instrument for any claim about length per se.

### Latency (512k cells, GB10, single-lane medians from log)

TTFT ~500–1400 s per 463k-token prompt (prefill-dominated; ~350–900 tok/s effective;
three concurrent client lanes during collection — treat as lower bounds; 5090 ≈ 6×
bandwidth). Decode adds ~10–60 s at ≤4096 max_tokens. Agentic 512k use on GB10 is
patient-but-usable; on the 5090 target it should be comfortably interactive.
