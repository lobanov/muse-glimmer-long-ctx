# Phase 6 Report — Context/Position Sampler (PLAN §6)

Status: **COMPLETE** (component level — it is trainer-independent by design and built
early because it needs no GPU; §5 is blocked on Z.ai credentials, see PLAN §5 status note).
Code: `src/muse_longctx/position_sampler.py`. Verified: built-in selftest (asserts below),
run via `python3 src/muse_longctx/position_sampler.py` → `selftest OK`.

## What it produces

`TrainingSample`: `input_ids`, `position_ids`, `labels`, `loss_mask`,
`evidence_positions` (virtual-space), `physical_len`, `virtual_len`, `mode`, `meta`.

## Modes (PLAN §6 numbering)

| # | Mode | Behaviour | Selftest assertion |
|---|---|---|---|
| 1 | `normal` | positions 0..L−1 | identity |
| 2 | `uniform_offset` | random constant offset, contiguous | start ≥ 0, end < V, span = L−1 |
| 3 | `random_segments` | k segments scattered into non-overlapping contiguous virtual blocks (largest-first rejection sampling; global monotonicity **not** guaranteed — by design) | run-decomposition → blocks pairwise non-overlapping, all in [0, V) |
| 4 | `pose` | PoSE (Zhu et al. 2024): contiguous chunks, forward jumps summing to V−L | strictly increasing, starts at 0, **hits V−1 exactly** |
| 5 | `yarn_random` | = pose with virtual = yarn_factor × physical (default 4×) | same, at V ∈ {2L, 4L, 3.5L} |
| 6 | `genuine` | identity, virtual == physical (flagged separately for bookkeeping) | identity |

Tested at L=8192 with V up to 4L; seeds via `random.Random` for reproducibility.

## Architecture scope (unchanged from PLAN)

Modes 2–5 only train absolute-position robustness of the 39 local RoPE/SWA layers; the 13
global NoPE layers receive no position IDs at all. Expect genuine-length data (mode 6) to
carry nearly all signal; virtual modes stay available for the §9 training-position ablation
(genuine ≫ mixed ≫ virtual is the prior).

## Integration contract

`build_training_sample(input_ids, loss_token_spans, evidence_token_spans, layout, …)`
takes pre-tokenized ids — tokenization and corpus construction (§5) live upstream; the
trainer (§7) consumes `TrainingSample` fields directly (labels already −100-masked).
