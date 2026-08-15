# Muse Glimmer 30B — 512k Context Adaptation

Make Muse Glimmer 30B **usefully** use a 512k-token context (4× its native 128k) as a
~17 GB K-Quant GGUF inside 32 GB VRAM, for long-horizon coding and agentic work.
"Usefully" = measured, not nominal: strong retrieval across the whole window, multi-hop
reasoning over distant evidence, repo-scale coding at 256k–512k, no material ≤128k
regression, and materially better than stock at equal length. Train on the DGX Spark;
deploy on an RTX 5090.

## Approach

1. **Measure before adapting.** A single eval harness (10 synthetic tasks + NoLiMa,
   LongBench v2, LongCodeQA, ∞Bench) runs every config under one locked
   sampling/template contract, at 32k–512k with position-controlled evidence.
2. **Let stock go first.** Glimmer's hybrid attention (39 SWA-RoPE local + 13 NoPE
   global layers) suggests the extrapolation burden sits on global selectivity, not
   positions. Baseline confirmed it: retrieval and 2-hop reasoning are *perfect*
   zero-shot to 512k; the real weaknesses are aggregation-under-distractor and
   semantic retrieval at range.
3. **Cheapest fix that wins.** Config-only `qk_scale_factor` arms (attention
   temperature on the NoPE layers) target the weak axis with zero training; YaRN as
   control. Only if they can't move it does QLoRA run — on a machine-verified corpus
   (planted-fact long docs, GitHub repos, agent trajectories) built by the agent's own
   teacher, with eval repos strictly excluded.
4. **Ship a verified artifact.** Best config → merge/export → K-Quant with
   long-context imatrix → metadata/parity/DFlash checks → 32 GB qualification
   (analytic fit: 22.1 GB @512k F16; on-device numbers pending RTX 5090).

## Repo map

`GOAL.md`→`PLAN.md`→`MODEL.md` (spec) · `STATUS.md` (live) · `docs/` (reports, benchmarks,
deliverables audit) · `evals/harness/` (benchmarks) · `src/muse_longctx/` (corpus,
sampler, trainer) · `scripts/` (queues, export pipeline) · `AGENTS.md` (operating rules).

## Status

§0–§2, §5–§6 complete; §3 baseline ≤128k reported, >128k grids, §4 arm sweep, gated §7
training and the §8/§11 export chain running armed end-to-end; §12 awaits hardware.
See `STATUS.md` for current interim results.
