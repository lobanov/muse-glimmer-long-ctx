# PLAN — Muse Glimmer 30B 512k Context Extension

> Revision 2 — updated with verified model facts and engine-compatibility research (Aug 2026).
> See `MODEL.md` for the full architecture reference. Key verified facts that shaped this revision:
> - Glimmer has **2 KV heads** (`num_key_value_heads: 2`, 16× GQA sharing) — the 512k F16 KV cache is only ~7 GB, so the original memory concern is resolved.
> - Glimmer ships a native attention-temperature knob (`qk_scale_factor: 3.87` after QK-RMSNorm) — a new zero-shot experiment arm.
> - The model is a **VLM** (28B text decoder + 2B vision tower) — training and GGUF export must handle the multimodal packaging.
> - Day-0 support: transformers v5, llama.cpp (build ≥ 10353), vLLM, SGLang (branch). TurboQuant KV exists in forks (RTX 5090-validated) and natively in vLLM.
> - **GLM-5.2** (753B MoE, MIT, 1M context) is the synthetic-data teacher, via Z.ai API.

---

## 0. Compatibility & Memory Spike (new, do first)

> **RTX 5090 status: not yet available.** All §0 work runs on the DGX Spark as a **close
> approximation**: component memory sizes (weights, KV cache, compute buffers) are
> hardware-independent and transfer directly to the 5090 (same CUDA backend, same
> sm_120/121 family, same GGUF); the 32 GB fit is then decided **analytically** from measured
> components. Throughput does *not* transfer (5090 has ~6× GB10 memory bandwidth) — treat
> all Spark speed numbers as lower bounds. Deferred items are listed at the end of §0.

Before any benchmark or training run:

1. **Engine bring-up**
   - transformers ≥ 5.15 (model requires `transformers_version: 5.15.0.dev0`); pin exact version.
   - llama.cpp build ≥ **10353** (Meta's requirement for the official GGUFs); CUDA **12.9+** toolchain — 12.8 lacks sm_121 (verified: `nvcc fatal: compute_121`), and 13.1 has a known MMQ segfault in TurboQuant forks.
   - vLLM with `--model-impl transformers --tool-call-parser muse_glimmer --reasoning-parser muse_glimmer`.
2. **Parity check**: stock BF16 (HF) vs official `Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf` on a RULER subset at 32k–128k. This is the quant-noise floor for all later comparisons.
3. **iSWA verification**: load the GGUF at 128k and confirm in llama.cpp logs that the 39 sliding-window layers get a *window-sized* SWA cache, not full-context cache (`--swa-full` must NOT be set). Expected fixed SWA cost ≈ 82 MB.
4. **KV memory model — measured on the Spark, decided analytically for 32 GB**: global-layer KV is 13,312 B/token at F16 (theory):
   | Context | F16 KV | Q8_0 KV | turbo3 (~4.9×) |
   |---|---:|---:|---:|
   | 128k | 1.75 GB | 0.9 GB | ~0.36 GB |
   | 256k | 3.5 GB | 1.75 GB | ~0.7 GB |
   | 512k | 7.0 GB | 3.5 GB | ~1.4 GB |
   | 1M | 14 GB | 7 GB | ~2.9 GB |
   Measure actual per-component sizes with llama.cpp buffer reports on the GB10 (`CUDA0 model buffer`, `CUDA0 KV buffer`, `CUDA0 compute buffer`) at `-c` = 128k/256k/384k/512k, F16 and Q8_0 KV. The 5090 fit is then: `weights + KV + compute buffers ≤ ~31 GB` (usable VRAM minus ≥1 GB margin; `CUDA_Host` buffers are pinned host RAM, not VRAM). Record measured totals in CONTRIBUTING.md §7.
5. **TurboQuant fork build** (optional, pre-staged, can also be approximated on the Spark): `Madreag/turbo3-cuda` is validated on RTX 5090 (sm_120, FA required, turbo3 only); GB10 is sm_121 — same Blackwell family, valid for correctness/memory approximation, not for performance. Also available: `TheTom/llama-cpp-turboquant` (turbo3+turbo4). Watch for the Blackwell CMake gotcha: stale `GGML_CUDA_FORCE_CUBLAS=ON` in the cache costs ~50% decode — always configure from a clean build dir.

**Go/no-go gates** (numeric, decided before Phase 3):
- Stock GGUF at 128k within ~2 points of BF16 on the parity set → quant artifact acceptable as baseline.
- If measured components (weights + KV + compute buffers) sum to > ~31 GB for F16-KV 512k → fall back to Q8_0 KV (upstream llama.cpp, zero forks) and treat turbo3 as upside.

**Deferred (awaits RTX 5090 hardware):** on-device 32 GB fit validation; on-device throughput/latency qualification (prefill/decode at 128k–512k); TurboQuant sm_120 fork validation on-target. These move to §12 when the GPU arrives; Spark measurements above stand in until then.

---

## 1. Establish the Reproducible Environment

Use:

- NVIDIA NGC PyTorch container on DGX Spark
- PyTorch, **transformers v5** (≥ 5.15), PEFT, bitsandbytes, Accelerate, TRL (Meta ships Glimmer TRL examples)
- Hugging Face Datasets / PyArrow / Parquet
- **vLLM as primary research-inference engine** (mainline day-0 support, fastest prefill at ≥256k, native TurboQuant KV via `--kv-cache-dtype turboquant_4bit_nc` with per-layer skip)
- llama.cpp for final GGUF/K-Quant deployment (and for TurboQuant via fork, see §0)
- SGLang only if a concrete need emerges (its Glimmer support is branch-based, `muse-glimmer` branch)

Pin all versions and commits once a known-good Glimmer configuration is established. Record llama.cpp build numbers, fork commits, and CUDA toolkit versions.

Avoid introducing DeepSpeed, FSDP, Axolotl, LLaMA-Factory or NeMo unless later experiments demonstrate a concrete need.

---

## 2. Build the Evaluation Harness First

Before training, create a common runner and normalized result format.

Evaluate at:

- 32k, 64k, 128k, 192k, 256k, 384k, 512k

Where practical, vary evidence position across approximately:

- 0%, 10%, 25%, 50%, 75%, 90%, 100%

Integrate:

1. RULER
2. NoLiMa
3. LongCodeBench / LongCodeQA
4. LongSWE-Bench
5. LongBench v2
6. ∞Bench
7. HELMET
8. custom agentic-memory tests

Store results in a common Parquet schema so configurations can be compared directly.

**Sampling and template controls** (all runs, all engines — otherwise results are not comparable to community reports):

- temperature 1.0, top-p 0.95, top-k 64 (Meta's recommended defaults)
- fix `reasoning_strength` explicitly (e.g., low) in every chat-template call; log it
- greedy decoding (temp 0) only for deterministic parity checks, never for capability scores

**Robustness**: report confidence intervals over ≥ 3 seeds / instance resamples for RULER-style tasks at extreme lengths (instance counts are small and variance is high). Include abstention cases (needle absent → expected "I don't know"). Log prompt-ingestion (prefill) wall-clock at every length — on a 5090, 512k prefill takes minutes and matters for the agentic use case.

Keep all evaluation benchmark repositories and examples strictly excluded from training data.

---

## 3. Establish the Stock Glimmer Baseline

Run unmodified Glimmer beyond its nominal 128k context limit where the runtime permits.

Test: 128k, 192k, 256k, 384k, 512k (and a 1M probe if memory allows — see below).

Measure:

- task accuracy
- retrieval position sensitivity
- multi-hop degradation
- code/repository performance
- perplexity where useful (note: `final_logit_softcapping: 20.0` — use a compatible PPL implementation)
- prompt-ingestion latency, peak memory, decode speed

Prior expectation to test, not assume: community reports (r/LocalLLaMA, Aug 2026) claim verified needle retrieval at **1M context zero-shot**, and Meta publishes a Beam 128K score of 65.1. The NoPE-global design may extrapolate substantially without training. If stock@512k is already within a few points of stock@128k on RULER/NoLiMa, the project shifts from "teach extrapolation" to "strengthen, qualify, and deploy" (per the Decision Rule) — write down that threshold now (suggested: ≥ 85% relative retention on retrieval tasks → training becomes optional/targeted).

---

## 4. Create Zero-Shot Extension Configurations

Two arms, both config-only, no training. Order matters — run (a) first.

### (a) Attention-temperature sweep on the global NoPE layers — primary hypothesis

Glimmer applies RMSNorm to every Q/K head and then multiplies queries by `qk_scale_factor` (3.87), which Meta describes as an inverse softmax temperature. Attention-entropy growth in NoPE layers under distractor load is the predicted failure mode at 256k–512k; this knob targets it directly (cf. Wu et al. 2024, NoPE length generalization via attention temperature).

- Sweep `qk_scale_factor` ≈ 3.87 → 4.1 / 4.3 / 4.6 / 5.0 (≈ 1.05–1.3× logit sharpening) on the 13 global layers.
- Also verify early (§0 spike): does `convert_hf_to_gguf.py` carry `qk_scale_factor` and `layer_rope_theta` into GGUF metadata, so config-only changes survive the deployment path?
- If per-layer values are supported, prefer tuning only global layers (local SWA layers never see > 2048 relative distance).

### (b) YaRN-4 — kept as a control, expected near-inert

Configure: max context 524,288; YaRN factor 4; original context 131,072; local RoPE theta (500,000) retained; standard YaRN beta/scaling defaults. Apply only through the existing RoPE mechanism of the SWA layers; do not force rotary embeddings into NoPE global layers.

Rationale for "control" status: with θ = 500k and a 2,048-token window, RoPE in the local layers operates at tiny relative distances; rescaling its frequencies should change almost nothing. Run it anyway — it is cheap, and a nonzero result would be an interesting finding.

Run the full baseline evaluation for both arms. Compare: **stock vs qk-scale sweep vs YaRN-4**. This isolates benefit or harm from each mechanism before any training.

---

## 5. Prepare the Training Corpus

Target mixture:

| Component | Approx. share |
|---|---:|
| Whole code repositories | 35% |
| Synthetic long-context reasoning | 30% |
| Long natural documents | 15% |
| Coding-agent trajectories | 10% |
| Short-context replay | 10% |

### Teacher: GLM-5.2

- `zai-org/GLM-5.2` — 753B MoE (~40B active), **MIT license**, 1M-token context, 131k output, trained specifically for long-horizon coding-agent scenarios. Ideal teacher for this corpus.
- Too large to self-host on the Spark (~380 GB even at 4-bit) → generate via **Z.ai API** (GLM-5.2 API priced same as GLM-5.1). Design for API throughput: batched templates, full caching of prompts/completions, seeds and params logged.
- Every generated task carries machine-checkable ground truth (planted needles, compiled tests, deterministic answer keys). Add a teacher-verification pass: regenerate or cross-check answers so flawed generations are filtered, not hand-repaired.
- Watch for a smaller GLM-5.2 open variant (e.g., Air-class); if one ships, local generation on the Spark becomes viable for the bulk volume, with 5.2 API reserved for the hardest multi-hop/agentic items.

### Repository Data

Use repository-level samples from sources such as The Stack v2 / Software Heritage.

Prefer repositories spanning: 32–64k, 64–128k, 128–256k, 256–512k tokens.

Generate tasks requiring evidence across multiple files and distant locations.

Strictly exclude repositories used by evaluation suites such as LongCodeBench, RepoQA and SWE-bench-derived held-out sets.

### Synthetic Long-Context Data

Generate tasks analogous in capability to RULER and NoLiMa, without copying benchmark examples or templates.

Include: single retrieval; multi-needle retrieval; conflicting facts; variable/entity tracking; multi-hop chains; aggregation/counting; set intersection; chronological reconstruction; semantic retrieval with little lexical overlap; **abstention** (needle absent).

Control both task difficulty and evidence position.

### Natural Long Documents

Limited amounts of: books, technical manuals, scientific papers, related-document collections, documentation corpora. Prefer tasks requiring cross-section synthesis rather than simple extraction.

### Coding-Agent Trajectories

Generate tool-using coding sessions over training-only repositories (GLM-5.2 as the agent; the sessions themselves are the data).

Retain: tool calls, tool results, failed hypotheses, tests, intermediate discoveries, long command output.

Construct tasks where later decisions depend on information observed tens or hundreds of thousands of tokens earlier.

### Short Replay

Retain ~10% high-quality ordinary instruction/coding data to reduce regression.

---

## 6. Build a Custom Context/Position Sampler

Implement a small reusable component producing:

- `input_ids`, `position_ids`, `labels`, `loss masks`, `evidence positions`, `physical sequence length`, `virtual context length`

Support:

1. normal positions
2. uniform positional offsets
3. randomized segments
4. PoSE-style skipped positions
5. Randomized-YaRN-style virtual ranges
6. genuine long sequences

**Scope note (architecture-specific):** for Glimmer, modes 2–5 train the local RoPE layers' absolute-position robustness only; the 13 global NoPE layers have no position IDs, and the SWA layers rarely attend beyond 2,048 relative distance. Expect virtual-position training to be close to inert for this model; genuine long sequences (mode 6) carry nearly all of the signal. Keep the modes implemented for ablation completeness, but do not budget majority training time to them (see §7 mixture, revised below).

This component remains independent from the trainer.

---

## 7. Run the First Adaptation Experiment

Start with QLoRA rather than full fine-tuning.

Initial configuration:

- 4-bit NF4 base, BF16 compute
- LoRA rank 16–32
- gradient checkpointing
- single-process Accelerate
- attention projections first: `q_proj`, `k_proj`, `v_proj`, `o_proj`
- **VLM handling**: load the full `MuseGlimmerForConditionalGeneration`, attach LoRA only to the text-decoder submodule, keep the vision tower and projector frozen. Text-only training data; no image/video tokens.
- Watch `final_logit_softcapping` compatibility in the trainer (it is part of the forward pass, not optional).

Pre-staged fallback arm: **BF16-base + LoRA** (no NF4). Meta's own TRL experiments needed ~80 GB for BF16 LoRA; the Spark's 128 GB unified memory accommodates it (slowly). Switch to this arm if the first QLoRA run shows weak lift over zero-shot — NF4 base noise can compound with long-context difficulty.

Training-length mixture (revised for this architecture — genuine length dominates):

- **55–70%: genuine 96–256k sequences** (the distractor load on the 13 global layers is the thing being trained)
- 10–20%: genuine 32–64k sequences (cheap volume, still real attention load)
- 10–15%: 32–64k physical sequences with virtual positions to 512k (ablation coverage; expected low yield)
- 10–20%: ordinary short-context replay

If Spark throughput makes 256k sequences impractical at volume, train genuine at 128–256k and evaluate extrapolation to 512k — legitimate here precisely because positions are not the bottleneck for the global layers.

---

## 8. Evaluate the First Trained Model

Compare states (all with the same length buckets and benchmark subsets):

1. stock Glimmer
2. best zero-shot configuration (qk-scale and/or YaRN, from §4)
3. YaRN-4/qk-scale + QLoRA adaptation (whichever config the trainer used)

Primary questions:

- Does the zero-shot temperature knob help at 256k–512k?
- Does training improve on the best zero-shot state?
- Is retrieval failure positional or primarily attention/selectivity related?
- What capability is lost at ≤ 128k?
- Where does degradation become steep?

---

## 9. Perform Targeted Ablations

Only after the first end-to-end result:

### LoRA location

Compare:

- local RoPE (SWA) layers only
- global NoPE layers only
- all attention layers

This reveals whether the limiting factor is positional adaptation or global retrieval/selectivity. (Prior: global layers only ≈ all layers ≫ local only.)

### LoRA capacity

Compare as needed: rank 8 / 16 / 32 / 64.

### Attention-temperature refinement

- fixed `qk_scale_factor` sweep values (from §4a) vs.
- learnable per-head or per-global-layer logit scale trained alongside LoRA

Caveat: a learnable logit scale is not expressible as a plain weight delta — it will not survive merge→GGUF unless exported as config/metadata. Decide early whether that export path exists; if not, restrict to fixed config values that GGUF carries natively.

### Training-position strategy

Compare: genuine long sequences only; virtual positions only; mixed strategy. (Expected: genuine ≫ mixed ≫ virtual.)

### YaRN factor

If it moved the needle at all in §4: 2× / 256k, 3× / ~384k, 4× / 512k.

The goal is not the largest nominal context; it is the strongest useful context under the deployment constraint.

---

## 10. Add Training Only Where the Diagnostics Indicate

If local positional errors dominate (unlikely given the 2,048 window):

- refine YaRN parameters; increase position-randomized training; concentrate LoRA capacity on local attention layers

If global retrieval/selectivity dominates (expected):

- increase multi-needle and distractor-heavy data
- increase genuine long-context examples
- concentrate adaptation on global NoPE attention layers
- revisit the attention-temperature knob (fixed or learnable)
- generate harder semantic and multi-hop retrieval tasks

If short-context regressions appear:

- increase replay proportion; reduce LoRA rank; reduce learning rate or training duration; constrain adaptation to fewer modules

---

## 11. Merge and Export the Best Adapter

Once a checkpoint wins:

1. reload the original Hugging Face model (full multimodal checkpoint)
2. merge the LoRA adapter into the **text decoder only**; vision tower, projector, and tokenizer untouched
3. validate the merged BF16 checkpoint (parity vs adapter-on-base at 128k on the RULER subset)
4. export to GGUF via `convert_hf_to_gguf.py`; **verify metadata survived**: `layer_types`/SWA flags, `layer_rope_theta` (NoPE pattern), `qk_scale_factor`, sliding window, chat template
5. quantize to the target ~17 GB K-Quant format with an **importance matrix generated from long-context code + retrieval-style calibration data** (not the default set)
6. **DFlash drafter check**: the stock `dflash-*.gguf` drafter must still load against the requantized model (`--spec-type draft-dflash`); measure draft acceptance rate vs the stock model — a big drop signals the merged model drifted off-distribution for the drafter
7. keep the official `mmproj` projector file loadable alongside the new GGUF

Do not assume BF16 evaluation results transfer perfectly through quantization. Run a cheap quant-parity mini-suite (128k RULER subset + short bench, BF16-merged vs GGUF) before any full 512k GGUF evaluation.

---

## 12. Qualify the Final 32 GB Deployment Artifact (RTX 5090)

Run the final GGUF in llama.cpp on the 5090.

Test: 128k, 256k, 384k, 512k.

Measure:

- total VRAM, KV-cache memory (confirm iSWA engagement in logs)
- prompt-processing speed, decode speed — with and without DFlash speculative decoding (drafter memory vs context trade-off must be measured at 512k)
- benchmark accuracy, stability, quantization regression

KV strategy (in order):

1. **F16 KV** — primary target; expected ~7 GB at 512k, fits
2. **Q8_0 KV** — fallback; upstream llama.cpp, zero forks, ~3.5 GB at 512k
3. **turbo3 (TurboQuant fork)** — headroom/1M enabler; ~1.4 GB at 512k. Two mandatory checks before trusting it:
   - Glimmer's QK-RMSNorm should neutralize the K-norm-disparity failure mode seen on Qwen-class models, and head_dim 128 is the validated regime — but verify with needle tests at full depth, not just PPL (community REFRACT data shows turbo3 V-cache can degrade generation trajectories on some dense models while PPL looks fine)
   - prefer asymmetric configs if quality dips (e.g., q8_0 K + turbo3 V is not the standard fix — test turbo3 K+V first, then q8_0/turbo3 mixes)

---

## 13. Optional 1M Follow-On

Only attempt 1M after 512k is robust.

Memory is no longer the binding constraint: turbo3 KV ≈ 2.9 GB at 1M (14 GB at F16 also borderline-fits with 17 GB weights). The binding constraint is **compute in the 13 global full-attention layers** (quadratic prefill, linear-per-token decode over 1M entries) plus prefill wall-clock on a single 5090.

Potential configuration: same ~17 GB weights, turbo3 KV, extended YaRN/position training only if §4 showed any positional effect.

Treat 1M as a separate research target; expect decode speed and prefill time, not VRAM, to be the practical limits.

---

## 14. Final Deliverables

Produce:

1. reproducible environment specification (incl. llama.cpp/fork build numbers, CUDA version, transformers v5 pin)
2. training-data generation scripts (GLM-5.2 API pipeline with verification passes)
3. context/position sampler
4. QLoRA training configuration (+ BF16-LoRA fallback config)
5. benchmark harness with pinned sampling/template controls
6. baseline, zero-shot (qk-scale, YaRN), and ablation results
7. merged Hugging Face checkpoint
8. ~17 GB K-Quant GGUF (with long-context imatrix, verified metadata, DFlash-compatible)
9. 32 GB / 512k deployment configuration (RTX 5090)
10. concise report describing: effective context, memory use, throughput, benchmark results, regressions, remaining limitations

## Decision Rule

The final model should be selected on **useful long-context performance under the 32 GB deployment constraint**, not on nominal context length alone.
