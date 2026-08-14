# Phase 4 Report — Zero-Shot Extension Arms (PLAN §4)

Status: **PREP COMPLETE — arms built & validated; runs queued behind §3 baseline (GPU is
serial on the Spark).** Artifacts: `outputs/arms/<name>/` (config-only; safetensors
symlinked — no 60 GB copies), generator `scripts/make_zero_shot_arm.py`, vLLM service
templated (`VLLM_MODEL`, `VLLM_MAX_MODEL_LEN` env; `../outputs/arms:/arms:ro` mount).

## Arm (a) — attention-temperature sweep (primary hypothesis)

`qk_scale_factor` swept **globally**: 3.87 (stock) → 4.1 / 4.3 / 4.6 / 5.0.

- Implementation fact (verified in `modeling_muse_glimmer.py`):
  `query_states = self.qk_norm(query_states) * self.qk_scale_factor` — a **single scalar
  applied on every layer**; per-layer values are NOT supported by the HF implementation, so
  the plan's "prefer tuning only global layers" is unavailable on this path. Tolerable:
  SWA layers attend ≤ 2,048 relative distance; a 1.05–1.3× logit sharpening there is part of
  what the sweep measures (llama.cpp path can also scope it via re-conversion if needed).
- Deployment-path fact (§0/§4 spike, commit `2dc2c10`): the GGUF has **no** qk metadata —
  the converter absorbs qk_scale_factor into synthesized `attn_q_norm` weights. Config-only
  sweeps live on the HF/vLLM path; the winning value reaches GGUF via §11 re-convert+re-quant
  (or, for a *learned* scale, by writing it back into `attn_q_norm` at merge time — that
  tensor is a plain weight, so the §9 learnable-scale export path exists).

## Arm (b) — YaRN-4 (control)

`text_config.rope_parameters = {rope_type: yarn, factor: 4.0,
original_max_position_embeddings: 131072, rope_theta: 500000.0}` +
`max_position_embeddings: 524288`.

- **Gotcha caught & fixed**: transformers' yarn init reads `rope_theta` from the
  `rope_parameters` dict and defaults it to 10,000 — without the explicit 500,000 the SWA
  layers' frequencies would be silently rewritten. Verified in serialized arm config.
- YaRN's native `attention_scaling` applies via the standard rope machinery; only the 39
  RoPE/SWA layers consume position embeddings (`position_embeddings=None` on the 13 NoPE
  layers), exactly as the plan requires ("YaRN through the existing RoPE mechanism only").
- Expected near-inert (θ=500k at ≤2,048 window) — run anyway; cheap and informative.

## Validation

`AutoConfig.from_pretrained` round-trip asserts per arm: qk value, rope_type, factor,
theta=500k, max_position_embeddings = **524,288 for all research arms** (yarn4 via its
scaling config; qk arms + stock-524k via the same mechanical window extension —
updated 2026-08-14 post-validation so every arm serves at `--max-model-len 524288`;
the knob remains each arm's ONLY variable). All pass.

## Run plan (when §3 grid frees the GPU)

Per arm: `VLLM_MODEL=/arms/<arm> docker compose --profile inference up -d vllm`
(~10 min cold start), then `evals/harness/run_eval.py --config-label <arm>` on the §2 grid
at 128k–512k first (decision-relevant), full grid if the arm is competitive vs stock.
