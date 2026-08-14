# Phase 7 Report — QLoRA Trainer (PLAN §7)

Status: **SKELETON COMPLETE — `src/muse_longctx/train_qlora.py`; awaits §5 corpus (blocked
on Z.ai key) and a free GPU for `--dry-run`.** Validated so far: py_compile, class name
`MuseGlimmerForConditionalGeneration` import, module-path verification against
`model.safetensors_index.json`.

## What is implemented

- **VLM handling** per plan: loads full `MuseGlimmerForConditionalGeneration`;
  LoRA scoped by regex `language_model\..*self_attn\.(q|k|v|o)_proj` → asserts exactly
  52×4 = 208 injected modules and **zero** under `vision_*` (vision tower parents are
  `attn.*`, not `self_attn.*` — verified tensor names, MODEL.md §2.5).
- **Modes**: `qlora` (NF4 + double-quant, BF16 compute) and `bf16_lora` fallback arm.
- **Softcapping guard**: asserts `text_config.final_logit_softcapping == 20.0` at load —
  a packaging change cannot silently drop it (it is part of forward; nothing to reimplement).
- **Position-aware training**: collator carries `position_ids` from the §6 sampler's
  `TrainingSample` rows (jsonl); labels already −100-masked.
- **`--config-override` json**: bridges §4/§9 fixed config values (e.g.
  `{"qk_scale_factor": 4.3}`) into training — same code path as the arm generator.
- **`--dry-run`**: full assembly + one forward/backward on random ids (seq 512) +
  trainable-parameter table, no save. This is the next validation step once the GPU frees.

## Deliberately minimal

Custom Accelerate loop (no TRL/Axolotl/etc. — PLAN §1 framework restraint). Gradient
checkpointing + `enable_input_require_grads`. Adapter-only save.

## First-run configuration (PLAN §7 / §14.4 — decided)

```bash
# arm-config bridge: train under the §4-winning qk value (or stock knobs)
docker exec muse-glimmer-long-ctx-dev-1 python3 src/muse_longctx/train_qlora.py \
  --data outputs/corpus/train_v1/train.jsonl \
  --out outputs/adapters/run1 --mode qlora \
  --lora-rank 32 --lora-scope all --lr 1e-4 \
  --micro-batch 1 --grad-accum 8 --seq-bucket 131072 \
  --config-override '{"qk_scale_factor": <winner or omit>}'
```

- QLoRA first (NF4 + double-quant, BF16 compute); `bf16_lora` fallback arm if lift over
  zero-shot is weak (PLAN §7 pre-staged fallback; Spark 128 GB accommodates ~80 GB).
- `--lora-scope all` for run 1; §9 ablation then runs `global` (prior: ≈ all ≫ local).
- Corpus: `train_v1` manifest governs the genuine-length mixture; virtual-position rows
  enter only at §9 via re-serialization with sampler modes 2–5.
- Pre-flight: `--dry-run` (one fwd/bwd @ seq 512 + trainable-param table) on a GPU window
  before the first long run.
