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

## Blockers / next steps

1. §5 corpus (Z.ai API key) — escalated; interim options (The Stack v2 slice + short
   replay) flagged to the owner in PLAN §5 status note.
2. `--dry-run` on GPU (needs vLLM sidecar down or a memory window — queued behind §3).
3. First real run config per plan: r16–32, attention-only, 55–70% genuine 96–256k,
   10–20% genuine 32–64k, 10–15% virtual-position ablation, 10–20% short replay.
