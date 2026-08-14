#!/usr/bin/env python3
"""PLAN §7 — QLoRA/BF16-LoRA trainer skeleton for Muse Glimmer long-context adaptation.

VLM handling (PLAN §7): loads the full `MuseGlimmerForConditionalGeneration`, attaches
LoRA ONLY to the text decoder (`model.language_model.layers.N.self_attn.{q,k,v,o}_proj`,
verified names); vision tower / adapter / projector / lm_head stay frozen. Text-only data.

Consumes serialized `TrainingSample` rows (src/muse_longctx/position_sampler.py):
    {"input_ids": [...], "position_ids": [...], "labels": [...], "mode": "genuine", ...}
Loss mask is already encoded in labels (-100 where masked). Sequences are padded to the
bucket length inside the collator; padded labels -100, position_ids continued (masked by
attention_mask anyway).

Modes:
    qlora     4-bit NF4 base (BitsAndBytes), BF16 compute, LoRA r16-32   [first arm]
    bf16_lora full BF16 base + LoRA (Spark 128 GB unified; ~80 GB)       [fallback arm]

--dry-run: assemble everything, one forward+backward on random ids (seq len 512), print
trainable-parameter table, save nothing. Use to validate wiring before data exists.

final_logit_softcapping (20.0) is part of the model forward — nothing to do here, and we
assert it is present in the loaded text_config so a packaging change cannot silently drop it.
"""
import argparse
import json
import os
import re

import torch
from torch.utils.data import Dataset

LORA_TARGET_RE = r"language_model\..*self_attn\.(q_proj|k_proj|v_proj|o_proj)"
EXPECTED_LORA_MODULES = 52 * 4  # layers × {q,k,v,o}


def load_model(base_model: str, mode: str, config_override: dict | None):
    from transformers import AutoProcessor, BitsAndBytesConfig, MuseGlimmerForConditionalGeneration
    kwargs = {"dtype": torch.bfloat16}
    if config_override:
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(base_model)
        tc = cfg.text_config
        for k, v in config_override.items():
            assert hasattr(tc, k), f"unknown text_config key {k}"
            setattr(tc, k, v)
        kwargs["config"] = cfg
    if mode == "qlora":
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    model = MuseGlimmerForConditionalGeneration.from_pretrained(base_model, **kwargs)
    processor = AutoProcessor.from_pretrained(base_model)
    assert getattr(model.config.text_config, "final_logit_softcapping", None) == 20.0, \
        "final_logit_softcapping missing/changed — trainer assumption broken"
    return model, processor


def attach_lora(model, rank: int, alpha_ratio: float = 2.0, dropout: float = 0.05):
    from peft import LoraConfig, get_peft_model
    cfg = LoraConfig(
        r=rank, lora_alpha=int(rank * alpha_ratio), lora_dropout=dropout,
        bias="none", task_type="CAUSAL_LM", target_modules=LORA_TARGET_RE)
    model = get_peft_model(model, cfg)
    injected = [n for n, _ in model.named_modules() if ".lora_A." in n]
    assert len(injected) == EXPECTED_LORA_MODULES, \
        f"expected {EXPECTED_LORA_MODULES} LoRA modules, got {len(injected)} " \
        f"(vision leakage? names: {injected[:3]})"
    assert not any(n.startswith("vision") or ".vision_" in n for n in injected)
    return model


class SampleDataset(Dataset):
    """Serialized TrainingSample rows from jsonl (one JSON object per line)."""

    def __init__(self, path):
        self.rows = []
        with open(path) as f:
            for line in f:
                if line.strip():
                    self.rows.append(json.loads(line))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return self.rows[i]


def collate(batch, pad_id: int):
    maxlen = max(len(r["input_ids"]) for r in batch)
    input_ids, position_ids, labels, attn = [], [], [], []
    for r in batch:
        ids, pos, lab = r["input_ids"], r["position_ids"], r["labels"]
        pad = maxlen - len(ids)
        input_ids.append(ids + [pad_id] * pad)
        position_ids.append(pos + [pos[-1]] * pad)
        labels.append(lab + [-100] * pad)
        attn.append([1] * len(ids) + [0] * pad)
    return (torch.tensor(input_ids), torch.tensor(position_ids),
            torch.tensor(labels), torch.tensor(attn))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default="meta-models/Muse-Glimmer-30B")
    ap.add_argument("--data", help="jsonl of serialized TrainingSample rows")
    ap.add_argument("--out", default="outputs/adapters/run1")
    ap.add_argument("--mode", choices=["qlora", "bf16_lora"], default="qlora")
    ap.add_argument("--lora-rank", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--micro-batch", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--seq-bucket", type=int, default=131072,
                    help="max physical tokens per batch (grad checkpointing budget)")
    ap.add_argument("--config-override", help='json, e.g. {"qk_scale_factor": 4.3}')
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    override = json.loads(args.config_override) if args.config_override else None
    model, processor = load_model(args.base_model, args.mode, override)
    model = attach_lora(model, args.lora_rank)
    model.print_trainable_parameters()
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    pad_id = processor.tokenizer.pad_token_id

    if args.dry_run:
        dev = next(model.parameters()).device
        n = 512
        ids = torch.randint(10_000, 20_000, (1, n), device=dev)
        pos = torch.arange(n, device=dev).unsqueeze(0)
        lab = ids.clone()
        lab[:, :-1] = -100
        out = model(input_ids=ids, position_ids=pos, labels=lab)
        out.loss.backward()
        print(f"[dry-run] forward+backward OK; loss={out.loss.item():.4f}")
        return

    from accelerate import Accelerator
    accel = Accelerator(gradient_accumulation_steps=args.grad_accum)
    ds = SampleDataset(args.data)
    dl = torch.utils.data.DataLoader(
        ds, batch_size=args.micro_batch, shuffle=True, collate_fn=lambda b: collate(b, pad_id))
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    model, opt, dl = accel.prepare(model, opt, dl)
    os.makedirs(args.out, exist_ok=True)
    step = 0
    for _epoch in range(int(args.epochs + 0.999)):
        for input_ids, position_ids, labels, attn in dl:
            with accel.accumulate(model):
                out = model(input_ids=input_ids, position_ids=position_ids,
                            attention_mask=attn, labels=labels)
                accel.backward(out.loss)
                opt.step()
                opt.zero_grad()
            if step % 10 == 0:
                print(f"step {step} loss {out.loss.item():.4f}", flush=True)
            step += 1
    unwrapped = accel.unwrap_model(model)
    unwrapped.save_pretrained(args.out)  # adapter only (trainable params)
    print(f"adapter saved -> {args.out}")


if __name__ == "__main__":
    main()
