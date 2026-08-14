#!/usr/bin/env python3
"""PLAN §5→§7 bridge: corpus documents + samples → tokenized TrainingSample jsonl.

Row format consumed by src/muse_longctx/train_qlora.py:
  {input_ids, position_ids, labels, mode, physical_len, virtual_len, meta}

- Chat rendering uses the OFFICIAL Glimmer template with the same
  chat_template_kwargs used at inference (reasoning_strength=low) — train/serve parity.
- Loss mask: assistant answer tokens only (prompt fully masked, -100).
- Position mode: `genuine` (PLAN §7: genuine-length data carries the signal for this
  architecture; virtual modes are ablation-only and applied by re-serializing with a
  different layout — not needed for v1).
- Long docs (up to 512k) serialize whole: the point is real distractor load on the 13
  NoPE-global layers.

Usage (dev container — tokenizer lives there):
  python3 src/muse_longctx/corpus/serialize.py --doc outputs/corpus/synth_v1/doc_000.jsonl \
      --out outputs/corpus/synth_v1/doc_000.samples.jsonl --validate
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from muse_longctx.position_sampler import build_training_sample, sample_positions  # noqa: E402

MODEL = "meta-models/Muse-Glimmer-30B"


def get_tokenizer():
    from transformers import AutoProcessor
    return AutoProcessor.from_pretrained(MODEL).tokenizer


def render(tok, user_prompt, answer):
    """Full = template(user) + answer(+eos); prompt_ids = template(user) alone."""
    base = tok.apply_chat_template([{"role": "user", "content": user_prompt}],
                                   tokenize=True, add_generation_prompt=True,
                                   chat_template_kwargs={"reasoning_strength": "low"})
    if hasattr(base, "input_ids"):  # transformers v5 returns BatchEncoding on tokenize=True
        base = base["input_ids"]
    if base and isinstance(base[0], list):
        base = base[0]
    # assistant turn: plain content answer + eos (reasoning channel empty at low strength
    # is the serve-time behavior we train toward)
    ans_ids = tok(answer + tok.eos_token, add_special_tokens=False)["input_ids"]
    return base, ans_ids


def serialize(doc_path, out_path, validate=False):
    tok = get_tokenizer()
    rows = [json.loads(l) for l in open(doc_path) if l.strip()]
    doc = rows[0]                       # {"body", "ledger", "target_tokens"}
    samples = rows[1:]                  # {"prompt","answer","question","axis"}
    n_out = 0
    with open(out_path, "w") as f:
        for i, s in enumerate(samples):
            prompt_ids, ans_ids = render(tok, s["prompt"], s["answer"])
            ids = prompt_ids + ans_ids
            loss_spans = [(len(prompt_ids), len(ids))]
            ts = build_training_sample(ids, loss_token_spans=loss_spans,
                                       evidence_token_spans=[(len(prompt_ids) - 1,
                                                              len(prompt_ids))],
                                       layout=sample_positions("genuine", len(ids)))
            row = {"input_ids": ts.input_ids, "position_ids": ts.position_ids,
                   "labels": ts.labels, "mode": "genuine",
                   "physical_len": ts.physical_len, "virtual_len": ts.virtual_len,
                   "meta": {"axis": s["axis"], "question": s["question"],
                            "answer": s["answer"], "source": os.path.basename(doc_path)}}
            f.write(json.dumps(row) + "\n")
            n_out += 1
            if validate and i == 0:
                assert all(l == -100 for l in ts.labels[:len(prompt_ids)])
                assert sum(1 for l in ts.labels if l != -100) == len(ans_ids)
                assert ts.position_ids == list(range(len(ids)))
                print(f"[validate] prompt={len(prompt_ids)} answer={len(ans_ids)} "
                      f"total={len(ids)} loss-on-answer OK, genuine positions OK")
    print(f"{doc_path}: {len(samples)} samples -> {n_out} rows -> {out_path}")
    return out_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--doc", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--validate", action="store_true")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    serialize(a.doc, a.out, a.validate)
