#!/usr/bin/env python3
"""PLAN §2 — LongBench v2 integration (THUDM/LongBench-v2, 503 MC instances, 8k-2M words).

Official-prompt faithful port: context + question + four lettered choices, answer with
the option letter directly (no CoT), scoring = first standalone A-D letter in the response
(paper's "directly answers" protocol). Instances are fixed-length; the harness's
`target_ctx` acts as an upper-length filter (instance tokens ∈ [0.5, 0.92] × target so
long targets don't collapse onto trivially short instances). `depth` is ignored (fixed
by the instance) and recorded as such.

Per-instance token counts are measured once with the Glimmer tokenizer and cached.

License: CC-BY-NC (LongBench v2) — research use; eval-only, never training data.

Register: --plugin longbench_v2     Use: --tasks longbench_v2 --ctx 32000,128000,...
"""
import json
import os
import random
import re

from huggingface_hub import hf_hub_download

DATASET = "THUDM/LongBench-v2"
CACHE = os.path.join(os.path.dirname(__file__), "..", "..", "outputs", "eval",
                     "longbench_v2_lengths.json")
PROMPT = ("Read the following context and answer the question based on it.\n\n"
          "{context}\n\nQuestion: {question}\n\n"
          "A. {a}\nB. {b}\nC. {c}\nD. {d}\n\n"
          "Answer with the option's letter from the given choices directly.")
_LETTER = re.compile(r"\b([ABCD])\b")

_cache = {}


def ensure_data():
    if "data" in _cache:
        return _cache
    p = hf_hub_download(DATASET, "data.json", repo_type="dataset")
    _cache["data"] = json.load(open(p))
    if os.path.exists(CACHE):
        _cache["lengths"] = json.load(open(CACHE))
    return _cache


def measure_lengths(tokenizer):
    d = ensure_data()
    if "lengths" in d:
        return d["lengths"]
    lengths = {}
    for i, inst in enumerate(d["data"]):
        lengths[inst["_id"]] = len(tokenizer(inst["context"], add_special_tokens=False)
                                   ["input_ids"])
        if (i + 1) % 100 == 0:
            print(f"  tokenized {i+1}/{len(d['data'])}", flush=True)
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    json.dump(lengths, open(CACHE, "w"))
    _cache["lengths"] = lengths
    return lengths


def build_longbench_v2(rng, target_tokens, depth):
    d = ensure_data()
    assert "lengths" in d, "run: python3 evals/harness/longbench_v2.py calibrate"
    pool = [x for x in d["data"]
            if 0.5 * target_tokens <= d["lengths"][x["_id"]] <= 0.92 * target_tokens]
    if not pool:  # fall back to anything that fits
        pool = [x for x in d["data"] if d["lengths"][x["_id"]] <= 0.92 * target_tokens]
    assert pool, f"no LongBench-v2 instance fits target {target_tokens}"
    inst = pool[rng.randrange(len(pool))]
    prompt = PROMPT.format(context=inst["context"], question=inst["question"],
                           a=inst["choice_A"], b=inst["choice_B"],
                           c=inst["choice_C"], d=inst["choice_D"])
    meta = {"gold": inst["answer"], "id": inst["_id"], "domain": inst["domain"],
            "sub_domain": inst["sub_domain"], "difficulty": inst["difficulty"],
            "bucket": inst["length"], "ctx_tokens": d["lengths"][inst["_id"]],
            "depth_ignored": True}
    return prompt, meta


def score_longbench_v2(content, meta):
    c = (content or "").strip()
    m = _LETTER.search(c)
    if not m:
        return 0.0, f"no-letter (gold {meta['gold']})"
    got = m.group(1)
    return (1.0, "hit") if got == meta["gold"] else (0.0, f"got {got} want {meta['gold']}")


def register(tasks_mod):
    tasks_mod.TASKS.append("longbench_v2")
    tasks_mod.BUILDERS["longbench_v2"] = build_longbench_v2
    orig_score = tasks_mod.score

    def score(task, content, meta):
        if task == "longbench_v2":
            return score_longbench_v2(content, meta)
        return orig_score(task, content, meta)

    tasks_mod.score = score
    print("longbench_v2 plugin registered: 503 MC instances, first-letter scoring")


if __name__ == "__main__":
    if __import__("sys").argv[1:] == ["calibrate"]:
        from transformers import AutoProcessor
        tok = AutoProcessor.from_pretrained("meta-models/Muse-Glimmer-30B").tokenizer
        lengths = measure_lengths(tok)
        import statistics
        vals = sorted(lengths.values())
        print(f"calibrated {len(vals)} instances: min={vals[0]:,} "
              f"med={vals[len(vals)//2]:,} max={vals[-1]:,}")
        for t in (8_000, 32_000, 128_000, 256_000, 512_000):
            n = sum(1 for v in vals if 0.5 * t <= v <= 0.92 * t)
            print(f"  target {t}: {n} instances in [0.5x, 0.92x] window")
