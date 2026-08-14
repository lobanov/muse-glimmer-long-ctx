#!/usr/bin/env python3
"""PLAN §2 — LongCodeQA integration (Steefano/LCB, LongCodeBench @1M contexts suite).

Repository-scale code comprehension in MC form: full repo text + question, answer with
the letter only. Buckets: 32K/64K/128K/256K/512K/1M (official; our eval grid stops at 512K).
Official prompt is shipped per instance (`prompt` field) and used verbatim; scoring = the
official `correct_letter` (first standalone letter, mirroring the official protocol).

`target_ctx` maps to the bucket <= target (e.g. 192000 -> 128K bucket); `depth` ignored
(fixed by instance). Instances resampled per rep via the harness seed.

License: LCB data — MIT (per repo); eval-only, never training data. Repos appear in
data/exclusions/eval_repos.json (exclusion verified during list build).

Register: --plugin longcodeqa     Use: --tasks longcodeqa --ctx 32000,64000,128000,256000,512000
"""
import json
import os
import random
import re
import zipfile

from huggingface_hub import hf_hub_download

DATASET = "Steefano/LCB"
BUCKETS = [(32_000, "LQA/32K.json"), (64_000, "LQA/64K.json"), (128_000, "LQA/128K.json"),
           (256_000, "LQA/256K.json"), (512_000, "LQA/512K.json"), (1_000_000, "LQA/1M.json")]
_LETTER = re.compile(r"\b([A-E])\b")
_cache = {}


def ensure_data():
    if "buckets" not in _cache:
        z = zipfile.ZipFile(hf_hub_download(DATASET, "LongCodeQA.zip", repo_type="dataset"))
        _cache["buckets"] = {ctx: json.loads(z.read(name)) for ctx, name in BUCKETS}
    return _cache


def build_longcodeqa(rng, target_tokens, depth):
    d = ensure_data()
    usable = [ctx for ctx, _ in BUCKETS if ctx <= target_tokens]
    assert usable, f"no LQA bucket <= {target_tokens}"
    bucket = d["buckets"][max(usable)]
    inst = bucket[rng.randrange(len(bucket))]
    return inst["prompt"], {"gold": inst["correct_letter"], "repo": inst["repo"],
                            "bucket_ctx": max(usable), "n_bucket": len(bucket),
                            "depth_ignored": True}


def score_longcodeqa(content, meta):
    c = (content or "").strip()
    m = _LETTER.search(c)
    if not m:
        return 0.0, f"no-letter (gold {meta['gold']})"
    got = m.group(1)
    return (1.0, "hit") if got == meta["gold"] else (0.0, f"got {got} want {meta['gold']}")


def register(tasks_mod):
    tasks_mod.TASKS.append("longcodeqa")
    tasks_mod.BUILDERS["longcodeqa"] = build_longcodeqa
    orig_score = tasks_mod.score

    def score(task, content, meta):
        if task == "longcodeqa":
            return score_longcodeqa(content, meta)
        return orig_score(task, content, meta)

    tasks_mod.score = score
    total = sum(len(v) for v in ensure_data()["buckets"].values())
    print(f"longcodeqa plugin registered: {total} MC instances across 6 official buckets")


if __name__ == "__main__":
    d = ensure_data()
    for ctx, items in sorted(d["buckets"].items()):
        hard = sum(1 for x in items if x.get("is_hard"))
        print(f"bucket {ctx:>7,}: {len(items):3d} instances ({hard} hard)")
