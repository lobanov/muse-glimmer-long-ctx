#!/usr/bin/env python3
"""MRCR v2 (google-deepmind/eval_hub, Apache-2.0) as a harness plugin — goal d56ed95d.

Uses the OFFICIAL public dataset (storage.googleapis.com/mrcr_v2 CSVs; downloaded by
scripts/mrcr_download into cache/mrcr_v2/): each row = full transcript prompt
(`queries`, ends with the follow-up question) + gold (`answer` = random hash + the
target text) + Gemini-tokenizer `context_len`. Recipe/scorer follow eval_hub's
run_evaluation defaults (strict exact-match on the full gold string; lenient
containment recorded in detail only).

Tasks: mrcr2 (2-needle), mrcr4 (4-needle). Length control: rows filtered to a
±15% band around target_ctx on the dataset's context_len (Glimmer ptok recorded
per-row server-side; tokenizer drift noted in docs). Eval-only; never training data.

Register: --plugin mrcr    Use: --tasks mrcr2,mrcr4
"""
import csv
import json
import os
import pickle
import random
import sys

CACHE = os.path.join(os.path.dirname(__file__), "..", "..", "cache", "mrcr_v2")
TASKS = ("mrcr2", "mrcr4")
BUCKETS = [  # (lo, hi, filename) — files are near-pointwise at bucket max (~131k/262k/524k)
    (0, 131072, "mrcr_v2p1_{n}needle_in_(65536,131072)_dynamic_fewshot_text_style_fast.csv"),
    (131072, 262144, "mrcr_v2p1_{n}needle_in_(131072,262144)_dynamic_fewshot_text_style_fast.csv"),
    (262144, 10 ** 9, "mrcr_v2p1_{n}needle_in_(262144,524288)_dynamic_fewshot_text_style_fast.csv"),
]
_cache = {}


def _pool(task, target_tokens):
    """Band-filtered rows for (task, target); caches a compact pickle per band."""
    n = task[-1]
    lo, hi, pat = next(b for b in BUCKETS if target_tokens <= b[1])
    band_lo, band_hi = int(0.85 * target_tokens), int(1.05 * target_tokens)
    key = f"{task}_{band_lo}_{band_hi}"
    if key in _cache:
        return _cache[key]
    pk = os.path.join(CACHE, f"band_{key}.pkl")
    if os.path.exists(pk):
        rows = pickle.load(open(pk, "rb"))
    else:
        csv.field_size_limit(10 ** 9)
        path = os.path.join(CACHE, pat.format(n=n))
        rows = []
        with open(path, newline="") as fh:
            for row in csv.DictReader(fh):
                cl = int(row["context_len"])
                if band_lo <= cl <= band_hi:
                    rows.append({"queries": row["queries"], "answer": row["answer"],
                                 "context_len": cl})
        pickle.dump(rows, open(pk, "wb"))
    assert rows, f"no {task} rows in band [{band_lo},{band_hi})"
    _cache[key] = rows
    return rows


def _builder(task):
    def build(rng, target_tokens, depth):
        pool = _pool(task, target_tokens)
        inst = pool[rng.randrange(len(pool))]
        meta = {"gold": inst["answer"], "ctx_tokens": inst["context_len"],
                "depth_ignored": True, "num_needles": int(task[-1])}
        return inst["queries"], meta
    return build


def _score(task, content, meta):
    c = (content or "").strip()
    gold = meta["gold"].strip()   # dataset golds carry trailing transcript whitespace
    if c == gold:
        return 1.0, "exact"
    if gold in c:
        return 0.0, "gold+extra (lenient-hit)"   # official strict: extra text fails
    if gold[:64] in c:
        return 0.0, "partial-prefix (lenient-hit)"
    return 0.0, "miss"


def register(tasks_mod):
    for t in TASKS:
        tasks_mod.TASKS.append(t)
        tasks_mod.BUILDERS[t] = _builder(t)
    orig_score = tasks_mod.score

    def score(task, content, meta):
        if task not in TASKS:
            return orig_score(task, content, meta)
        return _score(task, content, meta)

    tasks_mod.score = score
    print("mrcr plugin registered:", ", ".join(TASKS))


if __name__ == "__main__" and sys.argv[1:] == ["selftest"]:
    import time
    csv.field_size_limit(10 ** 9)
    rng = random.Random(0)
    ok = True
    for task in TASKS:
        t0 = time.time()
        build = _builder(task)
        prompt, meta = build(rng, 128000, 0.5)
        gold_full = meta["gold"]
        m = {"gold": gold_full}
        s_exact, d1 = _score(task, gold_full, m)
        s_wrong, d2 = _score(task, "completely wrong response", m)
        s_extra, d3 = _score(task, gold_full + " extra tail", m)
        print(f"{task}@128k: prompt_chars={len(prompt)} ctx_len={meta['ctx_tokens']} "
              f"needles={meta['num_needles']} | exact={s_exact}({d1}) "
              f"wrong={s_wrong} extra={s_extra}({d3}) | build {time.time()-t0:.1f}s")
        ok &= s_exact == 1.0 and s_wrong == 0.0 and s_extra == 0.0
    prompt, meta = _builder("mrcr2")(random.Random(1), 262144, 0.5)
    print(f"mrcr2@262144: prompt_chars={len(prompt)} ctx_len={meta['ctx_tokens']}")
    print("SELFTEST", "PASS" if ok else "FAIL")
