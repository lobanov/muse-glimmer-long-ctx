#!/usr/bin/env python3
"""PLAN §2 — ∞Bench (InfiniteBench) integration: broad >100k-context evaluation.

Curated subset with deterministic scoring (official answer fields):
    infb_kv        kv_retrieval      (500) ~50k-token UUID KV tables — exact UUID match
    infb_bookmc    longbook_choice_eng (229) ~280k-token books + MC — option-text match
    infb_codedebug code_debug        (394) ~200k-token repos, find the broken function — name match

Contexts are fixed per instance; `target_ctx` filters to instances whose Glimmer-token
context length fits [0.5, 0.92] × target (fallback: anything that fits). `depth` ignored.
Per-instance token lengths measured once, cached (outputs/eval/infbench_lengths.json).
Scoring: official containment (any gold string appears in the response; case-sensitive,
case-insensitive recorded as detail). Book MC additionally matches full option text.

License: CC-BY-NC (InfiniteBench) — research; eval-only, never training data.

Register: --plugin infbench    Use: --tasks infb_kv,infb_bookmc,infb_codedebug
"""
import json
import os
import random

from huggingface_hub import hf_hub_download

DATASET = "xinrongzhang2022/InfiniteBench"
TASKS = {
    "infb_kv": ("kv_retrieval.jsonl",),
    "infb_bookmc": ("longbook_choice_eng.jsonl",),
    "infb_codedebug": ("code_debug.jsonl",),
}
CACHE = os.path.join(os.path.dirname(__file__), "..", "..", "outputs", "eval",
                     "infbench_lengths.json")
# v3 (2026-08-17): task/id-keyed TRUE Glimmer tokenization. v1/v2 were bare-id
# keyed — ids collide across the three subsets, so each calibration pass overwrote
# the others (collision soup; v1 bookmc selections were invalid, v2 was uniform
# garbage). v3 is authoritative; validated against server-side prompt_tokens (±80).
CACHE_V3 = CACHE.replace(".json", "_v3.json")
# Optional honest-length band override for stratified runs (goal afe6584b):
#   INF_MIN_TOK / INF_MAX_TOK restrict the instance pool to a true-token band.
BAND_MIN = int(os.environ.get("INF_MIN_TOK", "0"))
BAND_MAX = int(os.environ.get("INF_MAX_TOK", "0")) or 10 ** 9
_cache = {}


def ensure_data():
    # NOTE: never seed _cache["lengths"] here — the legacy v1 CACHE is id-collision
    # soup (see _load_lengths); seeding it short-circuited v3 (bug found 2026-08-19:
    # bookmc bands2 KeyError'd on infb_bookmc/0 because the plugin silently used v1).
    if "data" not in _cache:
        _cache["data"] = {t: [json.loads(l) for l in
                              open(hf_hub_download(DATASET, f[0], repo_type="dataset"))
                              if l.strip()]
                          for t, f in TASKS.items()}
    return _cache


def measure_lengths(tokenizer):
    d = ensure_data()
    if "lengths" in d:
        return d["lengths"]
    lengths = {}
    for t, items in d["data"].items():
        for inst in items:
            lengths[str(inst["id"])] = len(tokenizer(inst["context"],
                                                     add_special_tokens=False)["input_ids"])
        print(f"  {t}: {len(items)} instances tokenized", flush=True)
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    json.dump(lengths, open(CACHE, "w"))
    _cache["lengths"] = lengths
    return lengths


def _load_lengths():
    """v3 (task/id-keyed true Glimmer tokenization) only, with chars/token sanity
    assert (English ~3.5-5.5) so a corrupt cache can never scramble lengths silently."""
    d = ensure_data()
    if "lengths" in d:
        return d["lengths"]
    if os.path.exists(CACHE_V3):
        L = json.load(open(CACHE_V3))
        for t, items in d["data"].items():
            if t == "infb_kv":
                continue  # UUID tables: ~1.6 chars/tok legitimately (not natural text)
            for inst in items[:20]:
                n = L[f"{t}/{inst['id']}"]
                ratio = len(inst["context"]) / max(n, 1)
                assert 3.0 <= ratio <= 6.5, \
                    f"v3 lengths cache fails chars/token sanity ({t} id {inst['id']}: " \
                    f"{ratio:.1f}) — recalibrate"
        _cache["lengths"] = L
        return L
    raise SystemExit("infbench_lengths_v3.json missing — recalibrate (task/id-keyed;"
                     " see scripts/infb_forensics.py history)")


def _builder(task):
    def build(rng, target_tokens, depth):
        d = ensure_data()
        L = _load_lengths()
        items = d["data"][task]
        if BAND_MIN or BAND_MAX < 10 ** 9:      # honest-length strata (goal afe6584b)
            pool = [x for x in items if BAND_MIN <= L[f"{task}/{x['id']}"] < BAND_MAX]
            assert pool, f"no {task} instance in band [{BAND_MIN},{BAND_MAX})"
        else:
            fits = [x for x in items if L[f"{task}/{x['id']}"] <= 0.92 * target_tokens]
            pool = [x for x in fits if L[f"{task}/{x['id']}"] >= 0.5 * target_tokens] or fits
            assert pool, f"no {task} instance fits target {target_tokens}"
        inst = pool[rng.randrange(len(pool))]
        prompt = f"{inst['context']}\n\n{inst['input']}"
        if task == "infb_bookmc":
            prompt += ("\n\nAnswer by repeating the full text of the correct option.")
        meta = {"gold": [str(a) for a in inst["answer"]],
                "id": inst["id"], "ctx_tokens": L[f"{task}/{inst['id']}"],
                "depth_ignored": True}
        if task == "infb_bookmc":
            meta["options"] = inst.get("options", [])
        return prompt, meta
    return build


def _scorer(content, meta):
    c = (content or "").strip()
    gold = meta["gold"]
    if any(g and g in c for g in gold):
        return 1.0, "contains"
    opts = [o for o in meta.get("options", []) if o and o in c]
    if opts:
        goldset = set(gold)
        hit = any(o in goldset for o in opts)
        return (1.0, "option-hit") if hit else (0.0, "wrong-option")
    if any(g and g.lower() in c.lower() for g in gold):
        return 0.0, "contains-case-miss"
    return 0.0, "miss"


def register(tasks_mod):
    for task in TASKS:
        tasks_mod.TASKS.append(task)
        tasks_mod.BUILDERS[task] = _builder(task)
    orig_score = tasks_mod.score

    def score(task, content, meta):
        if task in TASKS:
            return _scorer(content, meta)
        return orig_score(task, content, meta)

    tasks_mod.score = score
    print("infbench plugin registered:", ", ".join(TASKS))


if __name__ == "__main__":
    if __import__("sys").argv[1:] == ["calibrate"]:
        from transformers import AutoProcessor
        tok = AutoProcessor.from_pretrained("meta-models/Muse-Glimmer-30B").tokenizer
        lengths = measure_lengths(tok)
        d = ensure_data()["data"]
        for t, items in sorted(d.items()):
            ls = sorted(lengths[str(x["id"])] for x in items)
            for target in (128_000, 256_000, 512_000):
                n = sum(1 for v in ls if 0.5 * target <= v <= 0.92 * target)
                print(f"  {t} @ {target}: {n} instances in window", end="  ")
            print(f"(min={ls[0]:,} med={ls[len(ls)//2]:,} max={ls[-1]:,})")
