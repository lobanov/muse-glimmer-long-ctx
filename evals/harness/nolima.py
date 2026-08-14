#!/usr/bin/env python3
"""PLAN §2 — NoLiMa integration (official needle set + haystacks; eval-only, never training data).

Ports the official evaluation logic from github.com/adobe-research/NoLiMa
(evaluation/run_tests.py + async_evaluate.py + data/book_haystack.py), adapted to this
harness's builder/scorer/grid conventions:

- instances: needle_set.json tasks (10) × tests (~290 each) × question types
  (onehop|twohop); {CHAR} drawn from character_set (gold answer = character, official
  behavior); {1}/{2}/{3} filled from test input_args; official task_template kept.
- haystack: the official rand_shuffle / rand_shuffle_long word-shuffled books
  (low discourse coherence, realistic lexical stats), sliced to ~90% of target tokens
  (calibrated against the Glimmer tokenizer, cached), needle inserted at depth fraction.
- scoring: official "contains" metric — any gold answer appearing in the response
  (case-sensitive substring, matching upstream). detail records case-insensitive hit too.

License note: needle set + haystacks are Adobe Research License (non-commercial research).
This project uses them for evaluation only.

Register with the runner:  --plugin nolima
Then e.g.:                 --tasks nolima --ctx 8000,16000,32000 --depths 0,0.5,1.0
Canonical leaderboard lengths: 250, 500, 1K, 2K, 4K, 8K, 16K, 32K (words→tokens calibrated).
"""
import json
import os
import random

from huggingface_hub import hf_hub_download

DATASET = "amodaresi/NoLiMa"
CALIB_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "outputs", "eval",
                          "nolima_calib.json")
DEFAULT_CPT = 4.25          # chars per token fallback (English prose) until calibrated
FILL_FACTOR = 0.90

_cache = {}


def ensure_data():
    if "tests" in _cache:
        return _cache
    needle_path = hf_hub_download(DATASET, "needlesets/needle_set.json", repo_type="dataset")
    with open(needle_path) as f:
        needle_set = json.load(f)
    tests = []
    for exp in needle_set:
        for qtype, question in exp["questions"].items():
            for test_id, test in exp["tests"].items():
                needle, q = exp["needle"], question
                for i, arg in enumerate(test["input_args"]):
                    ph = "{" + str(i + 1) + "}"
                    needle = needle.replace(ph, arg)
                    q = q.replace(ph, arg)
                tests.append({
                    "instance": f"{exp['id']}_{test_id}_{qtype}",
                    "template": exp.get("task_template"),
                    "system_prompt": exp.get("system_prompt", ""),
                    "needle_tmpl": needle,
                    "question": q,
                    "character_set": exp.get("character_set", []),
                    "gold_answers": test.get("gold_answers", ""),
                })
    books = []
    for sub in ("rand_shuffle", "rand_shuffle_long"):
        for i in range(1, 6):
            p = hf_hub_download(DATASET, f"haystack/{sub}/rand_book_{i}.txt",
                                repo_type="dataset")
            books.append(open(p).read())
    _cache["tests"] = tests
    _cache["corpus"] = "\n".join(books)
    return _cache


def _chars_per_token(tok=None):
    if os.path.exists(CALIB_PATH):
        return json.load(open(CALIB_PATH))["cpt"]
    if tok is None:
        return DEFAULT_CPT
    sample = ensure_data()["corpus"][:40000]
    cpt = len(sample) / len(tok(sample)["input_ids"])
    os.makedirs(os.path.dirname(CALIB_PATH), exist_ok=True)
    json.dump({"cpt": cpt, "sample_chars": len(sample)}, open(CALIB_PATH, "w"))
    return cpt


def build_nolima(rng, target_tokens, depth):
    d = ensure_data()
    t = d["tests"][rng.randrange(len(d["tests"]))]
    if "{CHAR}" in t["needle_tmpl"]:
        char = rng.choice(t["character_set"])
        needle = t["needle_tmpl"].replace("{CHAR}", char)
        gold = [char]
    else:
        needle = t["needle_tmpl"]
        gold = t["gold_answers"] if isinstance(t["gold_answers"], list) else [t["gold_answers"]]
        assert gold and gold != [""], f"no gold for {t['instance']}"
    cpt = _chars_per_token()
    aim_chars = int(target_tokens * FILL_FACTOR * cpt)
    corpus = d["corpus"]
    start = rng.randrange(0, max(1, len(corpus) - aim_chars - 1))
    body = corpus[start:start + aim_chars]
    cut = int(len(body) * depth)
    text = body[:cut] + " " + needle + " " + body[cut:]
    prompt = t["template"].format(haystack=text, question=t["question"])
    return prompt, {"gold": gold, "instance": t["instance"], "needle": needle}


def score_nolima(content, meta):
    c = (content or "").strip()
    if any(g in c for g in meta["gold"]):
        return 1.0, "contains"
    if any(g.lower() in c.lower() for g in meta["gold"]):
        return 0.0, "contains-case-miss"
    return 0.0, "miss"


def calibrate(tokenizer):
    """Called once with the Glimmer tokenizer to cache chars/token for this corpus."""
    _chars_per_token(tokenizer)
    print("nolima calibration cached ->", CALIB_PATH)


def register(tasks_mod):
    tasks_mod.TASKS.append("nolima")
    tasks_mod.BUILDERS["nolima"] = build_nolima
    orig_score = tasks_mod.score

    def score(task, content, meta):
        if task == "nolima":
            return score_nolima(content, meta)
        return orig_score(task, content, meta)

    tasks_mod.score = score
    print("nolima plugin registered: 1 task (official needle set; contains metric)")


if __name__ == "__main__" and __import__("sys").argv[1:] == ["calibrate"]:
    from transformers import AutoProcessor
    tok = AutoProcessor.from_pretrained("meta-models/Muse-Glimmer-30B").tokenizer
    calibrate(tok)
    d = ensure_data()
    print(f"instances: {len(d['tests'])} | corpus chars: {len(d['corpus']):,}")
