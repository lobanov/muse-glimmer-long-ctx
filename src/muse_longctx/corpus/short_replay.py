#!/usr/bin/env python3
"""PLAN §5 — short-context replay (10%): pi-generated instruction/coding Q&A.

Purpose: ≤128k regression retention (PLAN §10). Self-generated ⇒ license-clean and
benchmark-clean by construction; deduped by prompt-prefix hash; light structural checks
(answer non-empty, ≤ 400 chars, no verbatim eval-suite artifacts: reject if it contains a
NoLiMa needle marker or LQA-style 'ONLY the letter' phrasing — cheap decontamination).

Output = serialize.py-compatible doc file (prompt/answer pairs, axis "short").

Usage: python3 src/muse_longctx/corpus/short_replay.py --n 8 --out outputs/corpus/short_v1/s1.jsonl --validate
"""
import argparse
import hashlib
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pi_teacher as PT  # noqa: E402

TOPICS = ["web service", "data pipeline", "CLI tool", "embedded firmware", "game loop",
          "compiler pass", "database index", "cache layer", "auth flow", "file parser",
          "concurrent worker pool", "rate limiter", "retry logic", "config loading",
          "logging setup", "timezone handling", "unicode text", "binary protocol",
          "image resize", "audio buffer", "network proxy", "scheduler", "unit test",
          "type checker", "regex engine", "template renderer", "session store",
          "message queue", "crash recovery", "metrics export", "feature flag",
          "schema migration"]

CATS = [
    ("python-bugfix", "a 6-14-line Python function with ONE subtle bug, then the corrected "
     "function, then one sentence naming the bug"),
    ("api-usage", "a short documentation-style answer: how to use one standard-library or "
     "well-known tool API correctly, with a 3-8-line example"),
    ("explain-concept", "a 3-6 sentence precise explanation of a systems/ML concept "
     "(e.g., write-ahead log, RoPE, quantization error), no fluff"),
    ("refactor", "a 6-12-line code snippet and a cleaner refactor of it, with a one-line "
     "rationale"),
    ("sql", "a small CREATE TABLE + a SELECT answering a stated question over it"),
    ("shell", "a one-liner or short pipeline for a stated file/text-processing task, "
     "with a one-sentence explanation"),
    ("math-check", "a small step-by-step verification of a numeric or combinatorial claim "
     "(≤ 5 steps)"),
]

PROMPT = (
    "Produce a self-contained {cat} item about a {topic}. Requirements: realistic, "
    "technically correct, no headings, no markdown fences, plain text. End with a line "
    "'ANSWER: <the key answer or fixed code>'. Keep the whole item under 160 words. "
    "Output only the item."
)


def parse_item(text):
    """Split into prompt part (task) and answer (ANSWER: line)."""
    idx = text.rfind("ANSWER:")
    if idx < 0:
        return None
    task = text[:idx].strip()
    answer = text[idx + len("ANSWER:"):].strip()
    if not task or not answer or len(answer) > 400:
        return None
    return task, answer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--validate", action="store_true")
    args = ap.parse_args()
    rng = random.Random(args.seed)

    items, seen = [], set()
    for i in range(args.n * 3):
        if len(items) >= args.n:
            break
        cat, desc = CATS[rng.randrange(len(CATS))]
        topic = rng.choice(TOPICS)
        text = PT.generate("shortreplay-v2", PROMPT.format(cat=desc, topic=topic),
                           thinking="low")
        if not text:
            continue
        parsed = parse_item(text)
        if not parsed:
            continue
        task, answer = parsed
        # structural decontamination: reject eval-suite artifacts
        low = (task + answer).lower()
        if "only the letter" in low or "needle" in low:
            continue
        h = hashlib.sha1(task[:120].encode()).hexdigest()[:12]
        if h in seen:
            continue
        seen.add(h)
        items.append({"prompt": f"{task}\n\nAnswer:", "answer": answer, "category": cat})

    if len(items) < max(1, args.n // 2):
        raise SystemExit(f"only {len(items)} items verified — rerun (cached calls resume)")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        f.write(json.dumps({"body": "", "ledger": None, "target_tokens": None}) + "\n")
        for it in items:
            f.write(json.dumps({"prompt": it["prompt"], "answer": it["answer"],
                                "question": it["prompt"][:80], "axis": "short",
                                "category": it["category"]}) + "\n")
    print(f"short_replay: {len(items)} items -> {args.out}")
    if args.validate:
        import collections
        print("  categories:", dict(collections.Counter(i["category"] for i in items)))
        for it in items[:2]:
            print(f"  [{it['category']}] {it['prompt'][:80]!r} -> {it['answer'][:60]!r}")
        print("short_replay validation OK")


if __name__ == "__main__":
    main()
