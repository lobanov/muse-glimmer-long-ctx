#!/usr/bin/env python3
"""PLAN §5 — natural long documents (15%): Project Gutenberg public-domain books.

Ground truth is DETERMINISTIC (no model trust): fact values are extracted by regex and
must occur EXACTLY ONCE in the whole text (unique 4-digit years, unique proper nouns).
pi (GLM-5.2 headless) only PHRASES the question from the source sentence; verification:
the answer value must not appear in the question, and must appear exactly once in the text.

Sample: prompt = book slice (genuine length) + question; answer = the extracted value.
Output = serialize.py-compatible doc file.

Usage (host): python3 src/muse_longctx/corpus/natural_docs.py --book 1342 \
                  --slice-tokens 65536 --out outputs/corpus/nat_v1/pride.jsonl --validate
"""
import argparse
import json
import os
import random
import re
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pi_teacher as PT  # noqa: E402

BOOKS = {  # stable Gutenberg ids, public domain
    1342: "Pride and Prejudice",
    84: "Frankenstein",
    2701: "Moby Dick",
    1661: "The Adventures of Sherlock Holmes",
    98: "A Tale of Two Cities",
    1400: "Great Expectations",
}
HDR = re.compile(r"\*\*\* ?START OF (THE|THIS) PROJECT GUTENBERG.*?\*\*\*", re.S | re.I)
FTR = re.compile(r"\*\*\* ?END OF (THE|THIS) PROJECT GUTENBERG.*", re.S | re.I)
YEAR = re.compile(r"\b(1[5-9]\d\d|20[0-2]\d)\b")
WORD = re.compile(r"\b([A-Z][a-z]{4,})\b")
COMMON = set("""there their these those which where when while whose being every after
against myself yourself himself herself itself themselves about above again against
because before below between during further having other should could would shall might
must upon whose chapter part first second third forth young little great good old house
lady lord captain doctor master general colonel major saviour english london paris
france england germany italy america spanish french german russian Chapter Volume Book
Project Gutenberg""".split())


def fetch_book(book_id):
    url = f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt"
    txt = urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent":
                                 "muse-glimmer-corpus/0.1"}), timeout=120).read()
    txt = txt.decode("utf-8", "replace")
    m = HDR.search(txt)
    if m:
        txt = txt[m.end():]
    m = FTR.search(txt)
    if m:
        txt = txt[:m.start()]
    return txt.strip()


def extract_facts(text, max_facts=40):
    from collections import Counter
    facts = []
    years = {}
    for m in YEAR.finditer(text):
        years.setdefault(m.group(1), 0)
        years[m.group(1)] += 1
    words = {}
    for m in WORD.finditer(text):
        words.setdefault(m.group(1), 0)
        words[m.group(1)] += 1
    words_low = {w.lower() for w in words}  # all words seen capitalized at least once
    uniq_years = [y for y, n in years.items() if n == 1]
    uniq_words = [w for w, n in words.items()
                  if n == 1 and w.lower() not in COMMON and not w.endswith("ly")
                  and w.lower() not in words_low]  # lowercase occurrence ⇒ positional cap
    # tier 2: rare distinctive words (any case), length ≥ 7, unique in slice
    allw = re.findall(r"\b([A-Za-z][a-z]{6,})\b", text)
    wcount = Counter(allw)
    rares = [w for w, n in wcount.items()
             if n == 1 and w.lower() not in COMMON and "." not in w][:max_facts]
    for y in uniq_years[:max_facts // 2]:
        i = text.find(y)
        facts.append(("year", y, text[max(0, i - 260):i + 260]))
    for w in uniq_words[:max_facts]:
        i = text.find(w)
        facts.append(("name", w, text[max(0, i - 260):i + 260]))
    for w in rares[:max_facts]:
        i = text.find(w)
        facts.append(("rare", w, text[max(0, i - 260):i + 260]))
    return facts


PHRASE_PROMPT = (
    "Here is a sentence fragment from a book:\n\n---\n{ctx}\n---\n\nThe target answer is "
    "exactly: `{value}`. Write ONE natural reading-comprehension question about this "
    "fragment whose correct answer is exactly that value. The question must NOT contain "
    "the answer value or any word of it. Question only, nothing else."
)


def make_sample(fact, rng):
    kind, value, ctx = fact
    q = PT.generate("natdoc-question-v1", PHRASE_PROMPT.format(ctx=ctx, value=value),
                    thinking="low")
    if not q:
        return None
    q = q.strip().strip('"')
    # verification: answer unique in book, absent from question, question is a question
    if value.lower() in q.lower() or not q.strip().endswith("?"):
        return None
    return {"question": q, "answer": value, "kind": kind}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--book", type=int, required=True)
    ap.add_argument("--slice-tokens", type=int, default=65536)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-samples", type=int, default=4)
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    text = fetch_book(args.book)
    aim = int(args.slice_tokens * 4.1)
    samples, body = [], ""
    for attempt in range(3):  # retry slice positions until enough verified facts
        rng = random.Random(args.seed + 1000 * attempt)
        start = rng.randrange(0, max(1, len(text) - aim - 1))
        body = text[start:start + aim]
        facts = extract_facts(body)
        for fact in rng.sample(facts, min(len(facts), args.n_samples * 3)):
            if len(samples) >= args.n_samples:
                break
            s = make_sample(fact, rng)
            if s and body.count(s["answer"]) == 1:
                samples.append(s)
        if len(samples) >= args.n_samples:
            break
    if not samples:
        raise SystemExit("no verified samples — try a different book/slice")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        f.write(json.dumps({"body": "", "ledger": None, "target_tokens": None}) + "\n")
        for s in samples:
            f.write(json.dumps({"prompt": f"Book text follows.\n\n{body}\n\nQuestion: "
                                          f"{s['question']}\nAnswer concisely (the value "
                                          f"only):",
                                "answer": s["answer"], "question": s["question"],
                                "axis": "natdoc", "kind": s["kind"]}) + "\n")
    print(f"book {args.book}: {len(text):,} chars | facts {len(facts)} | "
          f"{len(samples)} verified samples -> {args.out}")
    if args.validate:
        for s in samples:
            assert body.count(s["answer"]) == 1 and s["answer"] in body
            assert s["answer"].lower() not in s["question"].lower()
        print("natural_docs validation OK (unique in slice, grounded, "
              "answer absent from question)")


if __name__ == "__main__":
    main()
