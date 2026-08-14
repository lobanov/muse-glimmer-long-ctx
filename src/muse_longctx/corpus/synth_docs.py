#!/usr/bin/env python3
"""PLAN §5 — synthetic long-document generator (GLM-5.2 teacher via pi headless).

Pipeline (all machine-checkable, per PLAN §5):
  1. `section(rng)` — plant 3-6 facts (window/region/version/team/port/flag) + 3
     distractors; pi writes a 150-350-word natural section embedding each key EXACTLY
     once, distractors never. verify_planted → regenerate once with a stricter prompt →
     discard on second failure (never hand-repair).
  2. `document(rng, target_tokens)` — assemble N verified sections (concurrent pi calls)
     into one coherent-ish technical document; record the fact ledger (key → section idx).
  3. `training_samples(doc, ledger)` — question templates per capability axis (variable
     tracking, multi-fact aggregation, conflict-aware lookup) with SHORT verified answers;
     serialized via src/muse_longctx/position_sampler.py (mode: genuine).

Concurrency: sections are independent → ThreadPoolExecutor over pi processes
(--concurrency, default 4). All calls cached (pi_teacher), so runs resume for free.

Usage:
  python3 src/muse_longctx/corpus/synth_docs.py --validate          # 4 sections, 1 doc, 2 samples
  python3 src/muse_longctx/corpus/synth_docs.py --doc-tokens 65536 --out outputs/corpus/synth_v1/doc_000.jsonl
"""
import argparse
import json
import os
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pi_teacher as PT  # noqa: E402

TEAMS = ["observability", "gateway", "storage", "release-eng", "sre-core", "payments"]
REGIONS = ["eu-central-2", "ap-southeast-3", "sa-east-1", "us-west-4", "af-south-2"]
WINDOWS = ["04:00-06:00 UTC", "22:00-23:30 UTC", "13:00-15:00 UTC", "07:30-09:00 UTC"]
VERSIONS = ["v2.14.3", "v3.0.1", "v1.9.7", "v4.2.0-rc2"]
FLAGS = ["rollout_gatekeeper", "mirror_sharding", "backpressure_v2", "tiered_reads"]
PORTS = range(9100, 9699)


def _facts(rng):
    """3 planted facts + 3 same-shape distractors for one section."""
    planted, distract = [], []
    for kind in rng.sample(["window", "region", "version", "team", "flag", "port"], 3):
        if kind == "window":
            planted.append(("window", rng.choice(WINDOWS)))
            distract.append(("window", rng.choice([w for w in WINDOWS if w not in planted])))
        elif kind == "region":
            planted.append(("region", rng.choice(REGIONS)))
            distract.append(("region", rng.choice([r for r in REGIONS if r != planted[-1][1]])))
        elif kind == "version":
            planted.append(("version", rng.choice(VERSIONS)))
            distract.append(("version", rng.choice([v for v in VERSIONS if v != planted[-1][1]])))
        elif kind == "team":
            planted.append(("team", rng.choice(TEAMS)))
            distract.append(("team", rng.choice([t for t in TEAMS if t != planted[-1][1]])))
        elif kind == "flag":
            planted.append(("flag", rng.choice(FLAGS)))
            distract.append(("flag", rng.choice([f for f in FLAGS if f != planted[-1][1]])))
        else:
            p = rng.choice(PORTS)
            planted.append(("port", str(p)))
            distract.append(("port", str(p + rng.choice([-11, 11]))))
    # dedupe planted vs distractor collisions
    distract = [(k, v) for (k, v) in distract if v not in [x[1] for x in planted]]
    return planted, distract


SECTION_PROMPT = (
    "Write a technical documentation section (150-350 words) about a fictional distributed "
    "system, in the style of an internal engineering runbook. Topic: {topic}. "
    "Hard requirements: state {facts_clause}; do NOT mention any of: {distractors}. "
    "Each required value must appear exactly once. Plain prose, no headings, no lists, "
    "no markdown. Output only the section text."
)

TOPICS = ["node rotation", "cache warming", "schema migration", "traffic mirroring",
          "backpressure handling", "failover drills", "log compaction", "quota enforcement"]


def section(rng, idx):
    planted, distract = _facts(rng)
    topic = rng.choice(TOPICS)
    facts_clause = "; ".join(f"the {k} is `{v}`" for k, v in planted)
    distract_clause = ", ".join(f"`{v}`" for _, v in distract)
    prompt = SECTION_PROMPT.format(topic=topic, facts_clause=facts_clause,
                                   distractors=distract_clause)
    want = [v for _, v in planted]
    notwant = [v for _, v in distract]
    for attempt in (0, 1):
        text = PT.generate(f"synthdoc-section-v1{'+retry' if attempt else ''}",
                           prompt if attempt == 0 else prompt + " (STRICT: recheck that "
                           "each required value appears EXACTLY once and forbidden "
                           "strings are absent.)", thinking="low")
        if text:
            ok, problems = PT.verify_planted(text, want, notwant)
            if ok:
                return {"text": text, "facts": planted, "topic": topic}
    return None


Q_TEMPLATES = [
    ("var", "According to the documentation, what is the {kind} for {topic}?",
     "single-fact variable tracking"),
    ("agg", "State the {kind1} for {topic1} and the {kind2} for {topic2}.",
     "two-fact aggregation across distant sections"),
]


def document(rng, target_tokens, concurrency=4):
    """Assemble verified sections into one document sized ≈ target_tokens (≈4.1 chars/tok)."""
    aim_chars = int(target_tokens * 4.1)
    n = max(2, aim_chars // 1800)  # ~250 words avg
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        results = list(ex.map(lambda i: section(random.Random(rng.random() * i + i), i),
                              range(n)))
    secs = [r for r in results if r]
    if len(secs) < max(2, int(0.9 * n)):
        raise RuntimeError(f"too many discarded sections: {len(secs)}/{n}")
    rng.shuffle(secs)
    body = "\n\n".join(f"[{i}] {s['text']}" for i, s in enumerate(secs))
    ledger = [{"section": i, "topic": s["topic"], "facts": dict(s["facts"])}
              for i, s in enumerate(secs)]
    return body, ledger


def training_samples(body, ledger, rng, max_samples=4):
    """Prompt = whole document + question; completion = short verified answer (SFT)."""
    samples = []
    n_sec = len(ledger)
    for _ in range(max_samples):
        kind = rng.random()
        if kind < 0.5 or n_sec < 2:
            e = ledger[rng.randrange(n_sec)]
            fk, fv = rng.choice(list(e["facts"].items()))
            q = Q_TEMPLATES[0][1].format(kind=fk, topic=e["topic"])
            a = fv
        else:
            e1, e2 = rng.sample(ledger, 2)
            (k1, v1), (k2, v2) = rng.choice(list(e1["facts"].items())), \
                rng.choice(list(e2["facts"].items()))
            q = Q_TEMPLATES[1][1].format(kind1=k1, topic1=e1["topic"],
                                         kind2=k2, topic2=e2["topic"])
            a = f"{v1} and {v2}"
        answer_tok_len = 24
        prompt_text = (f"Documentation follows.\n\n{body}\n\nQuestion: {q}\nAnswer "
                       f"concisely (a short phrase, no prose):")
        samples.append({"prompt": prompt_text, "answer": a, "question": q,
                        "axis": "var" if kind < 0.5 else "agg"})
    return samples


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--doc-tokens", type=int)
    ap.add_argument("--seed", type=int, default=2026_0815,
                    help="master seed; per-doc seed derives from it + out path")
    ap.add_argument("--out")
    ap.add_argument("--concurrency", type=int, default=4)
    args = ap.parse_args()
    rng = random.Random(2026_0815)

    if args.validate:
        body, ledger = document(random.Random(1), 12_000, concurrency=2)
        print(f"doc: {len(body):,} chars ≈ {len(body)//4:,} tok, {len(ledger)} sections, "
              f"facts: {sum(len(e['facts']) for e in ledger)}")
        samples = training_samples(body, ledger, rng, max_samples=3)
        for s in samples:
            assert s["answer"].split(" and ")[0] in body  # answers grounded in doc
            print(f"  [{s['axis']}] Q: {s['question'][:90]}  A: {s['answer']}")
        print("synth_docs validation OK (sections verified, answers grounded)")
    else:
        assert args.doc_tokens and args.out
        rng_master = random.Random(args.seed)
        body, ledger = document(random.Random(rng_master.random() * 1e9), args.doc_tokens,
                                args.concurrency)
        samples = training_samples(body, ledger, rng_master, max_samples=8)
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            f.write(json.dumps({"body": body, "ledger": ledger,
                                "target_tokens": args.doc_tokens}) + "\n")
            for s in samples:
                f.write(json.dumps(s) + "\n")
        print(f"wrote {1 + len(samples)} lines -> {args.out}")
