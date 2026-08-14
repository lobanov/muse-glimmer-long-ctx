#!/usr/bin/env python3
"""PLAN §5 — repo-scale document samples (35% component): github_repos render + verified facts.

Renders a training-only repo (github_repos: license gate, exclusion fail-closed,
deterministic) and asks questions whose answers are regex-extracted values UNIQUE in the
rendered document (reuses natural_docs.extract_facts / make_sample — pi phrases the
question; ground truth stays deterministic).

Usage: python3 src/muse_longctx/corpus/repos_doc.py --repo simonw/sqlite-utils \
           --out outputs/corpus/repos_v1/sqlite-utils.jsonl --validate
"""
import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from github_repos import fetch_repo, render_repo  # noqa: E402
from natural_docs import extract_facts, make_sample  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "outputs", "corpus")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-samples", type=int, default=4)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--validate", action="store_true")
    args = ap.parse_args()
    rng = random.Random(args.seed)

    files, lic = fetch_repo(args.repo)
    body = render_repo(files, args.repo, lic)
    facts = extract_facts(body, max_facts=120)
    samples = []
    for fact in rng.sample(facts, min(len(facts), args.n_samples * 3)):
        if len(samples) >= args.n_samples:
            break
        s = make_sample(fact, rng)
        if s and body.count(s["answer"]) == 1:
            samples.append(s)
    if not samples:
        raise SystemExit(f"no verified samples from {args.repo} (facts={len(facts)})")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        f.write(json.dumps({"body": "", "ledger": None, "target_tokens": None}) + "\n")
        for s in samples:
            f.write(json.dumps({"prompt": f"Repository content follows.\n\n{body}\n\n"
                                          f"Question: {s['question']}\nAnswer concisely "
                                          f"(the value only):",
                                "answer": s["answer"], "question": s["question"],
                                "axis": "repo", "kind": s["kind"]}) + "\n")
    print(f"{args.repo}: render {len(body):,} chars | facts {len(facts)} | "
          f"{len(samples)} verified samples -> {args.out}")
    if args.validate:
        for s in samples:
            assert body.count(s["answer"]) == 1
            assert s["answer"].lower() not in s["question"].lower()
        print("repos_doc validation OK (unique in render, grounded, no answer leak)")


if __name__ == "__main__":
    main()
