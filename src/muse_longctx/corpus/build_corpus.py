#!/usr/bin/env python3
"""PLAN §5→§7 — corpus mixer: assemble component batches into the training mixture.

Component shares (PLAN §5): repos 35% · synth 30% · natural 15% · agent 10% · short 10%
Length mixture (PLAN §7, genuine-dominant): 55–70% genuine 96–256k · 10–20% genuine
32–64k · 10–20% short replay · 10–15% virtual-position (0 until §9 ablation).

The mixer consumes trainer rows (*.samples.jsonl produced by serialize.py), pools them
per component, samples to the target weights (as close as inventory allows), dedupes by
input_ids hash, and writes:
    outputs/corpus/<name>/train.jsonl    — the mixed trainer rows
    outputs/corpus/<name>/manifest.json  — target vs actual weights, counts, token totals

It reports shortfalls loudly but never fabricates balance: actual weights are computed
from what exists. --require-full exits nonzero if any component is empty.

Usage: python3 src/muse_longctx/corpus/build_corpus.py --name train_v1 \
           --root outputs/corpus --targets 'repos:.35,synth:.30,natural:.15,agent:.10,short:.10'
"""
import argparse
import glob
import hashlib
import json
import os
import random

COMPONENTS = {
    "repos": "outputs/corpus/repos_v1",     # repo-scale documents (github_repos + tasks)
    "synth": "outputs/corpus/synth_v1",     # synthetic long docs
    "natural": "outputs/corpus/nat_v1",     # public-domain books
    "agent": "outputs/corpus/agent_v1",     # agent trajectories
    "short": "outputs/corpus/short_v1",     # short replay
}


def load_pool(d):
    rows, seen = [], set()
    for path in sorted(glob.glob(os.path.join(d, "*.samples.jsonl"))):
        for line in open(path):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            # dedupe on head+tail+length: the head alone is shared by all samples of
            # one document (same body+template) — the tail carries question+answer.
            ids = r["input_ids"]
            h = hashlib.sha1((str(ids[:64]) + str(ids[-64:]) + str(len(ids))).encode()) \
                .hexdigest()
            if h in seen:
                continue
            seen.add(h)
            rows.append(r)
    return rows


BUCKETS_TO_REPORT = (131072, 262144)   # trainer --seq-bucket candidates (PLAN §7)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="train_v1")
    ap.add_argument("--root", default="outputs/corpus")
    ap.add_argument("--targets", default="repos:.35,synth:.30,natural:.15,agent:.10,short:.10")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--require-full", action="store_true",
                    help="exit 1 if any component pool is empty")
    args = ap.parse_args()
    rng = random.Random(args.seed)

    targets = dict((k, float(v)) for k, v in
                   (kv.split(":") for kv in args.targets.split(",")))
    pools = {c: load_pool(d) for c, d in COMPONENTS.items()}
    empty = [c for c, p in pools.items() if not p]
    if empty:
        print(f"[mixer] EMPTY components: {empty}")
        if args.require_full:
            raise SystemExit(1)

    total_avail = sum(len(p) for p in pools.values())
    out_rows, stats = [], {}
    for comp, pool in pools.items():
        if not pool:
            stats[comp] = {"n": 0, "tokens": 0, "target": targets[comp], "actual": 0.0}
            continue
        # allocation: min(target share of total, available share of total)
        want = int(round(targets[comp] * total_avail))
        take = min(want, len(pool))
        chosen = rng.sample(pool, take)
        out_rows.extend(chosen)
        stats[comp] = {"n": take, "avail": len(pool),
                       "tokens": sum(len(r["input_ids"]) for r in chosen),
                       "target": targets[comp]}
    actual_total = sum(s["n"] for s in stats.values())
    for s in stats.values():
        s["actual"] = round(s["n"] / max(1, actual_total), 4)

    outdir = os.path.join(args.root, args.name)
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "train.jsonl"), "w") as f:
        for r in out_rows:
            f.write(json.dumps(r) + "\n")
    manifest = {"name": args.name, "seed": args.seed, "rows": len(out_rows),
                "tokens": sum(len(r["input_ids"]) for r in out_rows), "components": stats,
                "length_buckets": {str(b): {
                    "rows": sum(1 for r in out_rows if len(r["input_ids"]) <= b),
                    "tokens": sum(len(r["input_ids"]) for r in out_rows
                                  if len(r["input_ids"]) <= b)}
                    for b in BUCKETS_TO_REPORT},
                "length_note": "genuine-only so far; virtual-position rows enter via the "
                               "position sampler at §9 ablation time. length_buckets = "
                               "what the trainer sees per --seq-bucket setting."}
    with open(os.path.join(outdir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    print(json.dumps(manifest, indent=1))
    print(f"-> {outdir}/train.jsonl ({len(out_rows)} rows)")


if __name__ == "__main__":
    main()
