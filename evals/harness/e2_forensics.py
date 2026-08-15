#!/usr/bin/env python3
"""E2 (review R2): counting-miss forensics — decode-strategy vs attention-dilution.

Re-runs the EXACT instances that missed (cell_seed is deterministic → same haystack,
same markers) under three conditions:
  base     : as-scored originally (capability contract) — sanity anchor
  greedy   : temp 0 — removes sampling noise; if misses persist, not a sampling artifact
  enum     : greedy + explicit enumeration instruction ("list each entry number as you
             count") — if this FIXES misses, the failure is decode/working-memory
             strategy, not attention retrieval (dilution hypothesis falsified)

Verdict rule (3-way, audit 2026-08-15):
  sampling-share   : fraction of capability misses that flip correct under greedy
                     (measures stochastic component — E2 full run: 5/17 flipped)
  systematic-decode: enum fixes >= 2/3 of the greedy-persistent misses
  attention/retrieval: greedy-persistent AND enumeration-resistant
The printed single-line verdict is a coarse summary; read the paired anatomy table
(STATUS) for the actual split.

Usage (dev container, vLLM serving):
  python3 evals/harness/e2_forensics.py --base-url http://vllm:8000/v1 \
      --out outputs/eval/e2_counting_forensics.jsonl
"""
import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import client  # noqa: E402
import tasks as T  # noqa: E402

ENUM_SUFFIX = ("\n\nBefore answering, enumerate methodically: go through the context "
               "entry by entry and list the entry number each time you spot the exact "
               "phrase, keeping a running tally; then state the final count as just "
               "the number.")


def miss_cells(paths):
    """(ctx, depth, rep) for score==0 counting cells across stock grids."""
    out = []
    for p in paths:
        if not os.path.exists(p):
            continue
        for line in open(p):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("task") == "counting" and not r.get("error") and r.get("score") == 0.0:
                out.append((r["target_ctx"], r["depth"], r["rep"]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", default="muse-glimmer")
    ap.add_argument("--out", required=True)
    ap.add_argument("--grids", default="outputs/eval/stock_vllm_le128k.jsonl,"
                                       "outputs/eval/stock_vllm_gt128k.jsonl,"
                                       "outputs/eval/stock_cwe.jsonl")
    args = ap.parse_args()
    cells = miss_cells(args.grids.split(","))
    print(f"miss cells: {len(cells)}")

    done = set()
    if os.path.exists(args.out):
        for line in open(args.out):
            try:
                r = json.loads(line)
                done.add((r["ctx"], r["depth"], r["rep"], r["cond"]))
            except Exception:
                pass

    for (ctx, depth, rep) in cells:
        rng = random.Random(T.cell_seed("counting", ctx, depth, rep))
        prompt, meta = T.BUILDERS["counting"](rng, ctx, depth)
        for cond, mode, suffix in (("greedy", "parity", ""),
                                   ("enum", "parity", ENUM_SUFFIX)):
            if (ctx, depth, rep, cond) in done:
                continue
            try:
                res = client.chat(args.base_url, args.model,
                                  [{"role": "user", "content": prompt + suffix}],
                                  mode=mode, max_tokens=8192, timeout=7200)
                ok = str(meta["count"]) in (res["content"] or "")
                # crude tally check: how many entry-numbers the model listed
                listed = sum(1 for tok in (res["reasoning"] or "").split()
                             if tok.rstrip(",;:").isdigit())
                row = {"ctx": ctx, "depth": depth, "rep": rep, "cond": cond,
                       "want": meta["count"], "correct": ok,
                       "finish": res["finish_reason"],
                       "content_head": (res["content"] or "")[:80],
                       "numbers_listed": listed,
                       "ctoks": res.get("completion_tokens"), "wall_s": res["wall_s"]}
            except Exception as e:  # noqa: BLE001
                row = {"ctx": ctx, "depth": depth, "rep": rep, "cond": cond,
                       "error": str(e)[:200]}
            with open(args.out, "a") as f:
                f.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)

    # verdict
    rows = [json.loads(l) for l in open(args.out) if l.strip()]
    g = [r for r in rows if r["cond"] == "greedy" and not r.get("error")]
    e = [r for r in rows if r["cond"] == "enum" and not r.get("error")]
    if g and e:
        gp = sum(1 for r in g if not r["correct"]) / len(g)
        fixed = sum(1 for r in e if r["correct"]) / len(e)
        verdict = ("decode-strategy (enumeration fixes)" if fixed >= 2 / 3
                   else "dilution-live (greedy persists, enumeration doesn't fix)"
                   if fixed <= 1 / 3 else "inconclusive")
        print(f"\nVERDICT: greedy-miss-rate {gp:.2f} | enum-fix-rate {fixed:.2f} -> {verdict}")


if __name__ == "__main__":
    main()
