#!/usr/bin/env python3
"""k-matched counting grid (audit F3, 2026-08-15): de-confound k-difficulty.

Verified confound: counting accuracy is strongly k-dependent (k=5: 1.00 → k=12: 0.33
≤128k) and mean-k drifts across cells (192k drew 10.0 vs 64k's 8.48) — cross-context
comparisons confound length with difficulty.

This grid FIXES k per cell (strata k=6 easy / k=11 hard), same ctx ladder, both decode
modes:
  capability : temp 1.0/0.95/64 (contract; comparable to main grids)
  greedy     : temp 0 (deterministic secondary — splits sampling noise, cf. E2: ~38% of
               capability-mode counting misses flipped correct under greedy)

Cells: 2 strata × {32k, 64k, 128k, 256k} × 5 reps × 2 modes = 80 rows.
Cell ids: counting-k<K>|ctx|depth|rep|mode — separate namespace, never merges into
main-grid aggregates; surfaced via summarize/compare by label.

Usage (dev): python3 evals/harness/kmatched_counting.py --base-url http://vllm:8000/v1 \
                 --out outputs/eval/kmatched_counting.jsonl
"""
import argparse
import json
import os
import random
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import client  # noqa: E402
import tasks as T  # noqa: E402

STRATA = (6, 11)
CTXS = (32_000, 64_000, 128_000, 256_000)
REPS = 5


def build_fixed_k(rng, target_tokens, k):
    """counting builder with k pinned (same rendering as T.build_counting).
    Marker string is drawn k-independent so re-insertion never duplicates it."""
    body = T._haystack(rng, target_tokens)
    marker = f"SIGNAL-{rng.choice('KRXZ')}{rng.randint(10, 99)} telemetry beacon acknowledged"
    # strip any chance occurrence of the marker stem in the haystack (defensive)
    stem = marker.split()[0]
    body = body.replace(stem, "LOGSIG")
    for i in range(k):
        cut = int(len(body) * (i + rng.uniform(0.2, 0.8)) / k)
        body = body[:cut] + f" {marker}. " + body[cut:]
    prompt = T._wrap(body, f"How many times does the exact phrase \"{marker}\" appear in the "
                           f"context above? Reply with just the number.")
    # the QUESTION restates the marker once — exclude it from the verification count
    # (the model is asked to count occurrences IN THE CONTEXT; the scorer compares
    # against k, and the builder test asserts p[:prompt_start].count(marker) == k)
    return prompt, {"marker": marker, "count": k, "_question": True}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", default="muse-glimmer")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    done = set()
    if os.path.exists(args.out):
        for line in open(args.out):
            try:
                r = json.loads(line)
                done.add((r["target_ctx"], r["k"], r["rep"], r["mode"]))
            except Exception:
                pass
    n_total = len(STRATA) * len(CTXS) * REPS * 2
    i = 0
    for ctx in CTXS:
        for k in STRATA:
            for rep in range(REPS):
                for mode in ("capability", "greedy"):
                    i += 1
                    if (ctx, k, rep, mode) in done:
                        continue
                    seed = T.cell_seed(f"counting-k{k}", ctx, 0.5, rep)
                    rng = random.Random(seed)
                    prompt, meta = build_fixed_k(rng, ctx, k)
                    try:
                        res = client.chat(args.base_url, args.model,
                                          [{"role": "user", "content": prompt}],
                                          mode=("capability" if mode == "capability" else "parity"),
                                          max_tokens=8192, timeout=7200)
                        m = re.search(r"\d+", res["content"] or "")
                        got = int(m.group()) if m else None
                        row = {"config_label": "stock", "task": f"counting-k{k}",
                               "target_ctx": ctx, "depth": 0.5, "rep": rep, "mode": mode,
                               "k": k, "want": k, "got": got,
                               "score": 1.0 if got == k else 0.0,
                               "cell_id": f"stock|vllm|{mode}|counting-k{k}|{ctx}|0.5|{rep}",
                               "finish_reason": res["finish_reason"],
                               "prompt_tokens": res.get("prompt_tokens"),
                               "wall_s": res["wall_s"]}
                    except Exception as e:  # noqa: BLE001
                        row = {"config_label": "stock", "task": f"counting-k{k}",
                               "target_ctx": ctx, "depth": 0.5, "rep": rep, "mode": mode,
                               "k": k, "error": str(e)[:200],
                               "cell_id": f"stock|vllm|{mode}|counting-k{k}|{ctx}|0.5|{rep}"}
                    with open(args.out, "a") as f:
                        f.write(json.dumps(row) + "\n")
                    ok = row.get("score")
                    print(f"[{i}/{n_total}] k={k} ctx={ctx} rep={rep} {mode}: "
                          f"{'OK' if ok == 1.0 else ('MISS' if ok is not None else 'ERR')}"
                          + (f" (got {row.get('got')})" if ok == 0.0 else ""), flush=True)


if __name__ == "__main__":
    main()
