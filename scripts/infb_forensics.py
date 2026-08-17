#!/usr/bin/env python3
"""Forensics for InfBench/LQA length confounds (goal afe6584b, no GPU needed).

Problems with the existing ">128k" reads that this script quantifies:
1. InfBench builder picks instances from a [0.5, 0.92]x target_ctx band -> "256k"
   cells contain 131k-236k actual prompts; every rep is a DIFFERENT instance
   (difficulty uncontrolled). LQA similar (512k target -> 350k prompts).
2. 'contains-case-miss' rows are scored 0 although gold appears case-insensitively.
3. 1024-token budget drains (suite used --max-tokens 1024) count as misses.

Re-derives, for every existing infbench row, the exact instance (rng replay of the
same cell_seed the harness used) + its true token length, and emits:
  outputs/eval/infb_forensics.jsonl   (per-row: id, ctx_tokens, band, strict/ci scores)
plus a stdout summary table.
"""
import json
import random
import sys

sys.path.insert(0, "evals/harness")
import tasks as T  # noqa: E402
import infbench  # noqa: E402

infbench.register(T)

ROWS_IN = ["outputs/eval/suite_infbench.jsonl",
           "outputs/eval/arm_qk4.3.jsonl", "outputs/eval/arm_qk5.0.jsonl"]
OUT = "outputs/eval/infb_forensics.jsonl"


def main():
    d = infbench.ensure_data()
    true_lengths = json.load(open("outputs/eval/infbench_lengths_v3.json"))
    v1 = json.load(open("outputs/eval/infbench_lengths.json"))
    out = []
    for path in ROWS_IN:
        try:
            rows = [json.loads(l) for l in open(path)]
        except FileNotFoundError:
            continue
        for r in rows:
            if r["task"] not in infbench.TASKS or r.get("error"):
                continue
            # replay the harness's exact rng to recover the instance
            seed = T.cell_seed(r["task"], r["target_ctx"], r["depth"], r["rep"])
            rng = random.Random(seed)
            items = d["data"][r["task"]]
            target = r["target_ctx"]
            # historical selection replay: the OLD cache was id-collision soup; the pool the
            # suite actually used came from infbench_lengths.json AS IT WAS then.
            # We replay with the v1 cache for fidelity and attach TRUE v3 length.
            fits = [x for x in items if v1[str(x["id"])] <= 0.92 * target]
            pool = [x for x in fits if v1[str(x["id"])] >= 0.5 * target] or fits
            inst = pool[rng.randrange(len(pool))]
            ctx_tok = true_lengths[f"{r['task']}/{inst['id']}"]  # true length (v3)
            content = r.get("response_head", "")
            # note: response_head is truncated; strict re-scoring uses detail instead.
            strict = r["score"]
            ci_hit = r["detail"] in ("contains", "option-hit", "contains-case-miss")
            drain = r.get("finish_reason") == "length"
            out.append({
                "src": path.split("/")[-1], "label": r["config_label"],
                "task": r["task"], "target_ctx": r["target_ctx"], "rep": r["rep"],
                "instance_id": inst["id"], "ctx_tokens": ctx_tok,
                "band": f"{(ctx_tok // 40000) * 40}-{(ctx_tok // 40000 + 1) * 40}k",
                "strict": strict, "ci_hit": ci_hit, "drain": drain,
                "detail": r["detail"], "max_toks_seen": r.get("completion_tokens"),
                "finish": r.get("finish_reason"),
            })
    with open(OUT, "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")

    # summary: strict vs case-insensitive vs non-drain, by task x target x band
    from collections import defaultdict
    agg = defaultdict(lambda: [0, 0, 0, 0])  # n, strict, ci, non-drain-strict
    for r in out:
        k = (r["task"], r["target_ctx"], r["band"])
        a = agg[k]
        a[0] += 1
        a[1] += r["strict"]
        a[2] += 1 if r["ci_hit"] else 0
        if not r["drain"]:
            a[3] += r["strict"]
    print(f"{'task':<16}{'target':>8}{'band':>7}{'n':>3}{'strict':>8}{'ci':>5}{'nodrain':>8}")
    for k in sorted(agg, key=lambda x: (x[0], x[1], x[2])):
        n, s, ci, nd = agg[k]
        print(f"{k[0]:<16}{k[1]//1000:>7}k{k[2]:>7}{n:>3}{s:>8}{ci:>5}{nd:>8}")
    print(f"\nwritten: {OUT} ({len(out)} rows)")


if __name__ == "__main__":
    main()
