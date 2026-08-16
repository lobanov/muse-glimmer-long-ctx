#!/usr/bin/env python3
"""Aggregate eval results (Parquet or JSONL) per PLAN §2: mean score + 95% CI over
instance resamples, per (config, engine, task, context). Also reports prefill latency.

CI: t-distribution (small n). n=3 -> t=4.303, n=4 -> 3.182, n=5 -> 2.776, else normal 1.96.

Usage:
  python3 evals/harness/summarize.py outputs/eval/stock_vllm.parquet [--markdown out.md]
"""
import argparse
import glob
import json
import math
import sys

T975 = {2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}


def load(paths):
    rows = []
    files = sorted(sum((glob.glob(p) for p in paths), []))
    for g in files:
        if g.endswith(".parquet"):
            import pyarrow.parquet as pq
            rows.extend(pq.read_table(g).to_pylist())
        else:
            with open(g) as f:
                for line in f:
                    if line.strip():
                        rows.append(json.loads(line))
    if not rows:
        sys.exit(f"no rows in {path}")
    return rows


def mean_ci(xs):
    n = len(xs)
    if n == 0:
        return 0.0, 0.0, 0
    m = sum(xs) / n
    if n == 1:
        return m, 0.0, 1
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))
    t = T975.get(n - 1, 1.96)
    return m, t * sd / math.sqrt(n), n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="+", help="glob(s): *.parquet or *.jsonl")
    ap.add_argument("--by-depth", action="store_true")
    ap.add_argument("--markdown", help="also write markdown table here")
    args = ap.parse_args()

    rows = load(args.path)
    keyf = (lambda r: (r["config_label"], r["engine"], r["task"], r["target_ctx"], r["depth"])) \
        if args.by_depth else \
        (lambda r: (r["config_label"], r["engine"], r["task"], r["target_ctx"]))

    groups = {}
    for r in rows:
        if r.get("error"):
            continue
        groups.setdefault(keyf(r), []).append(r)

    lines = []
    hdr = (f"| {'config':<10} {'engine':<9} {'task':<11} {'ctx':>7} {'depth':>5} | "
           f"{'score':>6} {'±95%':>5} {'n':>3} | {'ttft med':>8} {'wall med':>8} {'finish-length':>13} |")
    if not args.by_depth:
        hdr = f"| {'config':<10} {'engine':<9} {'task':<11} {'ctx':>7} | {'score':>6} {'±95%':>5} {'n':>3} | {'ttft med':>8} {'wall med':>8} {'finish-length':>13} |"
    lines.append(hdr)
    lines.append("-" * len(hdr))

    for key in sorted(groups, key=lambda k: (str(k[0]), str(k[1]), str(k[2]),
                                             -(k[3] if isinstance(k[3], int) else 0),
                                             str(k[4]) if len(k) > 4 else "")):
        rs = groups[key]
        scores = [r["score"] for r in rs if r.get("score") is not None]
        m, ci, n = mean_ci(scores)
        tts = sorted(r["ttft_s"] for r in rs if r.get("ttft_s"))
        wls = sorted(r["wall_s"] for r in rs if r.get("wall_s"))
        fl = sum(1 for r in rs if r.get("finish_reason") == "length")
        ttft_med = tts[len(tts) // 2] if tts else float("nan")
        wall_med = wls[len(wls) // 2] if wls else float("nan")
        depth = key[4] if args.by_depth else ""
        dcell = f"{depth:>5}" if args.by_depth else ""
        lines.append(
            f"| {key[0]:<10} {key[1]:<9} {key[2]:<11} {key[3]:>7} {dcell} | "
            f"{m:>6.3f} {ci:>5.3f} {n:>3} | {ttft_med:>8.1f} {wall_med:>8.1f} {fl:>13} |")

    out = "\n".join(lines)
    print(out)
    if args.markdown:
        with open(args.markdown, "w") as f:
            f.write(out + "\n")


if __name__ == "__main__":
    main()
