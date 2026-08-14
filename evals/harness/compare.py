#!/usr/bin/env python3
"""PLAN §8 — cross-configuration comparison: stock vs zero-shot arms vs trained adapters.

Input: any eval Parquet/JSONL set (multiple config_labels). Output: per (task, ctx)
mean-score table across configs with deltas vs a reference config (default: stock),
plus latency columns. Same CI convention as summarize.py (t-dist over resamples).

Usage:
  python3 evals/harness/compare.py 'outputs/eval/*.parquet' --ref stock \
      [--tasks niah,nolima,longbench_v2] [--markdown docs/_compare.md]
"""
import argparse
import glob
import json
import math
import sys

T975 = {2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447}


def load(paths):
    rows = []
    for pat in paths:
        for g in sorted(glob.glob(pat)):
            if g.endswith(".parquet"):
                import pyarrow.parquet as pq
                rows.extend(pq.read_table(g).to_pylist())
            else:
                with open(g) as f:
                    for line in f:
                        if line.strip():
                            rows.append(json.loads(line))
    return [r for r in rows if not r.get("error") and r.get("score") is not None]


def mean_ci(xs):
    n = len(xs)
    if n == 0:
        return float("nan"), 0.0, 0
    m = sum(xs) / n
    if n < 2:
        return m, 0.0, n
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))
    return m, T975.get(n - 1, 1.96) * sd / math.sqrt(n), n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--ref", default="stock")
    ap.add_argument("--tasks", help="comma filter")
    ap.add_argument("--markdown")
    args = ap.parse_args()

    rows = load(args.paths)
    if not rows:
        sys.exit("no scored rows")
    if args.tasks:
        keep = set(args.tasks.split(","))
        rows = [r for r in rows if r["task"] in keep]
    configs = sorted({r["config_label"] for r in rows})
    if args.ref not in configs:
        sys.exit(f"reference config {args.ref!r} not in {configs}")
    tasks = sorted({r["task"] for r in rows})
    ctxs = sorted({r["target_ctx"] for r in rows})

    lines = [f"# §8 comparison — ref: {args.ref}", f"configs: {configs}", ""]
    for t in tasks:
        lines.append(f"## {t}")
        lines.append(f"| {'ctx':>7} | " + " | ".join(f"{c:>18}" for c in configs)
                     + " | " + " | ".join(f"{'Δvs '+args.ref:>18}" for c in configs
                                           if c != args.ref) + " |")
        lines.append("|" + "---|" * (1 + len(configs) + len(configs) - 1))
        for c_ in ctxs:
            base = None
            cells, deltas = [], []
            for cfg in configs:
                m, ci, n = mean_ci([r["score"] for r in rows
                                    if r["task"] == t and r["target_ctx"] == c_
                                    and r["config_label"] == cfg])
                cells.append(f"{m*100:5.1f}±{ci*100:4.1f} ({n})" if n else "—")
                if cfg == args.ref:
                    base = m if n else None
            for cfg in configs:
                if cfg == args.ref:
                    continue
                m, ci, n = mean_ci([r["score"] for r in rows
                                    if r["task"] == t and r["target_ctx"] == c_
                                    and r["config_label"] == cfg])
                if not n or base is None:
                    deltas.append("—")
                else:
                    d = (m - base) * 100
                    sig = "*" if abs(d) > ci * 100 and abs(d) > 3 else " "
                    deltas.append(f"{d:+5.1f}{sig}")
            lines.append(f"| {c_:>7} | " + " | ".join(f"{c:>18}" for c in cells)
                         + " | " + " | ".join(f"{c:>18}" for c in deltas) + " |")
        lines.append("")
    lines.append("(± = 95% CI over resamples; Δ in points; * = |Δ| > CI and > 3 pts)")
    out = "\n".join(lines)
    print(out)
    if args.markdown:
        with open(args.markdown, "w") as f:
            f.write(out + "\n")
        print(f"\n[markdown] {args.markdown}", file=sys.stderr)


if __name__ == "__main__":
    main()
