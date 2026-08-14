#!/usr/bin/env python3
"""PLAN §3 decision-rule analysis: retention curves + verdict from eval Parquet/JSONL.

Retention(x) = mean score @ ctx x / mean score @ reference ctx (default 128k, or the
largest available <=128k). Verdict per PLAN §3: if stock retention at 256k+ stays >= 85%
on retrieval tasks, the project pivots from "teach extrapolation" to "strengthen,
qualify, deploy" (training becomes optional/targeted).

Usage:
  python3 evals/harness/retention.py 'outputs/eval/stock_vllm_le128k.parquet' \
      [more.parquet ...] [--ref 128000] [--markdown docs/_retention.md] [--label stock]

Retrieval task set (per task design): niah, niah_multi, multihop, semantic (+ nolima).
"""
import argparse
import glob
import json
import math
import sys

RETRIEVAL = {"niah", "niah_multi", "multihop", "semantic", "nolima"}
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
        return float("nan"), float("nan"), 0
    m = sum(xs) / n
    if n < 2:
        return m, 0.0, n
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))
    return m, T975.get(n - 1, 1.96) * sd / math.sqrt(n), n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--ref", type=int, default=128000)
    ap.add_argument("--label", default="stock")
    ap.add_argument("--markdown")
    args = ap.parse_args()

    rows = load(args.paths)
    if not rows:
        sys.exit("no scored rows")
    labels = sorted({r["config_label"] for r in rows})
    lines = [f"# Retention analysis — {args.label} ({', '.join(labels)})",
             f"rows: {len(rows)}; reference ctx = {args.ref}", ""]

    verdicts = []
    for label in labels:
        rs = [r for r in rows if r["config_label"] == label]
        tasks = sorted({r["task"] for r in rs})
        ctxs = sorted({r["target_ctx"] for r in rs})
        ref_ctx = max((c for c in ctxs if c <= args.ref), default=None)
        if ref_ctx is None:
            lines.append(f"({label}: no reference ctx <= {args.ref}; skipping)")
            continue
        lines.append(f"## {label}")
        lines.append(f"| {'task':<14} | " + " | ".join(f"{c:>13}" for c in ctxs) + " |")
        lines.append("|" + "---|" * (len(ctxs) + 1))
        agg = {}
        for t in tasks:
            cells = []
            for c in ctxs:
                m, ci, n = mean_ci([r["score"] for r in rs
                                    if r["task"] == t and r["target_ctx"] == c])
                agg[(t, c)] = m
                cells.append(f"{m*100:5.1f}±{ci*100:4.1f} ({n})" if n else "—")
            lines.append(f"| {t:<14} | " + " | ".join(f"{c:>13}" for c in cells) + " |")
        # retention rows (vs ref)
        lines.append("")
        lines.append(f"| {'retention':<14} | " + " | ".join(f"{c:>13}" for c in ctxs) + " |")
        lines.append("|" + "---|" * (len(ctxs) + 1))
        for t in tasks:
            ref = agg.get((t, ref_ctx))
            cells = []
            for c in ctxs:
                v = agg.get((t, c))
                cells.append("—" if (v is None or not ref or ref == 0) else f"{100*v/ref:5.1f}%")
            lines.append(f"| {t:<14} | " + " | ".join(f"{c:>13}" for c in cells) + " |")
        # decision rule (retrieval aggregate at each ctx >= 256k)
        lines.append("")
        for c in [c for c in ctxs if c >= 256_000]:
            vals = [agg[(t, c)] for t in tasks if t in RETRIEVAL and (t, c) in agg]
            refs = [agg[(t, ref_ctx)] for t in tasks if t in RETRIEVAL and (t, ref_ctx) in agg]
            if not vals or not refs:
                continue
            ret = sum(vals) / sum(refs)
            ok = ret >= 0.85
            verdicts.append((label, c, ret, ok))
            lines.append(f"- **{label} @ {c}: retrieval retention {ret*100:.1f}% "
                         f"{'≥' if ok else '<'} 85% → training "
                         f"{'OPTIONAL/targeted' if ok else 'REQUIRED for this length'}**")
    out = "\n".join(lines)
    print(out)
    if args.markdown:
        with open(args.markdown, "w") as f:
            f.write(out + "\n")
        print(f"\n[markdown] {args.markdown}")


if __name__ == "__main__":
    main()
