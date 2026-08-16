#!/usr/bin/env python3
"""§4 evidence pack — paired arm-vs-stock reads for the train1 approval decision.

Generates docs/phase4-evidence.md FROM DISK (never hand-typed numbers; audit F5 rule).
Pairing: cell_seed-identical instances (arm rep k == stock rep k). Stock source is
weak5-CONSISTENT (stock_weak5.jsonl overrides older files for weak-axis cells, matching
the gate's file order) — do not switch to first-wins: it mixes sampling sessions
incoherently (bug caught 2026-08-17).

Stats honesty (see STATUS reliability note): per-cell n=5 reads carry ±22pt binomial SE;
McNemar exact p on discordant pairs is printed so nobody over-reads a single cell.
Cells with ≥2 discordant pairs are flagged GREEDY-CONFIRM (pre-planned confirmation
lane before any approval action).

Usage: python3 scripts/stage4_evidence.py   (write: docs/phase4-evidence.md)
"""
import glob
import json
import math
import sys
from collections import defaultdict

ROOT = __file__.rsplit("/", 2)[0]
E = f"{ROOT}/outputs/eval"
# weak5 LAST: enriched stock owns weak-axis cells (gate semantics)
STOCK_FILES = ["stock_vllm_le128k.jsonl", "stock_vllm_gt128k.jsonl", "stock_cwe.jsonl",
               "suite_nolima.jsonl", "suite_infbench.jsonl", "stock_weak5.jsonl"]
ARMS = ["qk4.3", "qk5.0", "yarn4"]
GATE_TASKS = ("counting", "cwe")
GATE_CTXS = (128000, 256000)


def cells(path):
    d = {}
    try:
        fh = open(path)
    except FileNotFoundError:
        return d
    for line in fh:
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("error") or r.get("score") is None:
            continue
        d[(r["task"], r["target_ctx"], r["depth"], r["rep"], r["config_label"])] = r["score"]
    return d


def mcnemar_exact(b01, b10):
    """two-sided exact binomial test on discordant pairs"""
    n = b01 + b10
    if n == 0:
        return 1.0
    k = max(b01, b10)
    p = sum(math.comb(n, i) for i in range(k, n + 1)) / 2 ** n
    return min(1.0, 2 * p)


def main():
    stock = {}
    for f in STOCK_FILES:
        for (t, c, d, rep, lbl), v in cells(f"{E}/{f}").items():
            if lbl == "stock":
                stock[(t, c, d, rep)] = v
    lines = ["# §4 zero-shot arms — evidence pack (generated; do not hand-edit)",
             "", "All reads paired on cell_seed-identical instances.",
             "Per-cell n=5 → ±22pt binomial SE; read p before believing any single cell.",
             ""]
    for arm in ARMS:
        a = {(t, c, d, rep): v for (t, c, d, rep, lbl), v in cells(f"{E}/arm_{arm}.jsonl").items()
             if lbl == arm}
        if not a:
            continue
        lines += [f"## arm {arm}", "",
                  "| task@ctx | arm | stock | Δpts | discord (arm+/arm−) | McNemar p | flag |",
                  "|---|---|---|---|---|---|---|"]
        pool_a, pool_s = [], []
        for (t, c, d, rep) in sorted(a):
            if d != 0.5 or (t, c, d, rep) not in stock:
                continue
            s = stock[(t, c, d, rep)]
            cells_key = (t, c)
            # accumulate per-cell in second pass below; here collect pairs
            pool_a.append((t, c, a[(t, c, d, rep)], s))
        by_cell = defaultdict(list)
        for t, c, av, sv in pool_a:
            by_cell[(t, c)].append((av, sv))
        for (t, c), pairs in sorted(by_cell.items()):
            ah, sh = sum(p[0] for p in pairs), sum(p[1] for p in pairs)
            n = len(pairs)
            b01 = sum(1 for av, sv in pairs if sv == 0 and av == 1)
            b10 = sum(1 for av, sv in pairs if sv == 1 and av == 0)
            p = mcnemar_exact(b01, b10)
            flag = "**greedy-confirm**" if (b01 + b10) >= 2 else ""
            lines.append(f"| {t}@{c//1000}k | {ah:.0f}/{n} | {sh:.0f}/{n} | "
                         f"{(ah/n-sh/n)*100:+.0f} | {b01}/{b10} | {p:.2f} | {flag} |")
        gp = [(av, sv) for t, c, av, sv in pool_a if t in GATE_TASKS and c in GATE_CTXS]
        if gp:
            am = sum(p[0] for p in gp) / len(gp)
            sm = sum(p[1] for p in gp) / len(gp)
            harm = [(t, c, av, sv) for t, c, av, sv in pool_a if t == "niah" and c == 64000]
            hm = sum(h[2] for h in harm) / len(harm) if harm else None
            lines += ["", f"- **gate pool**: arm {am:.3f} vs stock {sm:.3f} "
                      f"({(am-sm)*100:+.1f} pts, n={len(gp)}; needs ≥ +10.0)",
                      f"- **harm niah@64k**: {hm:.3f} (n={len(harm)})" if harm else "- harm: none",
                      ]
        lines.append("")
    harm_note = ("Rules: 512k extension fired only on pooled ≥ +10 pts AND harm ≥ 0.9; "
                 "qk training override needs ≥2 cells with ≥5-rep wins > +15 pts AND harm ok. "
                 "greedy-confirm cells must be re-run greedily before any approval cites them.")
    lines += ["---", harm_note, ""]
    out = "\n".join(lines)
    print(out)
    with open(f"{ROOT}/docs/phase4-evidence.md", "w") as f:
        f.write(out)
    print(f"\nwritten: docs/phase4-evidence.md", file=sys.stderr)


if __name__ == "__main__":
    main()
