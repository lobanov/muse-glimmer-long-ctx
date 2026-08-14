#!/usr/bin/env python3
"""PLAN §10 — diagnostics: classify the dominant failure mode from eval results.

Reads stock + run1 (+ arm) result files and stratifies per PLAN §10's decision tree:

  A. Positional errors dominate   → depth-stratified: edge depths (0.0/1.0) collapse
     while middle holds                            → YaRN/position training, local layers
  B. Global retrieval/selectivity → task-stratified: exact-lexical tasks (niah) hold
     while semantic/multihop/agentmem degrade       → distractor-heavy data, global layers
  C. Short-context regression     → ≤32k cells of run1 drop vs stock beyond CI
                                                   → replay share ↑, rank/lr ↓
  D. Reasoning-channel drain      → finish_reason=length share on failures
                                                   → max_tokens/harness fix, not model

Output: markdown verdict with the recommended §10 actions (and §9 scope hint).

Usage: python3 evals/harness/diagnose.py outputs/eval/stock_vllm_le128k.jsonl \
          outputs/eval/stock_vllm_gt128k.jsonl outputs/eval/run1_vllm*.jsonl [--label run1]
"""
import argparse
import glob
import json
import math
import sys
from collections import defaultdict

T975 = {2: 4.303, 3: 3.182, 4: 2.776}
LEXICAL = {"niah", "niah_multi", "infb_kv"}          # exact-match retrieval
SELECTIVE = {"semantic", "multihop", "agentmem", "nolima"}  # inference-heavy retrieval


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def ci(xs):
    n = len(xs)
    if n < 2:
        return 0.0
    sd = math.sqrt(sum((x - mean(xs)) ** 2 for x in xs) / (n - 1))
    return T975.get(n - 1, 1.96) * sd / math.sqrt(n)


def load(paths):
    rows = []
    for pat in paths:
        for g in sorted(glob.glob(pat)):
            for line in open(g):
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not r.get("error") and r.get("score") is not None:
                    rows.append(r)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--label", default="run1")
    ap.add_argument("--ref", default="stock")
    ap.add_argument("--markdown")
    a = ap.parse_args()

    rows = load(a.paths)
    cfg = a.label
    cells = defaultdict(list)   # (label, task, ctx, depth) -> scores
    drains = defaultdict(int)   # label -> length-finish count
    totals = defaultdict(int)
    for r in rows:
        lab = r["config_label"]
        cells[(lab, r["task"], r["target_ctx"], r["depth"])].append(r["score"])
        totals[lab] += 1
        if r.get("finish_reason") == "length":
            drains[lab] += 1
    if (a.label, *cells.keys()) and not any(k[0] == a.label for k in cells):
        sys.exit(f"no rows for label {a.label!r} in inputs")

    out = [f"# §10 diagnostics — {a.label} vs {a.ref}", ""]

    # D. reasoning drain
    for lab in (a.ref, a.label):
        if totals.get(lab):
            out.append(f"- reasoning-drain ({lab}): {drains.get(lab, 0)}/{totals[lab]} "
                       f"finish=length")

    # A. positional: edge vs middle depth at each ctx (per label)
    out.append("\n## A. depth stratification (edge 0/1.0 vs middle 0.25–0.75)")
    edge_mid = {}
    for lab in (a.ref, a.label):
        for task in sorted({k[1] for k in cells if k[0] == lab}):
            for ctx in sorted({k[2] for k in cells if k[0] == lab and k[1] == task}):
                e = [v for k, xs in cells.items()
                     if k[0] == lab and k[1] == task and k[2] == ctx
                     and k[3] in (0.0, 1.0) for v in xs]
                m = [v for k, xs in cells.items()
                     if k[0] == lab and k[1] == task and k[2] == ctx
                     and 0.25 <= k[3] <= 0.75 for v in xs]
                if e and m:
                    edge_mid[(lab, task, ctx)] = (mean(e), mean(m))
                    out.append(f"- {lab:<6} {task:<12} ctx={ctx:>7}: edge={mean(e):.3f} "
                               f"mid={mean(m):.3f}")

    # B. task stratification: lexical vs selective at shared ctx
    out.append("\n## B. task stratification (lexical vs selective)")
    strat = {}
    for lab in (a.ref, a.label):
        lex = [v for k, xs in cells.items() if k[0] == lab and k[1] in LEXICAL
               for v in xs]
        sel = [v for k, xs in cells.items() if k[0] == lab and k[1] in SELECTIVE
               for v in xs]
        if lex and sel:
            strat[lab] = (mean(lex), mean(sel))
            out.append(f"- {lab:<6}: lexical={mean(lex):.3f} (n={len(lex)}) "
                       f"selective={mean(sel):.3f} (n={len(sel)})")

    # C. short regression
    out.append("\n## C. ≤32k regression (label vs ref, shared cells)")
    regressions = []
    for k, xs in cells.items():
        if k[0] != a.label or k[2] > 32_000:
            continue
        s = cells.get((a.ref, *k[1:]))
        if s and len(xs) >= 2:
            d = mean(s) - mean(xs)
            if d > 0.03 + max(ci(xs), ci(s)):
                regressions.append((k[1:], round(mean(s), 3), round(mean(xs), 3)))
    out.append(f"- beyond-CI regressions: {regressions or 'none'}")

    # verdict
    out.append("\n## Verdict (PLAN §10 decision tree)")
    actions = []
    ref_edge = [v for (lab, t, c), (e, m) in edge_mid.items()
                if lab == a.ref for v in (e,)]
    lab_edge = [v for (lab, t, c), (e, m) in edge_mid.items()
                if lab == a.label for v in (e,)]
    pos_drop = (mean(ref_edge) - mean(lab_edge)) if (ref_edge and lab_edge) else 0.0
    if strat.get(a.ref) and strat.get(a.label):
        sel_drop = strat[a.ref][1] - strat[a.label][1]
        lex_drop = strat[a.ref][0] - strat[a.label][0]
    else:
        sel_drop = lex_drop = 0.0
    if regressions:
        actions.append("C: raise short-replay share / lower rank / shorten schedule "
                       "(PLAN §10 short-regression recipe)")
    if sel_drop > 0.05 and sel_drop >= lex_drop:
        actions.append("B: add distractor-heavy + semantic/multi-hop data, concentrate "
                       "on global NoPE layers (§9 --lora-scope global prior)")
    if pos_drop > 0.05 and pos_drop > sel_drop:
        actions.append("A: positional — revisit YaRN/position-randomized data, local "
                       "layers (§9 --lora-scope local arm)")
    if drains.get(a.label, 0) > max(2, 0.05 * totals.get(a.label, 1)):
        actions.append("D: reasoning drain — verify max_tokens / scoring contract "
                       "before model conclusions")
    if not actions:
        actions.append("no dominant failure mode at these thresholds — proceed to §9 "
                       "capacity/scope ablations")
    for x in actions:
        out.append(f"- {x}")
    out.append(f"\n(metrics: positional_drop={pos_drop:.3f}, selective_drop={sel_drop:.3f}, "
               f"lexical_drop={lex_drop:.3f}, short_regressions={len(regressions)})")

    text = "\n".join(out)
    print(text)
    if a.markdown:
        open(a.markdown, "w").write(text + "\n")


if __name__ == "__main__":
    main()
