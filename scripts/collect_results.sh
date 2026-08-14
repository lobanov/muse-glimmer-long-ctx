#!/usr/bin/env bash
# Collect ALL eval results into a single snapshot (docs/results-snapshot.md).
# Safe to run any time — tools are read-only over whatever parquet/jsonl exists.
# Run on host: bash scripts/collect_results.sh   (executes inside the dev container)
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
DEV=muse-glimmer-long-ctx-dev-1
OUT=docs/results-snapshot.md
E=outputs/eval

{
echo "# Results snapshot — $(date '+%F %T')"
echo
echo "Data files present:"
for f in $E/*.jsonl; do
  [ -f "$f" ] || continue
  n=$(wc -l < "$f")
  echo "- \`$f\` ($n rows)"
done
echo

# ---- §3 stock grids ------------------------------------------------------------
STOCK_FILES=$(ls $E/stock_vllm_le128k.jsonl $E/stock_vllm_gt128k.jsonl 2>/dev/null | tr '\n' ' ')
if [ -n "$STOCK_FILES" ]; then
  echo "## §3 stock baseline — score by task × ctx (mean ± 95% CI (n))"
  echo '```'
  docker exec "$DEV" bash -c "cd /workspaces/muse-glimmer-long-ctx && \
    python3 evals/harness/summarize.py $STOCK_FILES" 2>/dev/null
  echo '```'
  echo "## §3 retention vs 128k + decision rule"
  echo '```'
  docker exec "$DEV" bash -c "cd /workspaces/muse-glimmer-long-ctx && \
    python3 evals/harness/retention.py $STOCK_FILES --ref 128000" 2>/dev/null
  echo '```'
fi

# ---- community suites ----------------------------------------------------------
for s in nolima longbench_v2 longcodeqa infbench agentmem synth3; do
  f="$E/suite_${s}.jsonl"
  [ -f "$f" ] || continue
  echo "## suite: $s"
  echo '```'
  docker exec "$DEV" bash -c "cd /workspaces/muse-glimmer-long-ctx && \
    python3 evals/harness/summarize.py '$f'" 2>/dev/null
  echo '```'
done

# ---- §4 arm comparison ---------------------------------------------------------
ARM_FILES=$(ls $E/arm_*.jsonl 2>/dev/null | tr '\n' ' ')
if [ -n "$ARM_FILES" ]; then
  echo "## §4 zero-shot arms vs stock (Δ in points; * = beyond CI and >3pts)"
  echo '```'
  docker exec "$DEV" bash -c "cd /workspaces/muse-glimmer-long-ctx && \
    python3 evals/harness/compare.py $ARM_FILES $STOCK_FILES --ref stock \
    --tasks niah,semantic" 2>/dev/null
  echo '```'
fi

# ---- counting error anatomy (most sensitive degradation metric) ------------------
CNT_FILES=$(ls $E/stock_vllm_*.jsonl $E/run1_vllm*.jsonl $E/abl_*.jsonl 2>/dev/null | tr '\n' ' ')
if [ -n "$CNT_FILES" ]; then
  echo "## counting error anatomy (off-by-one undercount = attention dilution)"
  echo '```'
  docker exec -i "$DEV" python3 - $(for f in $CNT_FILES; do echo "/workspaces/muse-glimmer-long-ctx/$f"; done) <<'PY'
import json, re, sys
from collections import Counter, defaultdict
pat = re.compile(r"got (\d+) want (\d+)")
stats = defaultdict(Counter)
for path in sys.argv[1:]:
    try:
        fh = open(path)
    except FileNotFoundError:
        continue
    for line in fh:
        try: r = json.loads(line)
        except Exception: continue
        if r.get("error") or r["task"] != "counting": continue
        lab, ctx = r["config_label"], r["target_ctx"]
        if r["score"] == 1.0:
            stats[(lab, ctx)]["exact"] += 1
        else:
            m = pat.search(r.get("detail") or "")
            if not m:
                stats[(lab, ctx)]["other"] += 1
            else:
                got, want = int(m.group(1)), int(m.group(2))
                d = got - want
                stats[(lab, ctx)]["under1" if d == -1 else
                       ("underN" if d < -1 else "over")] += 1
for (lab, ctx), c in sorted(stats.items(), key=lambda kv: (kv[0][0], kv[0][1])):
    tot = sum(c.values())
    print(f"{lab:<22} ctx={ctx:>7}: exact={c['exact']:>3}/{tot} "
          f"under-1={c['under1']} under-N={c['underN']} over={c['over']} other={c['other']}")
PY
  echo '```'
fi

# ---- trained model (§8: run1 + ablations) ---------------------------------------
RUN_FILES=$(ls $E/run1_vllm*.jsonl $E/abl_*.jsonl 2>/dev/null | tr '\n' ' ')
if [ -n "$RUN_FILES" ]; then
  echo "## §8 trained vs stock (Δ pts; * = beyond CI and >3pts)"
  echo '```'
  docker exec "$DEV" bash -c "cd /workspaces/muse-glimmer-long-ctx && \
    python3 evals/harness/compare.py $RUN_FILES $STOCK_FILES --ref stock \
    --tasks niah,semantic,multihop,abstain" 2>/dev/null
  echo '```'
  echo "## §10 failure-mode diagnostics (run1 vs stock)"
  echo '```'
  docker exec "$DEV" bash -c "cd /workspaces/muse-glimmer-long-ctx && \
    python3 evals/harness/diagnose.py $STOCK_FILES $RUN_FILES --label run1" 2>/dev/null
  echo '```'
fi

# ---- PPL probes ------------------------------------------------------------------
for ppl in $E/ppl_*.jsonl; do
  [ -f "$ppl" ] || continue
  echo "## PPL curve — $(basename "$ppl" .jsonl) (last-8k-token span)"
  echo '```'
  docker exec -i "$DEV" python3 - "$ppl" <<'PY'
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
rows.sort(key=lambda r: (r.get("config", ""), r.get("target_ctx", 0), r.get("rep", 0)))
for r in rows:
    if r.get("error"):
        print(f'{r.get("config", "?"):<12} {r.get("target_ctx", 0):>7} rep{r.get("rep", 0)}: ERROR {r["error"][:60]}')
    else:
        print(f'{r["config"]:<12} {r["target_ctx"]:>7} rep{r["rep"]}: '
              f'ppl={r["ppl"]:.4f} over n={r["n_eval"]} (prompt {r["prompt_tokens"]:,}) '
              f'[{r["wall_s"]}s]')
PY
  echo '```'
done

# ---- corpus manifest -------------------------------------------------------------
if [ -f outputs/corpus/train_v1/manifest.json ]; then
  echo "## §5 corpus manifest (train_v1)"
  echo '```json'
  cat outputs/corpus/train_v1/manifest.json
  echo '```'
fi

echo
echo "Generated by scripts/collect_results.sh — do not edit by hand."
} > "$OUT" 2>logs/collect-results.log
echo "wrote $OUT ($(wc -l < "$OUT") lines)"
grep -c '^## ' "$OUT" | xargs echo "sections:"
