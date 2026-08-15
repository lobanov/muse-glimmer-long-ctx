#!/usr/bin/env bash
# Stage-4 queue v2.1 (host-side): PLAN §4 REVISION (commit 4c875bc) implementation.
# Arms repurposed: zero-training treatment for GOAL criterion 7 on the WEAK AXES
# (counting, cwe, official NoLiMa) — extrapolation rescue is dead (stock 1.000 to 512k).
#
# Per arm (qk4.3, qk5.0 — bracket; 4.1/4.6 dropped):
#   primary   : counting, cwe, nolima @ {128k, 256k} × depth 0.5 × 5 reps
#               (nolima instances are cell-seed-deterministic → paired arm-vs-stock
#                per cell; only pooled/paired reads, never cross-length trends)
#   harm check: niah @ 64k × 5 reps (saturated task — can only show damage)
#   extension : 512k weak-axis cells ONLY if the arm shows signal (pooled paired
#               delta ≥ +10pts vs stock on counting/cwe @128k+256k) — PLAN: refine
#               only on signal
# yarn4 control: counting, cwe, niah @ 128k × 3 reps (cheap; expected near-inert)
# Ends serving stock for the stage5+ chain.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
DEV=muse-glimmer-long-ctx-dev-1
COMPOSE="docker compose -f .devcontainer/docker-compose.yml"
log() { echo "[$(date '+%F %T')] $*" >> logs/stage4-queue.log; }

log "stage4-v2.1 armed (pid $$); waiting for stage3"
while [ ! -f logs/stage3-queue.done ]; do sleep 300; done

serve_arm() {  # serve_arm <path>
    $COMPOSE --profile inference stop vllm >/dev/null 2>&1; sleep 10
    VLLM_MODEL=$1 VLLM_MAX_MODEL_LEN=524288 \
        $COMPOSE --profile inference up -d vllm >/dev/null 2>&1
    for _ in $(seq 1 80); do
        docker exec "$DEV" curl -s --max-time 3 http://vllm:8000/v1/models >/dev/null 2>&1 && return 0
        sleep 30
    done
    return 1
}

weak_signal() {  # weak_signal <arm> -> prints "yes"/"no" (pooled paired delta ≥ +10pts
                 # on counting or cwe at 128k+256k vs stock)
    python3 - "$1" <<'PY'
import glob, json, sys
from collections import defaultdict
arm = sys.argv[1]
def cells(label, files):
    d = defaultdict(dict)  # (task, ctx, depth, rep) -> score (cell-seed-paired keys)
    for f in files:
        try: fh = open(f)
        except FileNotFoundError: continue
        for line in fh:
            try: r = json.loads(line)
            except Exception: continue
            if r.get("error") or r.get("score") is None: continue
            if r["config_label"] != label: continue
            d[(r["task"], r["target_ctx"], r["depth"], r["rep"])] = r["score"]
    return d
s = cells("stock", glob.glob("outputs/eval/stock_vllm_le128k.jsonl") +
          glob.glob("outputs/eval/stock_vllm_gt128k.jsonl") +
          glob.glob("outputs/eval/stock_cwe.jsonl"))
a = cells(arm, glob.glob(f"outputs/eval/arm_{arm}.jsonl"))
signal = False
# NOTE (audit 2026-08-15): stock rows have 3 reps/cell -> per-task pairs max at 6
# (2 ctx x 3 reps); the old `len(pairs) >= 10` made this branch UNREACHABLE. Now pooled
# across counting+cwe with a floor of 5 pairs; per-task and pooled deltas both logged.
allpairs = []
for task in ("counting", "cwe"):
    pairs = [(v, s[k]) for k, v in a.items()
             if k[0] == task and k[1] in (128000, 256000) and k in s]
    allpairs += pairs
    if len(pairs) >= 3:
        am = sum(p[0] for p in pairs)/len(pairs); sm = sum(p[1] for p in pairs)/len(pairs)
        print(f"  [{arm}] {task}: paired {am:.3f} vs stock {sm:.3f} "
              f"({(am-sm)*100:+.1f} pts, n={len(pairs)})", file=sys.stderr)
if len(allpairs) >= 5:
    am = sum(p[0] for p in allpairs)/len(allpairs)
    sm = sum(p[1] for p in allpairs)/len(allpairs)
    if am - sm >= 0.10:
        signal = True
        print(f"  [{arm}] POOLED: {am:.3f} vs {sm:.3f} ({(am-sm)*100:+.1f} pts, "
              f"n={len(allpairs)}) -> signal", file=sys.stderr)
print("yes" if signal else "no")
PY
}

for ARM in qk4.3 qk5.0; do
    [ -f "logs/stage4-$ARM.done" ] && { log "$ARM done (skip)"; continue; }
    log "== arm $ARM: serving =="
    serve_arm /arms/$ARM || { log "ERROR: vLLM never came up for $ARM"; continue; }
    log "== arm $ARM: primary grids (counting,cwe,nolima @128k,256k ×5 reps) =="
    docker exec "$DEV" bash -c "cd /workspaces/muse-glimmer-long-ctx && \
        python3 evals/harness/run_eval.py --engine vllm --base-url http://vllm:8000/v1 \
        --config-label $ARM --plugin nolima --tasks counting,cwe,nolima \
        --ctx 128000,256000 --depths 0.5 --reps 5 \
        --mode capability --max-tokens 4096 \
        --out outputs/eval/arm_${ARM}.jsonl --write-parquet" \
        >> logs/stage4-$ARM-grid.log 2>&1 \
      || log "WARN: primary grid for $ARM partial"
    log "== arm $ARM: harm check (niah @64k ×5) =="
    docker exec "$DEV" bash -c "cd /workspaces/muse-glimmer-long-ctx && \
        python3 evals/harness/run_eval.py --engine vllm --base-url http://vllm:8000/v1 \
        --config-label $ARM --tasks niah --ctx 64000 --depths 0.5 --reps 5 \
        --mode capability --max-tokens 4096 \
        --out outputs/eval/arm_${ARM}.jsonl" \
        >> logs/stage4-$ARM-grid.log 2>&1 || log "WARN: harm check partial"
    # 512k extension only on signal (PLAN: refine only on signal)
    SIG=$(weak_signal $ARM 2>>logs/stage4-queue.log)
    log "== arm $ARM: 512k extension decision: $SIG =="
    if [ "$SIG" = "yes" ]; then
        docker exec "$DEV" bash -c "cd /workspaces/muse-glimmer-long-ctx && \
            python3 evals/harness/run_eval.py --engine vllm --base-url http://vllm:8000/v1 \
            --config-label $ARM --tasks counting,cwe \
            --ctx 512000 --depths 0.5 --reps 5 \
            --mode capability --max-tokens 4096 \
            --out outputs/eval/arm_${ARM}.jsonl --write-parquet" \
            >> logs/stage4-$ARM-grid.log 2>&1 || log "WARN: 512k extension partial"
    fi
    N=$(wc -l < outputs/eval/arm_${ARM}.jsonl 2>/dev/null || echo 0)
    log "== arm $ARM complete: $N rows =="
    touch "logs/stage4-$ARM.done"
done

# yarn4 control (cheap; expected near-inert; includes weak-axis cells so it can
# actually falsify the inertness prediction, not just confirm a saturated ceiling)
if [ ! -f logs/stage4-yarn4.done ]; then
    log "== yarn4 control =="
    serve_arm /arms/yarn4 || log "ERROR: yarn4 never served"
    docker exec "$DEV" bash -c "cd /workspaces/muse-glimmer-long-ctx && \
        python3 evals/harness/run_eval.py --engine vllm --base-url http://vllm:8000/v1 \
        --config-label yarn4 --tasks counting,cwe,niah --ctx 128000 --depths 0.5 --reps 3 \
        --mode capability --max-tokens 4096 \
        --out outputs/eval/arm_yarn4.jsonl" >> logs/stage4-yarn4-grid.log 2>&1 \
      || log "WARN: yarn4 partial"
    touch logs/stage4-yarn4.done
fi

# leave stock serving for the stage5+ chain
serve_arm /arms/stock-524k || log "WARN: could not restore stock serving"
echo "done $(date '+%F %T')" > logs/stage4-queue.done
log "stage4-v2.1 complete: $(cat logs/stage4-queue.done)"
