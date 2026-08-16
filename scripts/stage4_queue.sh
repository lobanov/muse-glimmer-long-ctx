#!/usr/bin/env bash
# Stage-4 queue v2.1 (host-side): PLAN §4 REVISION (commit 4c875bc) implementation.
# Arms repurposed: zero-training treatment for GOAL criterion 7 on the WEAK AXES
# (counting, cwe, official NoLiMa) — extrapolation rescue is dead (stock 1.000 to 512k).
#
# Per arm (qk4.3, qk5.0 — bracket; 4.1/4.6 dropped):
#   pre      : stock weak-axis enrichment — counting,cwe @ {128k,256k} ×5 reps
#              (audit F-1.1: 3 reps left 10/12 pooled pairs ceiling-bound, making
#               the +10pt gate require a perfect 12/12; n=20 pairs restores headroom)
#   primary   : counting, cwe, nolima @ {128k, 256k} × depth 0.5 × 5 reps
#               (nolima instances are cell-seed-deterministic → paired arm-vs-stock
#                per cell; only pooled/paired reads, never cross-length trends)
#   infbench : infb_codedebug, infb_bookmc @ {128k,256k} ×3 reps (audit F-5.1:
#              discovered weak axes get arm coverage, not just stock rows)
#   harm check: niah @ 64k × 5 reps (saturated task — can only show damage) and
#              VETOES the extension signal (audit F-1.2: was logged, never consumed)
#   extension : 512k weak-axis cells ONLY on signal (pooled paired delta ≥ +10pts
#               vs stock on counting/cwe @128k+256k, no harm) — PLAN: refine only on
#               signal. Arm markers conditional on expected rows (audit F-1.3).
# yarn4 control: counting, cwe, niah @ 128k × 3 reps (cheap; expected near-inert)
# Ends serving stock for the stage5+ chain.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
DEV=muse-glimmer-long-ctx-dev-1
COMPOSE="docker compose -f .devcontainer/docker-compose.yml"
log() { echo "[$(date '+%F %T')] $*" >> logs/stage4-queue.log; }
source "$ROOT/scripts/progress_lib.sh"
AI=0

log "stage4-v2.1 armed (pid $$); waiting for stage3"
progress_waiting "waiting for stage3 (agentmem+PPL)"
while [ ! -f logs/stage3-queue.done ]; do progress_waiting "waiting for stage3 (agentmem+PPL)"; sleep 300; done


serve_arm() {  # serve_arm <path> — polls until the SERVED ROOT matches the request.
    # (race found 2026-08-16: a stale container from a killed watcher can be serving
    #  the previous arm; reachability alone once mislabeled qk4.3 rows as stock)
    $COMPOSE --profile inference stop vllm >/dev/null 2>&1; sleep 10
    VLLM_MODEL=$1 VLLM_MAX_MODEL_LEN=524288 \
        $COMPOSE --profile inference up -d vllm >/dev/null 2>&1
    for _ in $(seq 1 80); do
        R=$(docker exec "$DEV" curl -s --max-time 3 http://vllm:8000/v1/models 2>/dev/null \
            | grep -o "\"root\":\"$1\"" | head -1)
        [ -n "$R" ] && return 0
        sleep 30
    done
    return 1
}

weak_signal() {  # weak_signal <arm> -> "yes"/"no": pooled paired delta ≥ +10pts
                 # on counting/cwe @128k+256k vs stock AND no harm on niah@64k
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
          glob.glob("outputs/eval/stock_cwe.jsonl") +
          glob.glob("outputs/eval/stock_weak5.jsonl"))
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
# harm veto (audit F-1.2): niah@64k is saturated at stock 1.000 — a material drop
# means the arm damages retrieval; PLAN: such an arm is worse than useless.
harm = [v for k, v in a.items() if k[0] == "niah" and k[1] == 64000]
if harm:
    hm = sum(harm)/len(harm)
    print(f"  [{arm}] HARM niah@64k: {hm:.3f} (n={len(harm)}, veto below 0.900)",
          file=sys.stderr)
    if hm < 0.9 and signal:
        signal = False
        print(f"  [{arm}] HARM VETO: signal suppressed", file=sys.stderr)
print("yes" if signal else "no")
PY
}

# ---- stock weak-axis enrichment (audit F-1.1): stock pooled cells had 3 reps with
# 10/12 ceiling-bound → the +10pt gate required a perfect 12/12. Re-run at 5 reps
# (same cell_seeds as arm reps 0-4 → paired) while stock is still serving.
if [ ! -f logs/stage4-stockweak5.done ]; then
    # verify STOCK is actually the served root (not merely that something answers)
    docker exec "$DEV" curl -s --max-time 3 http://vllm:8000/v1/models 2>/dev/null \
        | grep -q '"root":"/arms/stock-524k"' \
        || serve_arm /arms/stock-524k || log "ERROR: stock not serving for enrichment"
    log "== stock weak-axis enrichment (counting,cwe @128k+256k ×5 reps) =="
    progress_step 1 5 "stock weak-axis enrichment (audit F-1.1)"
    docker exec "$DEV" bash -c "cd /workspaces/muse-glimmer-long-ctx && \
        python3 evals/harness/run_eval.py --engine vllm --base-url http://vllm:8000/v1 \
        --config-label stock --tasks counting,cwe \
        --ctx 128000,256000 --depths 0.5 --reps 5 \
        --mode capability --max-tokens 4096 \
        --out outputs/eval/stock_weak5.jsonl --write-parquet" \
        >> logs/stage4-stockweak5-grid.log 2>&1 || log "WARN: stock enrichment partial"
    touch logs/stage4-stockweak5.done
fi

for ARM in qk4.3 qk5.0; do
    AI=$((AI+1))
    [ -f "logs/stage4-$ARM.done" ] && { log "$ARM done (skip)"; continue; }
    log "== arm $ARM: serving =="
    progress_step $((AI+1)) 5 "arm $ARM grids"
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
    log "== arm $ARM: infbench weak axes (infb_codedebug,infb_bookmc @128k+256k ×3 — audit F-5.1) =="
    docker exec "$DEV" bash -c "cd /workspaces/muse-glimmer-long-ctx && \
        python3 evals/harness/run_eval.py --engine vllm --base-url http://vllm:8000/v1 \
        --config-label $ARM --plugin infbench --tasks infb_codedebug,infb_bookmc \
        --ctx 128000,256000 --depths 0.5 --reps 3 \
        --mode capability --max-tokens 4096 \
        --out outputs/eval/arm_${ARM}.jsonl" \
        >> logs/stage4-$ARM-grid.log 2>&1 || log "WARN: infbench axis partial"
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
    # expected: primary 30 (3×2×5) + harm 5 + infbench 12 (2×2×3) = 47,
    # +20 if the 512k extension ran (counting,cwe @512k ×5). Audit F-1.3: a failed/
    # empty grid must NOT be marked done (permanent silent skip on re-arm otherwise).
    EXP=$((47 + $([ "$SIG" = "yes" ] && echo 20 || echo 0)))
    if [ "$N" -ge "$EXP" ]; then
        log "== arm $ARM complete: $N/$EXP rows =="
        touch "logs/stage4-$ARM.done"
    else
        log "ERROR: arm $ARM incomplete: $N/$EXP rows — NOT marking done (resumable retry)"
        echo "failed $N/$EXP $(date '+%F %T')" > "logs/stage4-$ARM.failed"
    fi
done

# yarn4 control (cheap; expected near-inert; includes weak-axis cells so it can
# actually falsify the inertness prediction, not just confirm a saturated ceiling)
if [ ! -f logs/stage4-yarn4.done ]; then
    log "== yarn4 control =="
progress_step 4 5 "yarn4 control probe"
    serve_arm /arms/yarn4 || log "ERROR: yarn4 never served"
    docker exec "$DEV" bash -c "cd /workspaces/muse-glimmer-long-ctx && \
        python3 evals/harness/run_eval.py --engine vllm --base-url http://vllm:8000/v1 \
        --config-label yarn4 --tasks counting,cwe,niah --ctx 128000 --depths 0.5 --reps 3 \
        --mode capability --max-tokens 4096 \
        --out outputs/eval/arm_yarn4.jsonl" >> logs/stage4-yarn4-grid.log 2>&1 \
      || log "WARN: yarn4 partial"
    NY=$(wc -l < outputs/eval/arm_yarn4.jsonl 2>/dev/null || echo 0)
    if [ "$NY" -ge 9 ]; then
        touch logs/stage4-yarn4.done
    else
        log "ERROR: yarn4 incomplete: $NY/9 rows"
        echo "failed $NY/9 $(date '+%F %T')" > logs/stage4-yarn4.failed
    fi
fi

# leave stock serving for the stage5+ chain
serve_arm /arms/stock-524k || log "WARN: could not restore stock serving"
progress_step 5 5 "restoring stock serving"
echo "done $(date '+%F %T')" > logs/stage4-queue.done
log "stage4-v2.1 complete: $(cat logs/stage4-queue.done)"
