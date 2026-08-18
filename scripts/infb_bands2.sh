#!/usr/bin/env bash
# infb_bands2.sh — bookmc TRUE-length band re-run (goal afe6584b).
# bands1 bookmc selections were soup-selected (v1 bare-id cache = codedebug lengths);
# invalidated to *.soup-invalid on 2026-08-17. These bands use v3 task/id keys
# (plugin now v3-only), edges matched to the real distribution:
#   [100,160) [160,220) [220,300) [300,400) [400,510)   n=3 each, + parity on 0.000.
# Chains after ruler_gt128k (GPU serial: synth -> ruler -> bookmc2).
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
DEV=muse-glimmer-long-ctx-dev-1
log() { echo "[$(date '+%F %T')] $*" >> logs/infbands2-queue.log; }

# GPU serialization (goal d56ed95d): fire only after ruler AND synth complete —
# concurrent 512k-prompt lanes cause vLLM preemption storms; serial is safer.
while [ ! -f logs/ruler-gt128k.done ] || [ ! -f logs/synth-gt128k.done ]; do
    sleep 300
done
log "== infb_bands2 start (pid $$); ruler done=$([ -f logs/ruler-gt128k.done ] && echo yes || echo NO) =="

run_band() {  # run_band <task> <min> <max> <tag> [mode]
    local task=$1 min=$2 max=$3 tag=$4 mode=${5:-capability}
    local marker="logs/infbands/${task}_${tag}_${mode}.done"
    [ -f "$marker" ] && { log "$task $tag $mode done (skip)"; return 0; }
    local out="outputs/eval/infb_${task}_${tag}_${mode}.jsonl"
    log "== $task band $tag [$min,$max) mode=$mode =="
    docker exec -e INF_MIN_TOK=$min -e INF_MAX_TOK=$max "$DEV" bash -c \
        "cd /workspaces/muse-glimmer-long-ctx && \
         python3 evals/harness/run_eval.py --engine vllm --base-url http://vllm:8000/v1 \
         --config-label stock --plugin infbench --tasks $task \
         --ctx 524288 --depths 0.5 --reps 3 \
         --mode $mode --max-tokens 4096 --out $out" \
        >> logs/infbands/${task}_${tag}_${mode}.log 2>&1 \
        || log "WARN: $task $tag $mode grid partial"
    local n; n=$(grep -c '"error": null' "$out" 2>/dev/null || echo 0)
    if [ "$n" -ge 3 ]; then
        touch "$marker"; log "== $task $tag $mode: $n rows =="
    else
        log "ERROR: $task $tag $mode incomplete ($n/3) — NOT marked"
    fi
}

run_band infb_bookmc 100000 160000 100-160k
run_band infb_bookmc 160000 220000 160-220k
run_band infb_bookmc 220000 300000 220-300k
run_band infb_bookmc 300000 400000 300-400k
run_band infb_bookmc 400000 510000 400-510k

# greedy parity on any sampled-0.000 band
python3 - > logs/infbands/zero_bands2.txt <<'PY'
import glob, json
for f in sorted(glob.glob("outputs/eval/infb_infb_bookmc_*_capability.jsonl")):
    rows = [json.loads(l) for l in open(f)]
    if rows and sum(r["score"] for r in rows if not r.get("error")) == 0.0:
        print("infb_bookmc", f.rsplit("_", 2)[1])
PY
while read -r task tag; do
    [ -z "$task" ] && continue
    log "greedy confirm: $task $tag (sampled 0.000)"
    case $tag in
        100-160k) run_band $task 100000 160000 $tag parity;;
        160-220k) run_band $task 160000 220000 $tag parity;;
        220-300k) run_band $task 220000 300000 $tag parity;;
        300-400k) run_band $task 300000 400000 $tag parity;;
        400-510k) run_band $task 400000 510000 $tag parity;;
    esac
done < logs/infbands/zero_bands2.txt

log "== infb_bands2 complete =="
echo "done $(date '+%F %T')" > logs/infbands2-queue.done
