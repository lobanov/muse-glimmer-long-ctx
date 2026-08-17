#!/usr/bin/env bash
# infb_bands.sh — honest-length stratified InfBench runs (goal afe6584b).
# True lengths from infbench_lengths_v3.json (task/id-keyed; v1/v2 were id-collision
# soup). Bands run via INF_MIN_TOK/INF_MAX_TOK env on the patched plugin.
# All runs: stock-524k sidecar, capability mode, max-tokens 4096 (fixes the 1024
# drain confound), n=3 per band. Greedy parity pass queued for bands reading 0.000.
# Resumable: per-band markers under logs/infbands/.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
DEV=muse-glimmer-long-ctx-dev-1
log() { echo "[$(date '+%F %T')] $*" >> logs/infbands-queue.log; }
mkdir -p logs/infbands

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
    local n; n=$(wc -l < "$out" 2>/dev/null || echo 0)
    if [ "$n" -ge 3 ]; then
        touch "$marker"; log "== $task $tag $mode: $n rows =="
    else
        log "ERROR: $task $tag $mode incomplete ($n/3) — NOT marked"
    fi
}

log "== infb_bands start (pid $$) =="

# codedebug (true lengths 74k-200k)
run_band infb_codedebug 100000 140000 100-140k
run_band infb_codedebug 140000 170000 140-170k
run_band infb_codedebug 170000 200000 170-200k

# bookmc (true lengths 69k-746k; serve cap 524288 -> bands to 500k)
run_band infb_bookmc 100000 140000 100-140k
run_band infb_bookmc 140000 180000 140-180k
run_band infb_bookmc 180000 220000 180-220k
run_band infb_bookmc 220000 260000 220-260k
run_band infb_bookmc 260000 320000 260-320k
run_band infb_bookmc 320000 400000 320-400k
run_band infb_bookmc 400000 500000 400-500k

# greedy parity on bands whose sampled mean is 0.000 (artifact check)
python3 - <<'PY' > logs/infbands/zero_bands.txt
import glob, json
from collections import defaultdict
for f in sorted(glob.glob("outputs/eval/infb_*_capability.jsonl")):
    rows = [json.loads(l) for l in open(f)]
    if not rows: continue
    s = sum(r["score"] for r in rows if not r.get("error"))
    if s == 0.0:
        task, tag = rows[0]["task"], f.rsplit("_", 2)[1]
        print(f"{task} {tag}")
PY
while read -r task tag; do
    [ -z "$task" ] && continue
    log "greedy confirm: $task $tag (sampled 0.000)"
    case $tag in
        100-140k) run_band $task 100000 140000 $tag parity;;
        140-170k) run_band $task 140000 170000 $tag parity;;
        170-200k) run_band $task 170000 200000 $tag parity;;
        140-180k) run_band $task 140000 180000 $tag parity;;
        180-220k) run_band $task 180000 220000 $tag parity;;
        220-260k) run_band $task 220000 260000 $tag parity;;
        260-320k) run_band $task 260000 320000 $tag parity;;
        320-400k) run_band $task 320000 400000 $tag parity;;
        400-500k) run_band $task 400000 500000 $tag parity;;
    esac
done < logs/infbands/zero_bands.txt

log "== infb_bands complete =="
echo "done $(date '+%F %T')" > logs/infbands-queue.done
