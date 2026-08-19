#!/usr/bin/env bash
# short_le128k.sh — RULER + MRCR short-context reference curves (goal 83d7bbb9).
# RULER: 4 tasks x 32k/64k/128k x 3 reps = 36 cells.
# MRCR:  mrcr2/mrcr4 true-length bands ~32k/~64k x 3 reps = 12 cells (128k reused
#        from mrcr_gt128k.jsonl @131k — not rerun).
# Serialized behind bands2 (last >128k lane).
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
DEV=muse-glimmer-long-ctx-dev-1
log() { echo "[$(date '+%F %T')] $*" >> logs/short-le128k.log; }

while ! [ -f logs/infbands2-queue.done ]; do
    # if bands2 died without a marker but nothing is running and no ruler/mrcr/synth
    # lanes are active either, proceed (bands2 is the only >128k lane left)
    pgrep -f 'bash scripts/infb_bands2.sh' >/dev/null || break
    sleep 300
done
log "== short_le128k start (pid $$); bands2 done=$([ -f logs/infbands2-queue.done ] && echo yes || echo no) =="

run_grid() {  # run_grid <tag> <marker> <args...>
    local tag=$1 marker=$2; shift 2
    [ -f "$marker" ] && { log "$tag done (skip)"; return 0; }
    local out="outputs/eval/${tag}.jsonl"
    log "== $tag =="
    docker exec "$DEV" bash -c "cd /workspaces/muse-glimmer-long-ctx && \
        python3 evals/harness/run_eval.py --engine vllm --base-url http://vllm:8000/v1 \
        --config-label stock $* --out $out --write-parquet" >> logs/short-${tag}.log 2>&1 \
        || log "WARN: $tag partial"
    local n; n=$(grep -c '"error": null' "$out" 2>/dev/null || echo 0)
    if [ "$n" -ge "$EXPECT" ]; then
        touch "$marker"; log "== $tag: $n valid rows =="
    else
        log "ERROR: $tag incomplete ($n/$EXPECT valid) — NOT marked"
    fi
}

EXPECT=36
run_grid ruler_le128k logs/short-ruler.done "--plugin ruler \
    --tasks ruler_vt,ruler_fwe,ruler_niah_mk,ruler_niah_mv \
    --ctx 32000,64000,128000 --depths 0.5 --reps 3 \
    --mode capability --max-tokens 4096"

EXPECT=12
run_grid mrcr_le128k logs/short-mrcr.done "--plugin mrcr --tasks mrcr2,mrcr4 \
    --ctx 32000,64000 --depths 0.5 --reps 3 \
    --mode capability --max-tokens 8192"

log "== short_le128k complete =="
echo "done $(date '+%F %T')" > logs/short-le128k.done
