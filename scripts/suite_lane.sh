#!/usr/bin/env bash
# Suite LANE (host-side): runs the community-suite grids SEQUENTIALLY as a second vLLM
# client, CONCURRENT with the overnight >128k grid (same stock-524k instance; scores
# unaffected, cell_ids identical so the later suite_queue skips them via resume).
# Latency caveat: ttft/wall for concurrent cells reflect shared serving — realistic for
# agentic deployment, but not clean single-stream numbers (noted in STATUS/snapshot).
# PPL probe deliberately NOT in this lane (echo requests are too heavy to share).
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
DEV=muse-glimmer-long-ctx-dev-1
OUT=outputs/eval
log() { echo "[$(date '+%F %T')] $*" >> logs/suite-lane.log; }

log "suite lane starting (concurrent with overnight grid)"

run() {
    local name="$1"; shift
    log "grid $name starting"
    docker exec "$DEV" bash -c "cd /workspaces/muse-glimmer-long-ctx && \
        python3 evals/harness/run_eval.py --engine vllm --base-url http://vllm:8000/v1 \
        --config-label stock --out $OUT/${name}.jsonl $*" \
        >> logs/suite-grid-${name}.log 2>&1 \
      || log "WARN: grid $name exited nonzero (partial data kept)"
    docker exec "$DEV" bash -c "cd /workspaces/muse-glimmer-long-ctx && \
        python3 evals/harness/to_parquet.py $OUT/${name}.jsonl -o $OUT/${name}.parquet" \
        >> logs/suite-grid-${name}.log 2>&1 || true
    log "grid $name rows: $([ -f $OUT/${name}.jsonl ] && wc -l < $OUT/${name}.jsonl)"
}

run suite_nolima "--plugin nolima --tasks nolima \
    --ctx 32000,64000,128000,256000,384000,512000 --depths 0.0,0.5,1.0 --reps 3 \
    --mode capability --max-tokens 1024"

run suite_longbench_v2 "--plugin longbench_v2 --tasks longbench_v2 \
    --ctx 32000,128000,256000,512000 --depths 0.5 --reps 3 \
    --mode capability --max-tokens 1024"

run suite_longcodeqa "--plugin longcodeqa --tasks longcodeqa \
    --ctx 32000,64000,128000,256000,512000 --depths 0.5 --reps 3 \
    --mode capability --max-tokens 1024"

run suite_infbench "--plugin infbench --tasks infb_kv,infb_bookmc,infb_codedebug \
    --ctx 128000,256000 --depths 0.5 --reps 3 \
    --mode capability --max-tokens 1024"

run suite_synth3 "--tasks conflicts,set_intersect,chronology \
    --ctx 32000,64000,128000 --depths 0.0,0.1,0.25,0.5,0.75,0.9,1.0 --reps 3 \
    --mode capability --max-tokens 4096"

echo "done $(date '+%F %T')" > logs/suite-lane.done
log "suite lane complete (suite_queue will no-op-skip these cells later)"
