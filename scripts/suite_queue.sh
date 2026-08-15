#!/usr/bin/env bash
# Second-stage GPU queue (host-side). Waits for the overnight chain to finish
# (logs/overnight-queue.done), then runs — on the already-serving vLLM @524288
# (stock-524k arm) — in decision-value order, all resumable:
#   A. NoLiMa      : 32k,64k,128k,256k,384k,512k x {0,0.5,1.0} x 3 reps   (semantic axis)
#   B. LongBench v2: 32k,128k,256k,512k x 3 instance-resamples            (realistic reasoning)
#   C. LongCodeQA  : 32k,64k,128k,256k,512k x 3 resamples                 (repo-scale coding)
#   D. InfiniteBench: infb_kv,infb_bookmc,infb_codedebug @ 128k,256k x 3   (broad >100k)
#   E. synthetic fill-in: conflicts,set_intersect,chronology @ 32k..128k x full depths x 3
# Then converts each to Parquet and writes logs/suite-queue.done.
#
# Run detached: nohup bash scripts/suite_queue.sh &      Log: logs/suite-queue.log
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
DEV=muse-glimmer-long-ctx-dev-1
OUT=outputs/eval
log() { echo "[$(date '+%F %T')] $*"; }

log "suite queue armed (pid $$); waiting for overnight chain"
while [ ! -f logs/overnight-queue.done ]; do sleep 300; done
log "overnight chain done: $(cat logs/overnight-queue.done)"

# do not race the concurrent suite LANE (same output files/cell ids): wait until it
# finishes (marker) or its process is gone; the queue then no-op-skips lane-completed
# cells via resume and runs only what remains.
while [ ! -f logs/suite-lane.done ] && pgrep -f "scripts/suite_lane" >/dev/null; do
    log "suite lane still running — waiting"
    sleep 300
done
log "suite lane settled; proceeding (resume skips its cells)"

# vLLM @524288 must be up (queue step 5 leaves it serving)
UP=0
for _ in $(seq 1 60); do
    if docker exec "$DEV" curl -s --max-time 3 http://vllm:8000/v1/models >/dev/null 2>&1; then UP=1; break; fi
    sleep 30
done
[ "$UP" = "1" ] || { log "ERROR: vLLM not serving after overnight chain; aborting"; exit 1; }
log "vLLM up; starting suite grids"

run() {  # run <name> <args...>
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

echo "done $(date '+%F %T')" > logs/suite-queue.done
log "suite queue complete: $(cat logs/suite-queue.done)"
