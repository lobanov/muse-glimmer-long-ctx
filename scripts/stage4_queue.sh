#!/usr/bin/env bash
# Stage-4 GPU queue (host-side): PLAN §4 zero-shot arm sweep.
# Waits for stage3 (logs/stage3-queue.done), then for each arm in priority order
# (qk4.3, qk5.0, qk4.1, qk4.6, yarn4-control):
#   - restart vLLM with VLLM_MODEL=/arms/<arm> VLLM_MAX_MODEL_LEN=524288
#   - quick decision grid: {niah, semantic} × {128k, 256k, 512k} × depths {0, 0.5, 1.0}
#     × 3 reps  (128k = knob-harm check vs stock grid; >128k = extrapolation benefit)
#   - config_label = arm name (compare.py diffs against stock automatically)
# Marker: logs/stage4-queue.done (+ per-arm logs/stage4-<arm>.done)
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
DEV=muse-glimmer-long-ctx-dev-1
COMPOSE="docker compose -f .devcontainer/docker-compose.yml"
ARMS="qk4.3 qk5.0 qk4.1 qk4.6 yarn4"
log() { echo "[$(date '+%F %T')] $*" >> logs/stage4-queue.log; }

log "stage4 armed (pid $$); waiting for stage3"
while [ ! -f logs/stage3-queue.done ]; do sleep 300; done

for ARM in $ARMS; do
    if [ -f "logs/stage4-$ARM.done" ]; then log "$ARM already done"; continue; fi
    log "== arm $ARM: restart vLLM =="
    $COMPOSE --profile inference stop vllm >/dev/null 2>&1
    sleep 10
    VLLM_MODEL=/arms/$ARM VLLM_MAX_MODEL_LEN=524288 \
        $COMPOSE --profile inference up -d vllm >/dev/null 2>&1
    UP=0
    for _ in $(seq 1 80); do
        if docker exec "$DEV" curl -s --max-time 3 http://vllm:8000/v1/models >/dev/null 2>&1; then UP=1; break; fi
        sleep 30
    done
    if [ "$UP" != "1" ]; then log "ERROR: vLLM never came up for $ARM — skipping"; continue; fi
    log "== arm $ARM: grid =="
    docker exec "$DEV" bash -c "cd /workspaces/muse-glimmer-long-ctx && \
        python3 evals/harness/run_eval.py --engine vllm --base-url http://vllm:8000/v1 \
        --config-label $ARM --tasks niah,semantic \
        --ctx 128000,256000,512000 --depths 0.0,0.5,1.0 --reps 3 \
        --mode capability --max-tokens 4096 \
        --out outputs/eval/arm_${ARM}.jsonl --write-parquet" \
        >> logs/stage4-$ARM-grid.log 2>&1 \
      || log "WARN: grid for $ARM exited nonzero (partial data kept)"
    N=$(wc -l < outputs/eval/arm_${ARM}.jsonl 2>/dev/null || echo 0)
    log "== arm $ARM complete: $N rows =="
    touch "logs/stage4-$ARM.done"
done

echo "done $(date '+%F %T')" > logs/stage4-queue.done
log "stage4 complete: $(cat logs/stage4-queue.done)"
