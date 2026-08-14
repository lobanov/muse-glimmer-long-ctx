#!/usr/bin/env bash
# Stage-5 GPU queue (host-side): final pre-flight after the §4 arm sweep.
#   1. wait for stage4 (logs/stage4-queue.done)
#   2. recreate the dev container on the rebuilt image (gguf dep; GPU queues are done,
#      so killing dev processes is safe now) → verify-env must pass (AGENTS rule 4)
#   3. stop vLLM — free the GPU for §7 training
#   4. run the §7 trainer --dry-run (one fwd/bwd @512, trainable-param table)
#   5. leave the GPU FREE (training-ready state for the next session)
# Marker: logs/stage5-queue.done (contains dry-run result line)
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
DEV=muse-glimmer-long-ctx-dev-1
COMPOSE="docker compose -f .devcontainer/docker-compose.yml"
log() { echo "[$(date '+%F %T')] $*" >> logs/stage5-queue.log; }

log "stage5 armed (pid $$); waiting for stage4"
while [ ! -f logs/stage4-queue.done ]; do sleep 300; done
log "stage4 done; refreshing dev container (new image: gguf dep)"

$COMPOSE stop dev >/dev/null 2>&1
$COMPOSE rm -f dev >/dev/null 2>&1
$COMPOSE up -d dev >/dev/null 2>&1
sleep 20
if docker exec "$DEV" bash scripts/verify-env.sh 2>&1 | tail -1 | grep -q "ALL CHECKS PASSED"; then
    log "dev recreated; verify-env PASSED"
else
    log "ERROR: verify-env failed after recreate — inspect before anything else"; exit 1
fi

log "stopping vLLM — GPU free for §7"
$COMPOSE --profile inference stop vllm >/dev/null 2>&1
sleep 15

log "running §7 trainer --dry-run (qlora, r32, all-scope)"
DRY=$(docker exec "$DEV" bash -c "cd /workspaces/muse-glimmer-long-ctx && \
    python3 src/muse_longctx/train_qlora.py --mode qlora --lora-rank 32 --lora-scope all \
    --dry-run 2>&1 | tail -25")
echo "$DRY" | tail -8 >> logs/stage5-queue.log
if echo "$DRY" | grep -q "forward+backward OK"; then
    echo "dry-run OK $(date '+%F %T')" > logs/stage5-queue.done
    log "stage5 complete — GPU left FREE, training-ready (corpus: outputs/corpus/train_v1)"
else
    echo "dry-run FAILED $(date '+%F %T')" > logs/stage5-queue.done
    log "ERROR: dry-run failed — full output above; fix before training"
fi
