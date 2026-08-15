#!/usr/bin/env bash
# Stage-4 queue v2 (host-side): PLAN §4 sweep, TRIMMED + REPOWERED (2026-08-15, post
# adversarial review R3/R1c — docs/review-glm53-verification.md).
#
# Old design (405 cells): 5 arms × niah,semantic,counting × 3 ctx × 3 depths × 3 reps.
# Verified problems: niah/semantic saturated at 1.000 stock → 2/3 of cells could only
# measure damage; n=3 binary CI ≈ ±54pts → winner rule detected only ≈+57pt effects;
# no ≤128k harm-check cells (GOAL criterion 6); cwe absent.
#
# New design (~83 cells, pooled decision rule):
#   qk4.3, qk5.0 (bracket; 4.1/4.6 dropped) × {counting, cwe} — the two DISCRIMINATING
#   tasks — @ {128k, 256k, 512k} × 5 instance-reps (pooled n=15/arm/task: detectable
#   ≈ ±26pts pooled; dose-response across ctx)
#   + harm-check: {counting, cwe} @ 64k × 5 reps per arm (stock data exists)
#   + yarn4 as a 3-cell probe (niah@128k × 3) — PLAN itself expects near-inert
# Decision read: pooled per-task arm-vs-stock delta + per-cell wins (stage6-v2 prints
# winner-info for the approval decision; no blind per-cell rule).
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
DEV=muse-glimmer-long-ctx-dev-1
COMPOSE="docker compose -f .devcontainer/docker-compose.yml"
log() { echo "[$(date '+%F %T')] $*" >> logs/stage4-queue.log; }

log "stage4-v2 armed (pid $$); waiting for stage3"
while [ ! -f logs/stage3-queue.done ]; do sleep 300; done

for ARM in qk4.3 qk5.0; do
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
    [ "$UP" != "1" ] && { log "ERROR: vLLM never came up for $ARM — skipping"; continue; }
    log "== arm $ARM: discriminating grids (counting,cwe @128k-512k, 5 reps) =="
    docker exec "$DEV" bash -c "cd /workspaces/muse-glimmer-long-ctx && \
        python3 evals/harness/run_eval.py --engine vllm --base-url http://vllm:8000/v1 \
        --config-label $ARM --tasks counting,cwe \
        --ctx 128000,256000,512000 --depths 0.5 --reps 5 \
        --mode capability --max-tokens 4096 \
        --out outputs/eval/arm_${ARM}.jsonl --write-parquet" \
        >> logs/stage4-$ARM-grid.log 2>&1 \
      || log "WARN: grid for $ARM exited nonzero (partial data kept)"
    log "== arm $ARM: harm-check @64k =="
    docker exec "$DEV" bash -c "cd /workspaces/muse-glimmer-long-ctx && \
        python3 evals/harness/run_eval.py --engine vllm --base-url http://vllm:8000/v1 \
        --config-label $ARM --tasks counting,cwe \
        --ctx 64000 --depths 0.5 --reps 5 \
        --mode capability --max-tokens 4096 \
        --out outputs/eval/arm_${ARM}.jsonl" \
        >> logs/stage4-$ARM-grid.log 2>&1 \
      || log "WARN: harm-check for $ARM partial"
    N=$(wc -l < outputs/eval/arm_${ARM}.jsonl 2>/dev/null || echo 0)
    log "== arm $ARM complete: $N rows =="
    touch "logs/stage4-$ARM.done"
done

# yarn4: probe only (PLAN §4b expects near-inert; 3 cells)
if [ ! -f logs/stage4-yarn4.done ]; then
    log "== yarn4 probe =="
    $COMPOSE --profile inference stop vllm >/dev/null 2>&1; sleep 10
    VLLM_MODEL=/arms/yarn4 VLLM_MAX_MODEL_LEN=524288 \
        $COMPOSE --profile inference up -d vllm >/dev/null 2>&1
    for _ in $(seq 1 80); do
        docker exec "$DEV" curl -s --max-time 3 http://vllm:8000/v1/models >/dev/null 2>&1 && break
        sleep 30
    done
    docker exec "$DEV" bash -c "cd /workspaces/muse-glimmer-long-ctx && \
        python3 evals/harness/run_eval.py --engine vllm --base-url http://vllm:8000/v1 \
        --config-label yarn4 --tasks niah,counting --ctx 128000 --depths 0.5 --reps 3 \
        --mode capability --max-tokens 4096 \
        --out outputs/eval/arm_yarn4.jsonl" >> logs/stage4-yarn4-grid.log 2>&1 \
      || log "WARN: yarn4 probe partial"
    touch logs/stage4-yarn4.done
fi

# leave vLLM serving stock for the stage5+ chain
$COMPOSE --profile inference stop vllm >/dev/null 2>&1; sleep 10
VLLM_MODEL=/arms/stock-524k VLLM_MAX_MODEL_LEN=524288 \
    $COMPOSE --profile inference up -d vllm >/dev/null 2>&1
echo "done $(date '+%F %T')" > logs/stage4-queue.done
log "stage4-v2 complete: $(cat logs/stage4-queue.done)"
