#!/usr/bin/env bash
# Third-stage GPU queue (host-side). Waits for the suite queue (logs/suite-queue.done),
# then on the still-serving vLLM @524288 runs:
#   A. agentmem grid: 32k..512k x {0, 0.1, 0.5, 0.9} x 3 reps   (custom agentic memory)
#   B. PPL probe: stock-524k at 32k/131k/262k/393k/524k x 2 reps (PLAN §3 perplexity curve)
# Marker: logs/stage3-queue.done
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
DEV=muse-glimmer-long-ctx-dev-1
log() { echo "[$(date '+%F %T')] $*"; }

log "stage3 queue armed (pid $$); waiting for suite queue"
while [ ! -f logs/suite-queue.done ]; do sleep 300; done
log "suite queue done; checking vLLM"
UP=0
for _ in $(seq 1 30); do
    if docker exec "$DEV" curl -s --max-time 3 http://vllm:8000/v1/models >/dev/null 2>&1; then UP=1; break; fi
    sleep 30
done
[ "$UP" = "1" ] || { log "ERROR: vLLM down; aborting (run manually later)"; exit 1; }

log "A. agentmem grid"
docker exec "$DEV" bash -c "cd /workspaces/muse-glimmer-long-ctx && \
    python3 evals/harness/run_eval.py --engine vllm --base-url http://vllm:8000/v1 \
    --config-label stock --plugin agentmem --tasks agentmem \
    --ctx 32000,64000,128000,256000,384000,512000 --depths 0.0,0.1,0.5,0.9 --reps 3 \
    --mode capability --max-tokens 1024 \
    --out outputs/eval/suite_agentmem.jsonl --write-parquet" \
    >> logs/suite-grid-suite_agentmem.log 2>&1 \
  || log "WARN: agentmem grid exited nonzero (partial data kept)"

log "B. PPL probe (stock-524k)"
docker exec "$DEV" bash -c "cd /workspaces/muse-glimmer-long-ctx && \
    python3 evals/ppl_probe.py --base-url http://vllm:8000/v1 --config-label stock-524k \
    --ctx 32000,131072,262144,393216,524288 --reps 2 \
    --out outputs/eval/ppl_stock.jsonl" \
    >> logs/ppl-probe.log 2>&1 \
  || log "WARN: ppl probe exited nonzero (partial data kept)"

echo "done $(date '+%F %T')" > logs/stage3-queue.done
log "stage3 complete: $(cat logs/stage3-queue.done)"
