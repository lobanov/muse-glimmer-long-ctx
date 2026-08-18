#!/usr/bin/env bash
# ruler_gt128k.sh — NVIDIA/RULER task recipes at 128k-512k (goal afe6584b, user ask).
# Chained behind synth-gt128k. 4 tasks x 4 ctx x 3 reps = 48 cells, capability mode.
# Partial-credit scorers (fwe/mv) give fractional scores; binary for vt/mk.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
DEV=muse-glimmer-long-ctx-dev-1
log() { echo "[$(date '+%F %T')] $*" >> logs/ruler-gt128k.log; }

# GPU gate (2026-08-18: co-running GPU work OOM-kills vLLM cold start; self-heal —
# attempt start, then poll; lanes behind this fire on ruler-gt128k.done)
COMPOSE_ALL="docker compose -f .devcontainer/docker-compose.yml -f .devcontainer/docker-compose.override.yml"
while ! docker exec "$DEV" curl -s --max-time 3 http://vllm:8000/v1/models 2>/dev/null \
        | grep -q '"root":"/arms/stock-524k"'; do
    echo "[$(date '+%F %T')] waiting for stock-524k vLLM (GPU contention with other work)" >> logs/ruler-gt128k.log
    VLLM_MODEL=/arms/stock-524k VLLM_MAX_MODEL_LEN=524288 \
        $COMPOSE_ALL --profile inference up -d vllm >/dev/null 2>&1 || true
    sleep 300
done
log "== ruler_gt128k start (pid $$); vLLM up =="

OUT=outputs/eval/ruler_gt128k.jsonl
[ -f logs/ruler-gt128k.done ] && { log "done (skip)"; exit 0; }
docker exec "$DEV" bash -c "cd /workspaces/muse-glimmer-long-ctx && \
    python3 evals/harness/run_eval.py --engine vllm --base-url http://vllm:8000/v1 \
    --config-label stock --plugin ruler \
    --tasks ruler_vt,ruler_fwe,ruler_niah_mk,ruler_niah_mv \
    --ctx 128000,256000,384000,512000 --depths 0.5 --reps 3 \
    --mode capability --max-tokens 4096 \
    --out $OUT --write-parquet" >> logs/ruler-gt128k-grid.log 2>&1 \
    || log "WARN: ruler grid partial"

N=$(grep -c '"error": null' "$OUT" 2>/dev/null || echo 0)
if [ "$N" -ge 48 ]; then
    log "== ruler complete: $N/48 rows =="
    echo "done $(date '+%F %T')" > logs/ruler-gt128k.done
else
    log "ERROR: ruler incomplete ($N/48) — resumable rerun needed"
fi
