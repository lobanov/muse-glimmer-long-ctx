#!/usr/bin/env bash
# Overnight GPU queue (host-side). Chains, unattended:
#   1. wait for the §3 stock ≤128k vLLM grid to finish (re-kick once if it died)
#   2. JSONL -> Parquet for the grid
#   3. stop vLLM  ->  start llama-server (K-Quant GGUF, -c 163840, iSWA default)
#   4. §0 parity-caveat re-run on the quant artifact (greedy + reasoning_strength low
#      + max_tokens 4096; closes the docs/phase0 action item on llama.cpp side)
#   5. remove llama-server  ->  relaunch vLLM with VLLM_MAX_MODEL_LEN=524288
#   6. §3 stock >128k grid (192k/256k/384k/512k × depths {0,0.5,1.0} × 3 reps × 6 tasks)
#   7. wait, convert to Parquet, write marker
#
# Run detached from the repo root:  nohup bash scripts/overnight_queue.sh &
# Log: logs/overnight-queue.log
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
DEV=muse-glimmer-long-ctx-dev-1
LE_JSONL=outputs/eval/stock_vllm_le128k.jsonl
LE_PARQUET=outputs/eval/stock_vllm_le128k.parquet
GT_JSONL=outputs/eval/stock_vllm_gt128k.jsonl
GT_PARQUET=outputs/eval/stock_vllm_gt128k.parquet
CAVEAT_JSONL=outputs/eval/parity_caveat_llamacpp.jsonl
COMPOSE="docker compose -f .devcontainer/docker-compose.yml"
GGUF=/cache/weights/Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf
MMPROJ=/cache/weights/mmproj-Muse-Glimmer-30B-Q4_K_M.gguf

log() { echo "[$(date '+%F %T')] $*"; }

rows() { [ -f "$1" ] && wc -l < "$1" | tr -d ' ' || echo 0; }

grid_running() {
    docker exec "$DEV" pgrep -f 'run_eval.py.*stock_vllm_le128k' >/dev/null 2>&1
}

log "queue started (pid $$)"

# ---- 1+2: wait for the ≤128k grid (one re-kick if it died early) -----------------
reKicks=0
while :; do
    N=$(rows "$LE_JSONL")
    if [ "$N" -ge 378 ]; then break; fi
    if ! grid_running; then
        if [ "$reKicks" -lt 1 ]; then
            reKicks=1
            log "grid not running with only $N/378 rows -> re-kicking (resume)"
            docker exec -d "$DEV" bash -c \
                'cd /workspaces/muse-glimmer-long-ctx && python3 evals/harness/run_eval.py \
                 --engine vllm --base-url http://vllm:8000/v1 --config-label stock \
                 --tasks niah,niah_multi,multihop,counting,semantic,abstain \
                 --ctx 32000,64000,128000 --depths 0.0,0.1,0.25,0.5,0.75,0.9,1.0 \
                 --reps 3 --mode capability --max-tokens 4096 \
                 --out outputs/eval/stock_vllm_le128k.jsonl --write-parquet \
                 >> /workspaces/muse-glimmer-long-ctx/logs/eval-stock-le128k.log 2>&1'
            sleep 60
        else
            log "ERROR: grid dead at $N/378 rows after re-kick; aborting chain (inspect manually)"
            exit 1
        fi
    fi
    sleep 120
done
log "≤128k grid complete: $(rows "$LE_JSONL") rows"
docker exec "$DEV" bash -c \
    "cd /workspaces/muse-glimmer-long-ctx && python3 evals/harness/to_parquet.py \
     outputs/eval/stock_vllm_le128k.jsonl -o outputs/eval/stock_vllm_le128k.parquet" \
    || log "WARN: parquet conversion failed (retry later; jsonl is intact)"

# ---- 3: vLLM down, llama-server up ----------------------------------------------
log "stopping vLLM sidecar"
$COMPOSE --profile inference stop vllm >/dev/null 2>&1
sleep 10
log "starting llama-server (K-Quant, -c 163840)"
$COMPOSE --profile llamacpp run -d llamacpp \
    llama-server -m "$GGUF" --mmproj "$MMPROJ" -ngl 99 -c 163840 \
    --host 0.0.0.0 --port 8080 --jinja --temp 1.0 --top-p 0.95 --top-k 64 \
    >/dev/null 2>&1
LLNAME=$(docker ps --filter label=com.docker.compose.service=llamacpp --format '{{.Names}}' | head -1)
if [ -z "$LLNAME" ]; then log "ERROR: llama-server container not found; aborting"; exit 1; fi
log "llama-server container: $LLNAME (waiting for /v1/models)"
UP=0
for _ in $(seq 1 90); do
    if docker exec "$DEV" curl -s --max-time 3 "http://$LLNAME:8080/v1/models" >/dev/null 2>&1; then UP=1; break; fi
    sleep 10
done
if [ "$UP" -ne 1 ]; then log "ERROR: llama-server never came up; aborting"; exit 1; fi
log "llama-server up"

# ---- 4: §0 caveat re-run on the quant artifact ----------------------------------
log "running parity-caveat re-run (llama.cpp, 128k @ 90%, greedy, 3 reps)"
docker exec "$DEV" bash -c \
    "cd /workspaces/muse-glimmer-long-ctx && python3 evals/harness/run_eval.py \
     --engine llamacpp --base-url http://$LLNAME:8080/v1 --config-label kquant \
     --tasks niah --ctx 128000 --depths 0.9 --reps 3 --mode parity --max-tokens 4096 \
     --out outputs/eval/parity_caveat_llamacpp.jsonl" \
    || log "WARN: caveat re-run had errors (see $CAVEAT_JSONL)"
log "caveat re-run rows: $(rows "$CAVEAT_JSONL") (3 = clean close of the phase-0 action)"

# ---- 5: llama-server down, vLLM up @ 524288 -------------------------------------
docker rm -f "$LLNAME" >/dev/null 2>&1
sleep 5
log "restarting vLLM with VLLM_MAX_MODEL_LEN=524288 (~10 min cold start)"
VLLM_MAX_MODEL_LEN=524288 $COMPOSE --profile inference up -d vllm >/dev/null 2>&1
UP=0
for _ in $(seq 1 80); do
    if docker exec "$DEV" curl -s --max-time 3 http://vllm:8000/v1/models >/dev/null 2>&1; then UP=1; break; fi
    sleep 30
done
if [ "$UP" -ne 1 ]; then log "ERROR: vLLM @524288 never came up; aborting"; exit 1; fi
log "vLLM up at 524288 max len"

# ---- 6: >128k stock grid ---------------------------------------------------------
log "launching >128k stock grid (216 cells, task-major order: niah first)"
docker exec -d "$DEV" bash -c \
    'cd /workspaces/muse-glimmer-long-ctx && python3 evals/harness/run_eval.py \
     --engine vllm --base-url http://vllm:8000/v1 --config-label stock \
     --tasks niah,semantic,multihop,abstain,counting,niah_multi \
     --ctx 192000,256000,384000,512000 --depths 0.0,0.5,1.0 \
     --reps 3 --mode capability --max-tokens 4096 \
     --out outputs/eval/stock_vllm_gt128k.jsonl --write-parquet \
     > /workspaces/muse-glimmer-long-ctx/logs/eval-stock-gt128k.log 2>&1'

# ---- 7: wait, convert, marker ----------------------------------------------------
GT_REPS=0
while :; do
    N=$(rows "$GT_JSONL")
    if [ "$N" -ge 216 ]; then break; fi
    if ! docker exec "$DEV" pgrep -f 'run_eval.py.*stock_vllm_gt128k' >/dev/null 2>&1; then
        if [ "$GT_REPS" -lt 1 ]; then
            GT_REPS=1
            log "gt128k grid died at $N/216 -> re-kicking once (resume)"
            docker exec -d "$DEV" bash -c \
                'cd /workspaces/muse-glimmer-long-ctx && python3 evals/harness/run_eval.py \
                 --engine vllm --base-url http://vllm:8000/v1 --config-label stock \
                 --tasks niah,semantic,multihop,abstain,counting,niah_multi \
                 --ctx 192000,256000,384000,512000 --depths 0.0,0.5,1.0 \
                 --reps 3 --mode capability --max-tokens 4096 \
                 --out outputs/eval/stock_vllm_gt128k.jsonl --write-parquet \
                 >> /workspaces/muse-glimmer-long-ctx/logs/eval-stock-gt128k.log 2>&1'
            sleep 60
        else
            log "ERROR: gt128k grid dead at $N/216 after re-kick; partial data kept"
            break
        fi
    fi
    sleep 300
done
docker exec "$DEV" bash -c \
    "cd /workspaces/muse-glimmer-long-ctx && python3 evals/harness/to_parquet.py \
     outputs/eval/stock_vllm_gt128k.jsonl -o outputs/eval/stock_vllm_gt128k.parquet" \
    || log "WARN: gt128k parquet conversion failed (jsonl intact)"
echo "done $(date '+%F %T'): le128k=$(rows "$LE_JSONL") caveat=$(rows "$CAVEAT_JSONL") gt128k=$(rows "$GT_JSONL")" \
    > logs/overnight-queue.done
log "queue complete: $(cat logs/overnight-queue.done)"
