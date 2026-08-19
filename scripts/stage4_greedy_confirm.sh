#!/usr/bin/env bash
# stage4_greedy_confirm.sh — close the greedy-confirm flags from docs/phase4-evidence.md.
# Runs the weak-axis cells (counting,cwe @128k+256k d0.5 x5) in PARITY (greedy) mode on
# stock + both qk arms. Same cell_seeds as the sampled runs (mode is not in the seed)
# -> paired greedy-vs-greedy reads decide whether the flagged sampled deltas are real.
# GPU discipline: run ONLY while stage6 watcher is paused (approval gate) so train1
# cannot launch mid-lane. Leaves stock serving at the end.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
DEV=muse-glimmer-long-ctx-dev-1
COMPOSE="docker compose -f .devcontainer/docker-compose.yml"
log() { echo "[$(date '+%F %T')] $*" >> logs/stage4-greedy-confirm.log; }

serve_root() {  # serve_root <path> — poll until /v1/models root matches
    $COMPOSE --profile inference stop vllm >/dev/null 2>&1; sleep 10
    VLLM_MODEL=$1 VLLM_MAX_MODEL_LEN=524288 \
        $COMPOSE --profile inference up -d vllm >/dev/null 2>&1
    for _ in $(seq 1 80); do
        docker exec "$DEV" curl -s --max-time 3 http://vllm:8000/v1/models 2>/dev/null \
            | grep -q "\"root\":\"$1\"" && return 0
        sleep 30
    done
    return 1
}

greedy_grid() {  # greedy_grid <label> <outfile>
    docker exec "$DEV" bash -c "cd /workspaces/muse-glimmer-long-ctx && \
        python3 evals/harness/run_eval.py --engine vllm --base-url http://vllm:8000/v1 \
        --config-label $1 --tasks counting,cwe \
        --ctx 128000,256000 --depths 0.5 --reps 5 \
        --mode parity --max-tokens 4096 \
        --out outputs/eval/$2" >> logs/stage4-greedy-confirm-grid.log 2>&1
    local n; n=$(wc -l < outputs/eval/$2 2>/dev/null || echo 0)
    log "greedy grid $1: $n/20 rows"
    [ "$n" -ge 20 ]
}

log "== greedy-confirm lane start (pid $$) =="
for LABEL in stock qk4.3 qk5.0; do
    case $LABEL in
        stock) P=/arms/stock-524k; F=confirm_greedy_stock.jsonl;;
        *)     P=/arms/$LABEL;     F=confirm_greedy_${LABEL}.jsonl;;
    esac
    [ -f "logs/greedy-confirm-$LABEL.done" ] && { log "$LABEL done (skip)"; continue; }
    log "serving $P"
    serve_root "$P" || { log "ERROR: $P never served"; exit 1; }
    log "greedy grid $LABEL -> $F"
    greedy_grid "$LABEL" "$F" \
        && touch "logs/greedy-confirm-$LABEL.done" \
        || { log "ERROR: $LABEL greedy grid incomplete"; exit 1; }
done
serve_root /arms/stock-524k || log "WARN: could not restore stock"
log "== greedy-confirm lane complete =="
echo "done $(date '+%F %T')" > logs/greedy-confirm.done
