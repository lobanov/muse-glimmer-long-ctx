#!/usr/bin/env bash
# synth_gt128k.sh — synthetic weak-axis reads at true 384k/512k (goal afe6584b).
# Chain: runs after infb_bands. Sampled n=3 (counting,cwe d0.5) + greedy n=3 on
# counting@512k (stochastic-share quantification at the extreme).
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
DEV=muse-glimmer-long-ctx-dev-1
log() { echo "[$(date '+%F %T')] $*" >> logs/synth-gt128k.log; }

while [ ! -f logs/infbands-queue.done ]; do sleep 300; done
log "== synth_gt128k start (pid $$) =="

run() {  # run <tag> <args...>
    local tag=$1; shift
    [ -f "logs/infbands/${tag}.done" ] && { log "$tag done (skip)"; return 0; }
    local out="outputs/eval/${tag}.jsonl"
    log "== $tag =="
    docker exec "$DEV" bash -c "cd /workspaces/muse-glimmer-long-ctx && \
        python3 evals/harness/run_eval.py --engine vllm --base-url http://vllm:8000/v1 \
        --config-label stock $* --out $out" >> logs/infbands/${tag}.log 2>&1 \
        || log "WARN: $tag partial"
    local n; n=$(wc -l < "$out" 2>/dev/null || echo 0)
    [ "$n" -ge 3 ] && touch "logs/infbands/${tag}.done" && log "== $tag: $n rows ==" \
        || log "ERROR: $tag incomplete ($n/3)"
}

run synth_384k "--tasks counting,cwe --ctx 384000 --depths 0.5 --reps 3 \
    --mode capability --max-tokens 4096"
run synth_512k "--tasks counting,cwe --ctx 512000 --depths 0.5 --reps 3 \
    --mode capability --max-tokens 4096"
run synth_512k_greedy_counting "--tasks counting --ctx 512000 --depths 0.5 --reps 3 \
    --mode parity --max-tokens 4096"

log "== synth_gt128k complete =="
echo "done $(date '+%F %T')" > logs/synth-gt128k.done
