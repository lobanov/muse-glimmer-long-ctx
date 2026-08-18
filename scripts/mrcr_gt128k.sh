#!/usr/bin/env bash
# mrcr_gt128k.sh — MRCR v2 official-data lanes (goal d56ed95d).
# mrcr2/mrcr4 @131072 (true ~131k) and @262144 (true ~262k), n=3 each = 12 cells.
# 524288-pointwise rows exist but prompt(~524.1k)+generation would overflow the
# 524288 window — deliberately excluded (noted in STATUS).
# Chained after ruler_gt128k for GPU serialization.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
DEV=muse-glimmer-long-ctx-dev-1
log() { echo "[$(date '+%F %T')] $*" >> logs/mrcr-gt128k.log; }

while ! [ -f logs/ruler-gt128k.done ] && pgrep -f 'bash scripts/ruler_gt128k.sh' >/dev/null; do
    sleep 300
done
log "== mrcr_gt128k start (pid $$); ruler done=$([ -f logs/ruler-gt128k.done ] && echo yes || echo NO) =="

[ -f logs/mrcr-gt128k.done ] && { log "done (skip)"; exit 0; }
OUT=outputs/eval/mrcr_gt128k.jsonl
docker exec "$DEV" bash -c "cd /workspaces/muse-glimmer-long-ctx && \
    python3 evals/harness/run_eval.py --engine vllm --base-url http://vllm:8000/v1 \
    --config-label stock --plugin mrcr --tasks mrcr2,mrcr4 \
    --ctx 131072,262144 --depths 0.5 --reps 3 \
    --mode capability --max-tokens 8192 \
    --out $OUT --write-parquet" >> logs/mrcr-gt128k-grid.log 2>&1 \
    || log "WARN: mrcr grid partial"

N=$(wc -l < "$OUT" 2>/dev/null || echo 0)
if [ "$N" -ge 12 ]; then
    log "== mrcr complete: $N/12 rows =="
    echo "done $(date '+%F %T')" > logs/mrcr-gt128k.done
else
    log "ERROR: mrcr incomplete ($N/12) — resumable rerun needed"
fi
