#!/usr/bin/env bash
# Stage-7 queue (host-side): §8 first trained-model evaluation.
#   G1: logs/train1.launched exists
#   G2: training process finished AND adapter saved (outputs/adapters/run1/adapter_config.json)
#   G3: GPU free (no vLLM serving)
# Then:
#   1. merge adapter → outputs/merged/run1 (export_pipeline stage 1; idempotent)
#   2. serve merged on vLLM @524288 (VLLM_MODEL=/outputs/merged/run1)
#   3. §8 grid — weak axes + regression check (stage4-v2.1-aligned):
#        counting, cwe, multihop, abstain, niah @ 128k/256k/512k × {0,0.5,1.0} × 3 reps
#        config_label=run1  (compare.py diffs vs stock + arms automatically)
#      plus ≤32k short-regression cells (niah, semantic, counting, CWE @ 32k × 3) —
#      cwe@32k added 2026-08-15 audit: sharpest short-context discriminator (0.778)
#   4. PPL probe on merged (32k..524k, 2 reps)
# Marker: logs/stage7-queue.done
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
DEV=muse-glimmer-long-ctx-dev-1
COMPOSE="docker compose -f .devcontainer/docker-compose.yml"
log() { echo "[$(date '+%F %T')] $*" >> logs/stage7-queue.log; }
source "$ROOT/scripts/progress_lib.sh"

log "stage7 armed (pid $$)"
progress_waiting "waiting for run1 adapter (stage6)"
while :; do
    [ -f logs/train1.launched ] && [ -f outputs/adapters/run1/adapter_config.json ] \
        && ! docker exec "$DEV" pgrep -f 'train_qlora.py' >/dev/null 2>&1 && break
    progress_waiting "waiting for run1 adapter"; sleep 300
done
progress_step 1 6 "adapter saved"
log "G1+G2 ok: adapter saved, trainer exited"

# GPU must be free (stage5/6 chain leaves vLLM stopped after dry-run; verify anyway)
$COMPOSE --profile inference stop vllm >/dev/null 2>&1; sleep 10

log "merging adapter (export stage 1)"
progress_step 2 6 "merging adapter"
bash scripts/export_pipeline.sh outputs/adapters/run1 run1 --stage 1 2>&1 | tail -2 >> logs/stage7-queue.log
[ -f outputs/merged/run1/config.json ] || { log "ERROR: merge failed"; exit 1; }
# mechanical window extension on the MERGED config (same semantics as outputs/arms/stock-524k,
# see docs/phase4-zeroshot-arms.md): training leaves max_position_embeddings=131072, but
# §8 evaluates at 256k/512k and vLLM caps max_model_len at the config value — without
# this patch the 524288 serve fails at cold start. Also flows into the §11 GGUF
# (context_length=524288) consistent with the 512k deployment artifact.
docker exec -i "$DEV" python3 - <<'PY' >> logs/stage7-queue.log 2>&1
import json
p = "/workspaces/muse-glimmer-long-ctx/outputs/merged/run1/config.json"
c = json.load(open(p))
old = c["text_config"]["max_position_embeddings"]
c["text_config"]["max_position_embeddings"] = 524288
json.dump(c, open(p, "w"), indent=2)
print(f"merged config: max_position_embeddings {old} -> 524288 (mechanical; eval+artifact)")
PY
docker exec "$DEV" python3 -c "
from transformers import AutoProcessor
p = AutoProcessor.from_pretrained('/workspaces/muse-glimmer-long-ctx/outputs/merged/run1')
print('processor ok:', type(p).__name__)" >> logs/stage7-queue.log 2>&1 \
  || { log "ERROR: merged dir missing processor files"; exit 1; }

log "serving merged @524288"
progress_step 3 6 "serving merged @524288"
VLLM_MODEL=/outputs/merged/run1 VLLM_MAX_MODEL_LEN=524288 \
    $COMPOSE --profile inference up -d vllm >/dev/null 2>&1
UP=0
for _ in $(seq 1 80); do
    if docker exec "$DEV" curl -s --max-time 3 http://vllm:8000/v1/models >/dev/null 2>&1; then UP=1; break; fi
    sleep 30
done
[ "$UP" = "1" ] || { log "ERROR: merged model never served"; exit 1; }
log "merged serving"

log "§8 grid: run1 decision subset + ≤32k regression"
progress_step 4 6 "§8 grids"
docker exec "$DEV" bash -c "cd /workspaces/muse-glimmer-long-ctx && \
    python3 evals/harness/run_eval.py --engine vllm --base-url http://vllm:8000/v1 \
    --config-label run1 --tasks niah,semantic,multihop,abstain,counting,cwe \
    --ctx 128000,256000,512000 --depths 0.0,0.5,1.0 --reps 3 \
    --mode capability --max-tokens 4096 \
    --out outputs/eval/run1_vllm.jsonl --write-parquet" \
    >> logs/stage7-grid.log 2>&1 || log "WARN: run1 grid partial (data kept)"
docker exec "$DEV" bash -c "cd /workspaces/muse-glimmer-long-ctx && \
    python3 evals/harness/run_eval.py --engine vllm --base-url http://vllm:8000/v1 \
    --config-label run1 --tasks niah,semantic,multihop,abstain,counting,cwe \
    --ctx 32000 --depths 0.0,0.5,1.0 --reps 3 \
    --mode capability --max-tokens 4096 \
    --out outputs/eval/run1_vllm_short.jsonl" \
    >> logs/stage7-grid.log 2>&1 || log "WARN: run1 short-regression partial"

log "PPL probe on merged"
progress_step 5 6 "PPL probe"
docker exec "$DEV" bash -c "cd /workspaces/muse-glimmer-long-ctx && \
    python3 evals/ppl_probe.py --base-url http://vllm:8000/v1 --config-label run1-merged \
    --ctx 32000,131072,262144,393216,524288 --reps 2 \
    --out outputs/eval/ppl_run1.jsonl" >> logs/ppl-probe.log 2>&1 || log "WARN: ppl partial"

bash scripts/collect_results.sh >/dev/null 2>&1 || true
progress_step 6 6 "snapshot refreshed"
echo "done $(date '+%F %T')" > logs/stage7-queue.done
log "stage7 complete — results in docs/results-snapshot.md; next decision: §9 ablations / §11 export (stage8)"
