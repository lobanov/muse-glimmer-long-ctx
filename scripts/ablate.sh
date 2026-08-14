#!/usr/bin/env bash
# PLAN §9 — ablation runner (MANUAL: run after reviewing §8 + evals/harness/diagnose.py).
#
# Sequenced arms, each: train on train_v1 → merged → §8-subset eval → adapter + results.
# Time-boxed by design: subset grid keeps each arm's evaluation tractable.
#
# Usage:  bash scripts/ablate.sh <subset>       subset ∈ {location, rank}
#   location: all / global / local     (prior per PLAN §9: global ≈ all ≫ local)
#   rank:     8 / 16 / 32 / 64         (run the two bracketing interesting values)
# GPU must be FREE (stage5+ chain finished or vLLM stopped).
set -euo pipefail
SUBSET="${1:?usage: ablate.sh <location|rank>}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
DEV=muse-glimmer-long-ctx-dev-1
DATA=outputs/corpus/train_v1/train.jsonl
LOG() { echo "[$(date '+%F %T')] $*" >> logs/ablate.log; }

run_arm() {  # run_arm <tag> <extra trainer args...>
    local TAG="$1"; shift
    if [ -f "outputs/adapters/$TAG/adapter_config.json" ]; then LOG "$TAG trained (skip)"; else
        LOG "training $TAG: $*"
        docker exec "$DEV" bash -c "cd /workspaces/muse-glimmer-long-ctx && \
            python3 src/muse_longctx/train_qlora.py --data $DATA \
            --out outputs/adapters/$TAG --mode qlora --lr 1e-4 \
            --micro-batch 1 --grad-accum 8 --seq-bucket 131072 --epochs 1 \
            $* > /workspaces/muse-glimmer-long-ctx/logs/train-$TAG.log 2>&1"
        LOG "$TAG training done"
    fi
    if [ -f "outputs/eval/abl_$TAG.jsonl" ]; then LOG "$TAG evaluated (skip)"; else
        LOG "evaluating $TAG"
        # stop anything serving, serve the merged arm at 524k, run decision subset
        docker compose -f .devcontainer/docker-compose.yml --profile inference stop vllm >/dev/null 2>&1 || true
        bash scripts/export_pipeline.sh "outputs/adapters/$TAG" "$TAG" --stage 1 >/dev/null 2>&1
        # merged dir gets the mechanical 524288 window (same rationale as stage7)
        docker exec -i "$DEV" python3 - <<PY
import json
p = "/workspaces/muse-glimmer-long-ctx/outputs/merged/$TAG/config.json"
c = json.load(open(p)); c["text_config"]["max_position_embeddings"] = 524288
json.dump(c, open(p, "w"), indent=2)
PY
        VLLM_MODEL=/outputs/merged/$TAG VLLM_MAX_MODEL_LEN=524288 \
            docker compose -f .devcontainer/docker-compose.yml --profile inference up -d vllm >/dev/null 2>&1
        UP=0
        for _ in $(seq 1 80); do
            docker exec "$DEV" curl -s --max-time 3 http://vllm:8000/v1/models >/dev/null 2>&1 && { UP=1; break; }
            sleep 30
        done
        [ "$UP" = "1" ] || { LOG "ERROR: $TAG never served"; return 1; }
        docker exec "$DEV" bash -c "cd /workspaces/muse-glimmer-long-ctx && \
            python3 evals/harness/run_eval.py --engine vllm --base-url http://vllm:8000/v1 \
            --config-label abl_$TAG --tasks niah,semantic,multihop \
            --ctx 128000,256000,512000 --depths 0.0,0.5,1.0 --reps 3 \
            --mode capability --max-tokens 4096 \
            --out outputs/eval/abl_$TAG.jsonl --write-parquet" >> logs/abl-$TAG-grid.log 2>&1
        LOG "$TAG evaluation done"
    fi
}

case "$SUBSET" in
    location)
        run_arm loc-all    --lora-rank 32 --lora-scope all
        run_arm loc-global --lora-rank 32 --lora-scope global
        run_arm loc-local  --lora-rank 32 --lora-scope local
        ;;
    rank)
        run_arm rank-8  --lora-rank 8  --lora-scope all
        run_arm rank-16 --lora-rank 16 --lora-scope all
        run_arm rank-64 --lora-rank 64 --lora-scope all
        ;;
    *) echo "unknown subset $SUBSET (location|rank)"; exit 1;;
esac

LOG "ablate $SUBSET complete — comparison:"
docker exec "$DEV" bash -c "cd /workspaces/muse-glimmer-long-ctx && \
    python3 evals/harness/compare.py outputs/eval/abl_*.jsonl \
    outputs/eval/stock_vllm_le128k.jsonl outputs/eval/stock_vllm_gt128k.jsonl \
    --ref stock --tasks niah,semantic,multihop --markdown docs/_ablate-$SUBSET.md" \
    >> logs/ablate.log 2>&1 || LOG "WARN: compare step failed (run manually)"
LOG "done — see docs/_ablate-$SUBSET.md"
