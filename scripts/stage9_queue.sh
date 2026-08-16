#!/usr/bin/env bash
# Stage-9 queue (host-side): §11 quant-parity mini-suite + DFlash acceptance check.
#   G1: stage8 done with an artifact (marker starts with "done" and names the GGUF)
# Then (GPU used serially, freed between):
#   1. serve merged BF16 on vLLM @131072 → greedy parity grid: niah @128k × {0,0.5,1.0}
#      × 3 reps (label run1-bf16-parity)
#   2. vLLM down → llama-server on the new Q4_K_M GGUF (-c 163840, iSWA default) →
#      same parity grid (label run1-gguf-parity)  [greedy: the ONE mode where temp 0 is
#      correct — deterministic quant-noise floor measurement, cf. §0 parity gate]
#   3. verdict: PASS if gguf ≥ bf16 − 0.05 per depth cell (5-pt quant-noise floor ≈
#      §0's 2-pt rule + margin); reasoning-length drains recorded as caveat, not fail
#   4. DFlash acceptance (PLAN §11.6): llama-bench with the stock drafter on the new
#      GGUF vs the stock GGUF — defensive: if flags unsupported in this build, log + skip
# Marker: logs/stage9-queue.done (contains PASS/FAIL verdict)
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
DEV=muse-glimmer-long-ctx-dev-1
COMPOSE="docker compose -f .devcontainer/docker-compose.yml"
GGUF=outputs/gguf/run1/run1-Q4_K_M.gguf
MMPROJ=/cache/weights/mmproj-Muse-Glimmer-30B-Q4_K_M.gguf
DRAFT=/cache/weights/dflash-Muse-Glimmer-30B-Q4_K_M.gguf
STOCK=/cache/weights/Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf
log() { echo "[$(date '+%F %T')] $*" >> logs/stage9-queue.log; }
source "$ROOT/scripts/progress_lib.sh"

log "stage9 armed (pid $$)"
progress_waiting "waiting for stage8 (export artifact)"
while [ ! -f logs/stage8-queue.done ]; do progress_waiting "waiting for stage8 (export artifact)"; sleep 300; done
progress_step 1 5 "artifact present; BF16 parity"
grep -q "^done" logs/stage8-queue.done || { echo "skipped (stage8 not clean)" > logs/stage9-queue.done; log "stage8 not done-clean — skipping"; exit 0; }
[ -f "$GGUF" ] || { echo "failed (artifact missing)" > logs/stage9-queue.done; log "ERROR: $GGUF missing"; exit 1; }
log "artifact present: $(du -h "$GGUF" | cut -f1)"

# ---- 1. BF16-merged parity side -------------------------------------------------
$COMPOSE --profile inference stop vllm >/dev/null 2>&1; sleep 10
VLLM_MODEL=/outputs/merged/run1 VLLM_MAX_MODEL_LEN=131072 \
    $COMPOSE --profile inference up -d vllm >/dev/null 2>&1
UP=0
for _ in $(seq 1 80); do
    docker exec "$DEV" curl -s --max-time 3 http://vllm:8000/v1/models >/dev/null 2>&1 && { UP=1; break; }
    sleep 30
done
[ "$UP" = "1" ] || { echo "failed (bf16 never served)" > logs/stage9-queue.done; log "ERROR: merged never served"; exit 1; }
log "BF16-merged serving; parity grid (niah @128k, greedy)"
docker exec "$DEV" bash -c "cd /workspaces/muse-glimmer-long-ctx && \
    python3 evals/harness/run_eval.py --engine vllm --base-url http://vllm:8000/v1 \
    --config-label run1-bf16-parity --tasks niah --ctx 128000 \
    --depths 0.0,0.5,1.0 --reps 3 --mode parity --max-tokens 4096 \
    --out outputs/eval/parity_run1_bf16.jsonl" >> logs/stage9-grid.log 2>&1
$COMPOSE --profile inference stop vllm >/dev/null 2>&1; sleep 10

# ---- 2. GGUF parity side ---------------------------------------------------------
log "starting llama-server on new GGUF (-c 163840)"
progress_step 2 5 "GGUF parity grid"
$COMPOSE --profile llamacpp run -d llamacpp \
    llama-server -m /cache/weights/export/run1/run1-Q4_K_M.gguf --mmproj "$MMPROJ" \
    -ngl 99 -c 163840 --host 0.0.0.0 --port 8080 --jinja \
    --temp 0.0 >/dev/null 2>&1
LLNAME=$(docker ps --filter label=com.docker.compose.service=llamacpp --format '{{.Names}}' | head -1)
UP=0
for _ in $(seq 1 90); do
    [ -n "$LLNAME" ] && docker exec "$DEV" curl -s --max-time 3 "http://$LLNAME:8080/v1/models" >/dev/null 2>&1 && { UP=1; break; }
    sleep 10
done
[ "$UP" = "1" ] || { echo "failed (gguf never served)" > logs/stage9-queue.done; log "ERROR: llama-server never came up"; exit 1; }
docker exec "$DEV" bash -c "cd /workspaces/muse-glimmer-long-ctx && \
    python3 evals/harness/run_eval.py --engine llamacpp --base-url http://$LLNAME:8080/v1 \
    --config-label run1-gguf-parity --tasks niah --ctx 128000 \
    --depths 0.0,0.5,1.0 --reps 3 --mode parity --max-tokens 4096 \
    --out outputs/eval/parity_run1_gguf.jsonl" >> logs/stage9-grid.log 2>&1

# ---- 3. verdict ------------------------------------------------------------------
VERDICT=$(python3 - <<'PY'
import json
from collections import defaultdict
def cells(path):
    d = defaultdict(list)
    try:
        for line in open(path):
            r = json.loads(line)
            if not r.get("error") and r.get("score") is not None:
                d[r["depth"]].append(r["score"])
    except FileNotFoundError:
        pass
    return d
b, g = cells("outputs/eval/parity_run1_bf16.jsonl"), cells("outputs/eval/parity_run1_gguf.jsonl")
if not b or not g:
    print("FAIL insufficient data", len(b), len(g)); raise SystemExit
bad, drains = [], 0
for d in sorted(set(b) | set(g)):
    bm = sum(b[d])/len(b[d]) if b.get(d) else float("nan")
    gm = sum(g[d])/len(g[d]) if g.get(d) else float("nan")
    if bm - gm > 0.05: bad.append((d, round(bm,3), round(gm,3)))
for line in open("outputs/eval/parity_run1_gguf.jsonl"):
    r = json.loads(line)
    if r.get("finish_reason") == "length": drains += 1
print(("PASS" if not bad else f"FAIL {bad}") + (f" | length-drains: {drains}" if drains else ""))
PY
)
log "parity verdict: $VERDICT"
progress_step 3 5 "verdict: $VERDICT"

# ---- 4. DFlash acceptance (defensive) --------------------------------------------
$COMPOSE --profile llamacpp run --rm --no-deps llamacpp bash -c "
B=/src/llama.cpp/build/bin; W=/cache/weights/export/run1
echo '--- new GGUF + stock drafter ---'
timeout 900 \$B/llama-bench -m \$W/run1-Q4_K_M.gguf -md $DRAFT -p 512 -n 128 2>&1 | tail -6
echo '--- stock GGUF + stock drafter (reference) ---'
timeout 900 \$B/llama-bench -m $STOCK -md $DRAFT -p 512 -n 128 2>&1 | tail -6
" >> logs/stage9-dflash.log 2>&1 || log "WARN: llama-bench draft run failed/unsupported — inspect logs/stage9-dflash.log"

docker rm -f "$LLNAME" >/dev/null 2>&1 || true
progress_step 4 5 "dflash acceptance check"
progress_done "quant-parity: $VERDICT"
echo "done $(date '+%F %T') :: $VERDICT" > logs/stage9-queue.done
log "stage9 complete — $VERDICT (results: outputs/eval/parity_run1_*.jsonl, logs/stage9-dflash.log)"
