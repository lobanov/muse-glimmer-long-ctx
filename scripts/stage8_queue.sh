#!/usr/bin/env bash
# Stage-8 queue (host-side): §11 export chain after §8 evaluation.
#   G1: logs/stage7-queue.done
#   G2: §8 verdict — run1 must NOT be significantly WORSE than stock beyond CI on any
#       ≤128k cell (regression guard); if it is, mark BLOCKED-for-decision and stop
#       (fallback arms per PLAN §7/§10 need a human/agent decision, not blind retry)
# Then (GPU freed first):
#   export stages 2(skip—evaluated in stage7)..7: convert → metadata audit →
#   imatrix(from corpus) → Q4_K_M → dflash/mmproj smoke
# Marker: logs/stage8-queue.done
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
DEV=muse-glimmer-long-ctx-dev-1
COMPOSE="docker compose -f .devcontainer/docker-compose.yml"
log() { echo "[$(date '+%F %T')] $*" >> logs/stage8-queue.log; }
source "$ROOT/scripts/progress_lib.sh"

log "stage8 armed (pid $$)"
progress_waiting "waiting for stage7 (§8 eval)"
while [ ! -f logs/stage7-queue.done ]; do progress_waiting "waiting for stage7 (§8 eval)"; sleep 300; done
progress_step 1 4 "stage7 done; regression guard"

# G2: regression guard (stock comparison on shared cells; any beyond-CI drop >3pts blocks)
VERDICT=$(python3 - <<'PY'
import glob, json, math
from collections import defaultdict
T975 = {2: 4.303, 3: 3.182}
def cells(label, files):
    d = defaultdict(list)
    for f in files:
        for line in open(f):
            try: r = json.loads(line)
            except Exception: continue
            if r.get("error") or r.get("score") is None: continue
            if r["config_label"] != label: continue
            d[(r["task"], r["target_ctx"], r["depth"])].append(r["score"])
    return d
stock = cells("stock", glob.glob("outputs/eval/stock_vllm_le128k.jsonl"))
run1 = cells("run1", glob.glob("outputs/eval/run1_vllm*.jsonl"))
def ci95(xs):  # audit F-5.3: fixed 3pt on n=3 run1 cells false-BLOCKs on sampling
    n = len(xs)            # noise (stock counting@32k=0.952, ~30% stochastic misses);
    if n < 2:              # block only when the drop exceeds the stock cell's CI
        return 0.0
    m = sum(xs)/n
    sd = (sum((x-m)**2 for x in xs)/(n-1)) ** 0.5
    return T975.get(n-1, 1.96) * sd / n**0.5
regressions = []
for k, xs in run1.items():
    if k[1] > 128_000: continue           # guard is the ≤128k regression axis
    s = stock.get(k)
    if not s or len(xs) < 2: continue
    am = sum(xs)/len(xs)
    sm = sum(s)/len(s)
    if sm - am > max(0.03, ci95(s)):      # beyond noise AND ≥3pts (audit F-5.3)
        regressions.append((k, round(sm,3), round(am,3)))
print("BLOCK" if regressions else "GO", regressions[:5])
PY
)
log "G2 verdict: $VERDICT"
case "$VERDICT" in
    BLOCK*) progress_blocked "≤128k regression — fallback decision required"
        echo "blocked-for-decision $(date '+%F %T') :: $VERDICT" > logs/stage8-queue.done
        log "§11 export BLOCKED — ≤128k regression detected; decide fallback (BF16-LoRA / data mix) per PLAN §10"
        exit 0;;
esac

log "freeing GPU (stop vLLM)"
progress_step 2 4 "guard GO; freeing GPU"
$COMPOSE --profile inference stop vllm >/dev/null 2>&1; sleep 10

log "export stages 3–7"
progress_step 3 4 "export chain (convert/imatrix/quantize/dflash)"
bash scripts/export_pipeline.sh outputs/adapters/run1 run1 --stage 3,4,5,6,7 2>&1 \
    | while read -r l; do log "export: $l"; done
KQ=outputs/gguf/run1/run1-Q4_K_M.gguf
if [ -f "$KQ" ]; then
    SZ=$(du -h "$KQ" | cut -f1)
    SZB=$(stat -c%s "$KQ")
    # audit F-7.1: gate the artifact size (GOAL: ~17GB K-quant inside 32GB VRAM;
    # a runaway quant must not pass silently). Ceiling 19 GiB.
    if [ "$SZB" -ge $((19 * 1024 * 1024 * 1024)) ]; then
        progress_blocked "artifact too large: $SZ (≥19GiB) — quant config regression"
        echo "blocked-size $(date '+%F %T') :: $SZ" > logs/stage8-queue.done
        log "ERROR: $KQ is $SZ (≥19GiB) — re-quantize with correct settings"
        exit 0
    fi
    progress_done "artifact: $KQ ($SZ)"
    echo "done $(date '+%F %T') artifact=$KQ size=$SZ" > logs/stage8-queue.done
    log "stage8 complete: $KQ ($SZ) — remaining: §9 ablations, §12 on-device (hardware), §14 report"
else
    echo "failed $(date '+%F %T')" > logs/stage8-queue.done
    log "ERROR: K-Quant artifact missing after export chain — inspect logs above"
fi
