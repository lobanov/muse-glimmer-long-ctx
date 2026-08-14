#!/usr/bin/env bash
# Stage-6 queue (host-side): launch §7 first QLoRA run when everything is ready.
# Gates (ALL required, checked in order):
#   G1 stage5 done with "dry-run OK"          (trainer wiring proven, GPU free)
#   G2 corpus batch finished (process gone)   (batch_generate.py)
#   G3 corpus volume: train_v1 ≥ 100 rows and ≥ 5M tokens (manifest.json)
# Then:
#   - pick §4 winner: qk arm that beats stock beyond CI on ≥2 of {128k,256k,512k}×{niah,
#     semantic} (python inline over outputs/eval/arm_*.jsonl); else no override (stock knobs)
#   - launch QLoRA detached inside dev (log: logs/train-run1.log), marker logs/train1.launched
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
DEV=muse-glimmer-long-ctx-dev-1
log() { echo "[$(date '+%F %T')] $*" >> logs/stage6-queue.log; }

log "stage6 armed (pid $$)"
while :; do
    if [ -f logs/stage5-queue.done ]; then
        grep -q "dry-run OK" logs/stage5-queue.done && break
        if grep -q "dry-run FAILED" logs/stage5-queue.done; then
            echo "blocked: stage5 dry-run FAILED $(date '+%F %T')" > logs/train1.launched
            log "BLOCKED: §7 dry-run failed (see logs/stage5-queue.log) — NOT training; fix trainer wiring first"
            exit 1
        fi
    fi
    sleep 300
done
log "G1 ok: stage5 dry-run OK"
while pgrep -f 'batch_generate' >/dev/null; do sleep 300; done
log "G2 ok: corpus batch finished"
python3 - <<'PY' || { log "G3 FAILED: corpus too small — NOT launching (needs manual scale-up)"; exit 1; }
import json, sys
m = json.load(open("outputs/corpus/train_v1/manifest.json"))
ok = m["rows"] >= 100 and m["tokens"] >= 5_000_000
print(f'G3: rows={m["rows"]} tokens={m["tokens"]:,} -> {"ok" if ok else "TOO SMALL"}')
sys.exit(0 if ok else 1)
PY

# §4 winner: significant win beyond CI on >=2 task×ctx cells vs stock
OVERRIDE_JSON=$(python3 - <<'PY'
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
            d[(r["task"], r["target_ctx"])].append(r["score"])
    return d
stock = cells("stock", glob.glob("outputs/eval/stock_vllm_*.jsonl"))
for arm in ("qk4.1", "qk4.3", "qk4.6", "qk5.0"):
    a = cells(arm, glob.glob(f"outputs/eval/arm_{arm}.jsonl"))
    wins = 0
    for k, xs in a.items():
        s = stock.get(k)
        if not s or len(xs) < 2: continue
        am, ci = sum(xs)/len(xs), T975.get(len(xs)-1, 1.96)*(0.0 if len(xs)<2 else
            math.sqrt(sum((x-sum(xs)/len(xs))**2 for x in xs)/(len(xs)-1)))/math.sqrt(len(xs))
        sm = sum(s)/len(s)
        if am - ci > sm and (am - sm) * 100 > 3:
            wins += 1
    if wins >= 2:
        val = arm[2:]
        print(json.dumps({"qk_scale_factor": float(val)}))
        break
else:
    print("")
PY
)
if [ -n "$OVERRIDE_JSON" ]; then
    log "§4 winner detected -> --config-override $OVERRIDE_JSON"
else
    log "no significant §4 winner -> training with stock knobs (3.87)"
fi

docker exec -d "$DEV" bash -c "cd /workspaces/muse-glimmer-long-ctx && \
    python3 src/muse_longctx/train_qlora.py \
    --data outputs/corpus/train_v1/train.jsonl \
    --out outputs/adapters/run1 --mode qlora \
    --lora-rank 32 --lora-scope all --lr 1e-4 \
    --micro-batch 1 --grad-accum 8 --seq-bucket 131072 --epochs 1 \
    $([ -n \"$OVERRIDE_JSON\" ] && echo \"--config-override '$OVERRIDE_JSON'\") \
    > /workspaces/muse-glimmer-long-ctx/logs/train-run1.log 2>&1"
echo "launched $(date '+%F %T') override=${OVERRIDE_JSON:-none}" > logs/train1.launched
log "§7 run1 launched — monitor: logs/train-run1.log (loss lines every 10 steps)"
