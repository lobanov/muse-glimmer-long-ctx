#!/usr/bin/env bash
# Stage-6 queue v2 (host-side): CONDITIONAL §7 launch — approval-gated.
#
# Change 2026-08-15 after adversarial review (docs/review-glm53-verification.md, R1):
# the previous version auto-trained on ANY dry-run-OK + corpus gates. Verified facts
# that killed it: trainer-visible corpus = 174 rows → ~21 optimizer steps; winner rule
# detectable effect ≈ +57pts (n=9 binary CI); stage6 never had a no-training branch.
#
# Now: gates on (1) stage5 dry-run OK  (2) corpus bucket-aware size  (3) an EXPLICIT
# human/agent approval marker: logs/train1.approved — created only after reviewing
# §4 sweep + §3 counting/cwe >128k evidence (or a deliberate decision to train anyway).
# The §4-winner computation stays (informational, printed to the log for the decision).
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
DEV=muse-glimmer-long-ctx-dev-1
log() { echo "[$(date '+%F %T')] $*" >> logs/stage6-queue.log; }
source "$ROOT/scripts/progress_lib.sh"

log "stage6-v2 armed (pid $$): approval-gated (requires logs/train1.approved)"
progress_waiting "gates: dry-run OK, corpus, approval marker"

while :; do
    if [ -f logs/stage5-queue.done ]; then
        grep -q "dry-run OK" logs/stage5-queue.done && break
        if grep -q "dry-run FAILED" logs/stage5-queue.done; then
            log "BLOCKED: §7 dry-run failed — fix trainer wiring first"; exit 1
        fi
    fi
    progress_waiting "gate G1: stage5 dry-run"; sleep 300
done
log "G1 ok: stage5 dry-run OK"
progress_step 1 5 "G1 ok"

while pgrep -f 'batch_generate' >/dev/null; do progress_waiting "gate G2: corpus batch running"; sleep 300; done
progress_step 2 5 "G2 ok (batch done)"
python3 - <<'PY' || { log "G3 FAILED: corpus too small at bucket 131072"; exit 1; }
import json, sys
m = json.load(open("outputs/corpus/train_v1/manifest.json"))
b = m.get("length_buckets", {}).get("131072", m)
ok = b["rows"] >= 100 and b["tokens"] >= 5_000_000
print(f'G3@131072: rows={b["rows"]} tokens={b["tokens"]:,} -> {"ok" if ok else "TOO SMALL"}')
sys.exit(0 if ok else 1)
PY
log "G2+G3 ok"

# informational: §4 winner under the OLD per-cell rule (printed for the approval decision)
python3 - <<'PY' >> logs/stage6-queue.log 2>&1
import glob, json, math
from collections import defaultdict
T975 = {2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571}
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
stock = cells("stock", glob.glob("outputs/eval/stock_vllm_*.jsonl") +
          glob.glob("outputs/eval/stock_cwe.jsonl") +
          glob.glob("outputs/eval/stock_weak5.jsonl") +
          glob.glob("outputs/eval/suite_nolima.jsonl"))  # audit F-2.2
for arm in ("qk4.3", "qk5.0"):
    a = cells(arm, glob.glob(f"outputs/eval/arm_{arm}.jsonl"))
    pooled_a, pooled_s = [], []
    percell = []
    for k, xs in a.items():
        s = stock.get(k)
        if not s: continue
        am, sm = sum(xs)/len(xs), sum(s)/len(s)
        pooled_a += xs; pooled_s += s
        percell.append((k, round(am-sm, 3)))
    if pooled_a:
        # pooled sign read (small-n binary: per-cell CIs are near-useless at n<=5)
        wins = sum(1 for k, d in percell if d > 0)
        pm = sum(pooled_a)/len(pooled_a); psm = sum(pooled_s)/len(pooled_s)
        print(f"[winner-info] {arm}: per-cell wins {wins}/{len(percell)}; "
              f"pooled arm {pm:.3f} vs stock {psm:.3f} ({(pm-psm)*100:+.1f} pts, "
              f"n={len(pooled_a)}) — REVIEW before approving train1")
PY

while [ ! -f logs/train1.approved ]; do
    log "waiting for approval marker logs/train1.approved (§4 + §3 evidence review)"
    progress_blocked "AWAITING APPROVAL: logs/train1.approved (review §4 sweep + weak-axis evidence)"
    sleep 600
done
log "APPROVED: $(cat logs/train1.approved 2>/dev/null)"
progress_step 4 5 "approved"

OVERRIDE_JSON=$(python3 - <<'PY'
import glob, json
from collections import defaultdict
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
stock = cells("stock", glob.glob("outputs/eval/stock_vllm_*.jsonl") +
          glob.glob("outputs/eval/stock_cwe.jsonl") +
          glob.glob("outputs/eval/stock_weak5.jsonl") +
          glob.glob("outputs/eval/suite_nolima.jsonl"))  # audit F-2.2
for arm in ("qk4.3", "qk5.0"):
    a = cells(arm, glob.glob(f"outputs/eval/arm_{arm}.jsonl"))
    # harm veto (audit F-1.2): an arm that damages niah@64k retrieval must
    # not become a training override, regardless of weak-axis wins.
    harm = [x for (t, c), xs in a.items() if t == "niah" and c == 64000 for x in xs]
    harm_ok = bool(harm) and sum(harm)/len(harm) >= 0.9
    if harm and not harm_ok:
        print(f"[override] {arm}: HARM niah@64k={sum(harm)/len(harm):.3f} -> veto", file=__import__("sys").stderr)
    wins = 0
    for k, xs in a.items():
        s = stock.get(k)
        if not s or len(xs) < 3: continue
        am, sm = sum(xs)/len(xs), sum(s)/len(s)
        if am - sm > 0.15 and len(xs) >= 5:   # pooled-instance significance is judged by the reviewer; 15pt floor
            wins += 1
    if wins >= 2 and harm_ok:
        print(json.dumps({"qk_scale_factor": float(arm[2:])})); break
else:
    print("")
PY
)
[ -n "$OVERRIDE_JSON" ] && log "qk override: $OVERRIDE_JSON" || log "no qk override (stock knobs)"

docker exec -d "$DEV" bash -c "cd /workspaces/muse-glimmer-long-ctx && \
    python3 src/muse_longctx/train_qlora.py \
    --data outputs/corpus/train_v1/train.jsonl \
    --out outputs/adapters/run1 --mode qlora \
    --lora-rank 32 --lora-scope all --lr 1e-4 \
    --micro-batch 1 --grad-accum 8 --seq-bucket 131072 --epochs 1 \
    $([ -n \"$OVERRIDE_JSON\" ] && echo \"--config-override '$OVERRIDE_JSON'\") \
    > /workspaces/muse-glimmer-long-ctx/logs/train-run1.log 2>&1"
progress_step 5 5 "train1 launched"
echo "launched $(date '+%F %T') override=${OVERRIDE_JSON:-none}" > logs/train1.launched
log "§7 run1 launched (post-approval) — monitor logs/train-run1.log"
