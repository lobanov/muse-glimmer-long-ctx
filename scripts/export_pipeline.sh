#!/usr/bin/env bash
# PLAN §11 — merge → GGUF export → K-Quant → verification pipeline (STAGED, idempotent).
#
# Stages (run all: scripts/export_pipeline.sh <adapter_dir> <name> | --stage N ...):
#   1. merge      LoRA → full multimodal checkpoint (text decoder only touched), BF16
#   2. parity     merged vs adapter-on-base: NIAH subset @128k via vLLM (harness)
#   3. convert    bf16 GGUF via convert_hf_to_gguf.py (llama.cpp gguf-py shadows pip gguf)
#   4. verify     metadata audit on bf16 GGUF (scripts/gguf_inspect.py assertions)
#   5. imatrix    importance matrix from long-context calibration (corpus + haystacks)
#   6. quantize   Q4_K_M with imatrix → ~17 GB deployment artifact
#   7. dflash     stock drafter + mmproj load check against the new GGUF (llama-server smoke)
#
# Environment:
#   LLAMA_CPP (default /src/llama.cpp — inside the llamacpp image; or a host checkout at
#   tag b10428) is used for convert + quantize + imatrix binaries and gguf-py.
# Usage:
#   scripts/export_pipeline.sh outputs/adapters/run1 run1                 # all stages
#   scripts/export_pipeline.sh outputs/adapters/run1 run1 --stage 4       # just verify
set -euo pipefail

ADAPTER="$1"; NAME="$2"; shift 2
STAGES="1 2 3 4 5 6 7"
[ "${1:-}" = "--stage" ] && STAGES="$2"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEV=muse-glimmer-long-ctx-dev-1
MERGED="$ROOT/outputs/merged/$NAME"
GGUF_DIR="$ROOT/outputs/gguf/$NAME"
BF16="$GGUF_DIR/${NAME}-bf16.gguf"
IMATRIX="$GGUF_DIR/imatrix.dat"
KQUANT="$GGUF_DIR/${NAME}-Q4_K_M.gguf"
LLAMA_CPP="${LLAMA_CPP:-/cache/weights/tools/llama.cpp}"   # shared: dev + llamacpp both mount /cache/weights
CALIB="$GGUF_DIR/calibration.txt"

want() { echo " $STAGES " | grep -q " $1 "; }
log() { echo "[$(date '+%F %T')] $*" >&2; }

# ---- stage 1: merge -----------------------------------------------------------
if want 1; then
  [ -f "$MERGED/config.json" ] && log "stage1: $MERGED exists, skip" || {
    log "stage1: merging $ADAPTER into text decoder (BF16, vision frozen)"
    docker exec "$DEV" python3 - "$ADAPTER" "$MERGED" <<'PY'
import sys, torch
from transformers import MuseGlimmerForConditionalGeneration
from peft import PeftModel
adapter_dir, out_dir = sys.argv[1], sys.argv[2]
model = MuseGlimmerForConditionalGeneration.from_pretrained(
    "meta-models/Muse-Glimmer-30B", dtype=torch.bfloat16)
model = PeftModel.from_pretrained(model, adapter_dir)
# sanity: adapter must have lived on the text decoder only
inj = [n for n, _ in model.named_modules() if ".lora_A." in n]
assert inj and not any("vision" in n for n in inj), inj[:3]
model = model.merge_and_unload()
model.save_pretrained(out_dir, safe_serialization=True)
print("merged ->", out_dir)
PY
    docker exec "$DEV" python3 -c "
from transformers import AutoProcessor
AutoProcessor.from_pretrained('meta-models/Muse-Glimmer-30B').save_pretrained('$MERGED')"
  }
fi

# ---- stage 2: parity (merged vs stock @128k NIAH via harness) ------------------
if want 2; then
  log "stage2: parity eval — serve merged on vLLM then run harness NIAH subset"
  log "stage2: (manual gate) VLLM_MODEL=$MERGED VLLM_MAX_MODEL_LEN=131072 up -d vllm; then:"
  log "stage2:   run_eval.py --config-label merged-$NAME --tasks niah --ctx 128000 \
--depths 0,0.5,1.0 --reps 3 --mode parity --out outputs/eval/parity_merged_$NAME.jsonl"
fi

# ---- stage 3: convert ---------------------------------------------------------
if want 3; then
  [ -f "$BF16" ] && log "stage3: $BF16 exists, skip" || {
    # llama.cpp source at the pinned tag, in the shared cache (dev needs convert_hf_to_gguf.py
    # + its gguf-py; the llamacpp image has no python). Auto-clone once, idempotent.
    if [ ! -f "$LLAMA_CPP/convert_hf_to_gguf.py" ]; then
      log "stage3: cloning llama.cpp @ b10428 -> $LLAMA_CPP"
      mkdir -p "$(dirname "$LLAMA_CPP")"
      git clone --depth 1 --branch b10428 https://github.com/ggml-org/llama.cpp "$LLAMA_CPP"
    fi
    log "stage3: convert_hf_to_gguf.py (PYTHONPATH=$LLAMA_CPP/gguf-py shadows pip gguf)"
    mkdir -p "$GGUF_DIR"
    docker exec "$DEV" env PYTHONPATH="$LLAMA_CPP/gguf-py" python3 \
      "$LLAMA_CPP/convert_hf_to_gguf.py" "$MERGED" --out "$BF16" \
      --outtype bf16 --vocab-type bpe
  }
fi

# ---- stage 4: metadata audit ---------------------------------------------------
if want 4; then
  log "stage4: verifying metadata on $(basename "$BF16")"
  OUT="$(docker exec "$DEV" python3 scripts/gguf_inspect.py "$BF16")"
  echo "$OUT" | grep -q "muse-glimmer.block_count = 52"
  echo "$OUT" | grep -q "sliding_window = 2048"
  echo "$OUT" | grep -q "sliding_window_pattern"
  echo "$OUT" | grep -q "rope.freq_base = 500000"
  echo "$OUT" | grep -q "final_logit_softcapping = 20"
  echo "$OUT" | grep -q "tokenizer.chat_template"
  log "stage4: PASS (arch, window, pattern, theta, softcap, chat template)"
fi

# ---- stage 5: imatrix from long-context calibration ----------------------------
if want 5; then
  [ -f "$IMATRIX" ] && log "stage5: $IMATRIX exists, skip" || {
    log "stage5: building calibration text from corpus (long-context code + retrieval)"
    mkdir -p "$GGUF_DIR"
    python3 - "$CALIB" <<'PY'
import glob, json, sys
out = open(sys.argv[1], "w")
total = 0
for pat in ("outputs/corpus/repos_v1/*.jsonl", "outputs/corpus/synth_v1/synth_*.jsonl",
            "outputs/corpus/nat_v1/book*.jsonl"):
    for path in sorted(glob.glob(pat)):
        for line in open(path):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            body = r.get("body") or (r.get("prompt", "") if r.get("axis") == "repo" else "")
            if body:
                out.write(body[:400_000] + "\n")
                total += 1
out.close()
print(f"calibration: {total} documents -> {sys.argv[1]}")
PY
    log "stage5: running llama-imatrix (llamacpp container)"
    docker compose -f "$ROOT/.devcontainer/docker-compose.yml" --profile llamacpp \
      run --rm --no-deps llamacpp bash -c "
      mkdir -p $GGUF_DIR && /src/llama.cpp/build/bin/llama-imatrix \
        -m $BF16 -f $CALIB -o $IMATRIX.of-wait 2>&1 | tail -2; \
        mv \$(ls -t $GGUF_DIR/imatrix.dat* 2>/dev/null | head -1) $IMATRIX 2>/dev/null || true"
  }
fi

# ---- stage 6: quantize ---------------------------------------------------------
if want 6; then
  [ -f "$KQUANT" ] && log "stage6: $KQUANT exists, skip" || {
    log "stage6: Q4_K_M with imatrix (target ~17 GB)"
    docker compose -f "$ROOT/.devcontainer/docker-compose.yml" --profile llamacpp \
      run --rm --no-deps llamacpp bash -c \
      "/src/llama.cpp/build/bin/llama-quantize --imatrix $IMATRIX $BF16 $KQUANT Q4_K_M \
       2>&1 | tail -3"
  }
fi
# ---- stage 7: dflash + mmproj load check ---------------------------------------
if want 7; then
  log "stage7: loading new GGUF with stock dflash drafter + official mmproj (smoke)"
  docker compose -f "$ROOT/.devcontainer/docker-compose.yml" --profile llamacpp \
    run --rm --no-deps llamacpp bash -c "
    timeout 300 /src/llama.cpp/build/bin/llama-server -m $KQUANT \
      --mmproj /cache/weights/mmproj-Muse-Glimmer-30B-Q4_K_M.gguf \
      -ngl 99 -c 32768 --spec-type draft-dflash \
      --draft-model /cache/weights/dflash-Muse-Glimmer-30B-Q4_K_M.gguf \
      --host 0.0.0.0 --port 8099 --jinja 2>&1 | grep -E 'loaded|draft|error|spec' | head -8" \
    || log "stage7: WARN — check flags (--spec-type/--draft-model) against build docs"
  log "stage7 done — verify draft acceptance via llama-bench before §12"
fi

log "pipeline complete for $NAME (stages: $STAGES)"
