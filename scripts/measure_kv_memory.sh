#!/usr/bin/env bash
# PLAN §0.4 — measure llama.cpp component memory (model / KV / compute buffers) for the
# K-Quant GGUF at increasing context sizes, on the DGX Spark. Component sizes are
# hardware-independent; the 32 GB RTX 5090 fit is decided analytically from them
# (weights + KV + compute buffers <= ~31 GB usable; CUDA_Host buffers are host RAM).
#
# Also captures the iSWA cache lines (§0.3 verification: SWA layers must get a
# window-sized cache, not full-context).
#
# Requires: the vLLM sidecar STOPPED (it holds 0.9 GPU); GGUF in cache/weights/.
# Output: logs/mem-<kvtype>-<ctx>.log + a parsed summary table on stdout.
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/.devcontainer"

GGUF_NAME="Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf"
LOGDIR=../logs
mkdir -p "$LOGDIR"

run_one() {  # ctx kvtype extra-flags...
  local ctx=$1 kv=$2; shift 2
  local log="$LOGDIR/mem-${kv}-${ctx}.log"
  echo ">> ctx=$ctx kv=$kv ..."
  # pipe /exit: conversation mode reads it as its exit command and quits cleanly.
  # (With </dev/null the CLI spins printing empty "> " prompts until timeout kills it —
  # GPU sits idle the whole time; measured the hard way.)
  printf '/exit\n' | timeout 600 docker compose --profile llamacpp run -T --rm llamacpp \
    llama-cli -m "/cache/weights/${GGUF_NAME}" -ngl 99 -c "$ctx" \
    -p "hello" -n 1 --temp 0 --verbose "$@" >"$log" 2>&1 || true
}

# args (optional): ctx:kvtype specs, e.g. 524288:f16 524288:q8_0 — default full matrix.
# q8_0 implies '-fa on -ctk q8_0 -ctv q8_0'. Already-summarized logs can be skipped by
# passing only the missing specs.
SPECS=("${@:-131072:f16 262144:f16 393216:f16 524288:f16 524288:q8_0}")
for spec in ${SPECS[@]}; do
  ctx="${spec%%:*}"; kv="${spec##*:}"
  if [ "$kv" = q8_0 ]; then
    run_one "$ctx" "$kv" -fa on -ctk q8_0 -ctv q8_0
  else
    run_one "$ctx" "$kv"
  fi
done

echo
printf "%-10s %-9s %10s %10s %10s %10s %10s\n" ctx kv modelGB kvGB computeGB hostGB totalGB
for log in "$LOGDIR"/mem-*.log; do
  base=$(basename "$log" .log)                       # mem-<kv>-<ctx>
  kv=${base#mem-}; ctx=${kv#*-}; kv=${kv%%-*}
  # CUDA0 device-side buffers (what counts against 5090 VRAM)
  model=$(grep -oE 'load_tensors: +CUDA0 model buffer size = [0-9.]+ MiB' "$log" | grep -oE '[0-9.]+')
  kvbuf=$(grep -oE 'CUDA0 KV buffer size = [0-9.]+ MiB' "$log" | grep -oE '[0-9.]+' | paste -sd+ | bc)
  comp=$(grep -oE 'CUDA0 compute buffer size = [0-9.]+ MiB' "$log" | grep -oE '[0-9.]+')
  host=$(grep -oE '(CUDA_Host|CPU) (model|KV|compute) buffer size = [0-9.]+ MiB' "$log" | grep -oE '[0-9.]+' | paste -sd+ | bc)
  total=$(printf '%s+%s+%s\n' "${model:-0}" "${kvbuf:-0}" "${comp:-0}" | bc)
  printf "%-10s %-9s %10s %10s %10s %10s %10s\n" "$ctx" "$kv" \
    "${model:--}" "${kvbuf:--}" "${comp:--}" "${host:--}" "${total:--}"
done
echo
echo "iSWA check (128k run):"; grep -E 'kv_cache_iswa|SWA' "$LOGDIR/mem-f16-131072.log" | head -6 || true
