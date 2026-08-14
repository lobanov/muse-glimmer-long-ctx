#!/usr/bin/env bash
# Post-create / CI verification for the dev container.
# Verifies: GPU visibility, NGC torch (not replaced), Muse Glimmer transformers support,
# flash-attn, bitsandbytes, and that the shared caches are mounted where expected.
set -uo pipefail
fail=0
ok()   { echo "  [ok]   $1"; }
bad()  { echo "  [FAIL] $1"; fail=1; }

echo "== GPU =="
if nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader 2>/dev/null; then
  ok "nvidia-smi (host driver libs reachable)"
else
  bad "nvidia-smi — driver libs/devices not passed through (check compose env + scripts/detect-host-gpu.sh)"
fi

echo "== PyTorch (must be the NGC build) =="
python3 - <<'PY' || fail=1
import sys, torch
ver = torch.__version__
print(f"  torch {ver} | cuda runtime {torch.version.cuda}")
if "nv" not in ver and "a0" not in ver:
    print("  [FAIL] torch is not the NGC build — a dependency replaced it"); sys.exit(1)
if not torch.cuda.is_available():
    print("  [FAIL] CUDA not available in container"); sys.exit(1)
name = torch.cuda.get_device_name(0)
cap = torch.cuda.get_device_capability(0)
try:
    free_b, total_b = torch.cuda.mem_get_info()
    x = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
    assert (x @ x).isfinite().all().item(), "bf16 matmul failed"
    print(f"  [ok]   {name} sm_{cap[0]}{cap[1]}, bf16 matmul passes ({free_b/2**30:.1f} GiB free)")
except (torch.OutOfMemoryError, torch.AcceleratorError) as e:
    # Another service (e.g. the vLLM sidecar at 0.9 gpu-memory-utilization) holds the GPU's
    # unified memory — on GB10 even cudaMemGetInfo/context creation raises. Device enumeration
    # above already proved driver + passthrough work, so this is "busy", not "broken".
    if "out of memory" in str(e).lower() or "MemoryAllocation" in str(e):
        print(f"  [warn] {name} sm_{cap[0]}{cap[1]} visible, but CUDA context cannot allocate — "
              "GPU busy (sidecar running?); passthrough itself is OK")
    else:
        raise
PY

echo "== Transformers / Muse Glimmer =="
python3 - <<'PY' || fail=1
import sys, transformers
print(f"  transformers {transformers.__version__}")
need = (5, 15, 0)
have = tuple(int(p) for p in transformers.__version__.split(".")[:3])
if have < need:
    print(f"  [FAIL] need >= {need} for Muse Glimmer, got {have}"); sys.exit(1)
try:
    from transformers.models import muse_glimmer  # noqa: F401
    print("  [ok]   transformers.models.muse_glimmer present")
except ImportError as e:
    print(f"  [FAIL] muse_glimmer module missing: {e}"); sys.exit(1)
try:
    import flash_attn
    print(f"  [ok]   flash_attn {flash_attn.__version__}")
except ImportError:
    print("  [warn] flash_attn not importable (SDPA fallback will be used)")
PY

echo "== Training stack =="
python3 - <<'PY' || fail=1
import sys
import accelerate, peft, trl, bitsandbytes, datasets
print(f"  [ok]   accelerate {accelerate.__version__} | peft {peft.__version__} | "
      f"trl {trl.__version__} | bitsandbytes {bitsandbytes.__version__} | datasets {datasets.__version__}")
PY

echo "== Shared caches =="
for d in /cache/huggingface /cache/torch /cache/triton /cache/weights; do
  if [ -d "$d" ] && touch "$d/.write-test" 2>/dev/null; then rm -f "$d/.write-test"; ok "$d writable"
  else bad "$d missing or read-only"; fi
done
echo "HF_HOME=${HF_HOME:-<unset>}"

echo
[ $fail -eq 0 ] && echo "ALL CHECKS PASSED" || { echo "SOME CHECKS FAILED"; exit 1; }
