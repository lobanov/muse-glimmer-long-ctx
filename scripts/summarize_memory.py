#!/usr/bin/env python3
"""Summarize PLAN §0 memory-spike logs (logs/mem-*.log) into the component table and the
analytic 32 GB RTX 5090 fit (hardware-independent components measured on GB10)."""
import glob, re, sys

ROWS = []
for log in sorted(glob.glob("logs/mem-*.log")):
    m = re.match(r".*mem-(\w+)-(\d+)\.log$", log)
    if not m:
        continue  # e.g. mem-spike-summary.log
    txt = open(log, errors="replace").read()
    kv, ctx = m.group(1), int(m.group(2))

    model = max(map(float, re.findall(r"load_tensors:\s+CUDA0 model buffer size =\s+([\d.]+) MiB", txt)), default=0)
    kv_caches = re.findall(r"llama_kv_cache: size =\s+([\d.]+) MiB \(\s*(\d+) cells,\s+(\d+) layers", txt)
    kv_total = sum(float(s) for s, _, _ in kv_caches) / 2  # logged twice (plan + actual alloc)
    swa = next((f"{cells}c/{layers}L={size}MiB" for size, cells, layers in kv_caches if layers == "39"), "-")
    compute = max(map(float, re.findall(r"CUDA0 compute buffer size =\s+([\d.]+) MiB", txt)), default=0)
    host = sum(map(float, re.findall(r"CUDA_Host (?:model|KV|compute| output) ?buffer size =\s+([\d.]+) MiB", txt)))
    proj = re.search(r"projected to use (\d+) MiB of device memory", txt)

    ROWS.append(dict(kv=kv, ctx=ctx, model=model, kv_total=kv_total, swa=swa,
                     compute=compute, host=host,
                     device=model + kv_total + compute,
                     proj=int(proj.group(1)) if proj else None))

print(f"{'ctx':>7} {'kv':>5} {'model':>8} {'KV':>8} {'compute':>8} {'device':>8} {'proj':>8}  swa-cache")
print("-" * 78)
for r in sorted(ROWS, key=lambda r: (r["ctx"], r["kv"])):
    print(f"{r['ctx']:>7} {r['kv']:>5} {r['model']:>7.0f}M {r['kv_total']:>7.0f}M "
          f"{r['compute']:>7.0f}M {r['device']:>7.0f}M {str(r['proj'] or '-'):>8}  {r['swa']}")

print("\nAnalytic RTX 5090 fit (usable ~31 GB = 32 GB - 1 GB margin):")
print(f"{'ctx':>7} {'kv':>5} {'device-total':>13} {'fits 31GB?':>11}")
for r in sorted(ROWS, key=lambda r: (r["ctx"], r["kv"])):
    gb = r["device"] / 1024
    print(f"{r['ctx']:>7} {r['kv']:>5} {gb:>12.2f}G {'YES' if gb <= 31 else 'NO':>11}")
print("\n(host-side CUDA_Host buffers excluded — pinned host RAM, not VRAM)")
