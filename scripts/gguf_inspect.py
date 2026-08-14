#!/usr/bin/env python3
"""Read-only GGUF metadata inspector (pure python + numpy-free).

Walks the GGUF v3 header + metadata KV section only (no tensors). Used for:
- PLAN §4 spike: does the official GGUF carry qk_scale_factor / layer_rope_theta?
- PLAN §11 step 4: verify metadata survived merge -> convert -> quantize.

Usage: python3 scripts/gguf_inspect.py <model.gguf> [--match substr]
"""
import struct
import sys

TYPES = {0: ("<B", 1), 1: ("<b", 1), 2: ("<H", 2), 3: ("<h", 2), 4: ("<I", 4),
         5: ("<i", 4), 6: ("<f", 4), 7: ("<B", 1), 10: ("<Q", 8), 11: ("<q", 8),
         12: ("<d", 8)}


def read_str(f):
    (n,) = struct.unpack("<Q", f.read(8))
    return f.read(n).decode("utf-8", "replace")


def read_value(f, vtype):
    if vtype == 8:
        return read_str(f)
    if vtype == 9:
        (etype,) = struct.unpack("<I", f.read(4))
        (n,) = struct.unpack("<Q", f.read(8))
        return [read_value(f, etype) for _ in range(n)]
    fmt, size = TYPES[vtype]
    (v,) = struct.unpack(fmt, f.read(size))
    return bool(v) if vtype == 7 else v


def inspect(path, match=None):
    with open(path, "rb") as f:
        magic = f.read(4)
        assert magic == b"GGUF", f"not a GGUF file: {magic!r}"
        (version,) = struct.unpack("<I", f.read(4))
        (n_tensors,) = struct.unpack("<Q", f.read(8))
        (n_kv,) = struct.unpack("<Q", f.read(8))
        print(f"# {path}\n# gguf v{version} tensors={n_tensors} metadata_kvs={n_kv}")
        for _ in range(n_kv):
            key = read_str(f)
            (vtype,) = struct.unpack("<I", f.read(4))
            val = read_value(f, vtype)
            if match is None or match in key:
                s = repr(val)
                if isinstance(val, list) and len(val) > 24:
                    s = f"[{len(val)} items] {val[:8]} ... {val[-4:]}"
                print(f"{key} = {s[:400]}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    m = None
    if "--match" in sys.argv:
        i = sys.argv.index("--match")
        m = sys.argv[i + 1]
        del sys.argv[i:i + 2]
    inspect(sys.argv[1], m)
