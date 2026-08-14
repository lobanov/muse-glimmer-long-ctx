#!/usr/bin/env python3
"""JSONL -> common Parquet schema (PLAN §2). Run where pyarrow lives (dev container)."""
import argparse
import glob
import json

import pyarrow as pa
import pyarrow.parquet as pq

SCHEMA = pa.schema([
    ("run_id", pa.string()), ("ts", pa.string()), ("engine", pa.string()),
    ("model", pa.string()), ("config_label", pa.string()), ("mode", pa.string()),
    ("task", pa.string()), ("target_ctx", pa.int64()), ("depth", pa.float64()),
    ("rep", pa.int64()), ("cell_id", pa.string()), ("sampling", pa.string()),
    ("prompt_tokens", pa.int64()), ("completion_tokens", pa.int64()),
    ("finish_reason", pa.string()), ("wall_s", pa.float64()),
    ("ttft_s", pa.float64()), ("tok_per_s", pa.float64()),
    ("score", pa.float64()), ("detail", pa.string()), ("expected", pa.string()),
    ("response_head", pa.string()), ("reasoning_head", pa.string()),
    ("reasoning", pa.string()), ("error", pa.string()),
])

_INT = {"target_ctx", "rep", "prompt_tokens", "completion_tokens"}
_FLOAT = {"depth", "wall_s", "ttft_s", "tok_per_s", "score"}


def _coerce(col, v):
    if v is None:
        return None
    try:
        if col in _INT:
            return int(v)
        if col in _FLOAT:
            return float(v)
    except (TypeError, ValueError):
        return None
    return str(v)


def convert(paths, out_parquet):
    rows = []
    for p in paths:
        for g in sorted(glob.glob(p)):
            with open(g) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        r = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    row = {}
                    for field in SCHEMA:
                        c = field.name
                        if c == "sampling":
                            row[c] = json.dumps(r.get("sampling"))
                        elif c == "expected":
                            row[c] = json.dumps(r.get("expected"))
                        else:
                            row[c] = _coerce(c, r.get(c))
                    rows.append(row)
    if not rows:
        raise SystemExit("no rows found")
    pq.write_table(pa.Table.from_pylist(rows, schema=SCHEMA), out_parquet)
    print(f"{len(rows)} rows -> {out_parquet}")
    return out_parquet


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl", nargs="+")
    ap.add_argument("-o", "--out", required=True)
    a = ap.parse_args()
    convert(a.jsonl, a.out)
