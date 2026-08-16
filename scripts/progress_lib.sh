#!/usr/bin/env bash
# progress_lib.sh — generic stage-progress reporting for queue scripts.
# Source from a script that has ROOT set, then call:
#   progress_waiting "detail"              -> state=waiting
#   progress_step <done> <total> "detail"  -> state=running; rate+ETA from prev sample
#   progress_done "detail"                 -> state=done
#   progress_blocked "detail"              -> state=blocked
# Writes outputs/progress/<stage>.json (one small file per stage; single writer).
# ETA: rate = Δdone/Δt vs the previous sample's (_ts, done) kept in the file.
# Stage name derives from $0 (stage4_queue.sh -> stage4); override: PROGRESS_NAME=x

_progress_name() {
  if [ -n "${PROGRESS_NAME:-}" ]; then echo "$PROGRESS_NAME"; return; fi
  local b; b="$(basename "${0:-unknown}")"; echo "${b%_queue.sh}" | sed 's/\.sh$//'
}

_progress_write() {  # state detail done total
  local state="$1" detail="$2" done="$3" total="$4"
  local dir="${ROOT:-$(pwd)}/outputs/progress"
  local f="$dir/$(_progress_name).json"
  local now
  now="$(date +%s)"
  mkdir -p "$dir" 2>/dev/null || true
  python3 - "$f" "$state" "$detail" "$done" "$total" "$now" <<'PY'
import json, sys, os
f, state, detail, done, total, now = sys.argv[1:7]
done, total, now = float(done), float(total), int(now)
try:
    old = json.load(open(f))
except Exception:
    old = {}
prev_ts, prev_done = int(old.get("_ts", 0) or 0), float(old.get("done", 0) or 0)
rate = eta = None
if state == "running" and prev_ts and done > prev_done and now > prev_ts:
    r = (done - prev_done) / (now - prev_ts)
    if r > 0:
        rate = round(r, 4)
        s = int((total - done) / r); h, m = divmod(s, 3600)
        eta = (f"{h}h" if h else "") + f"{m // 60}m"
doc = {"stage": os.path.basename(f)[:-5], "state": state, "detail": detail,
       "done": done, "total": total, "updated": now, "_ts": now,
       "_prev_ts": prev_ts, "_prev_done": prev_done,
       "rate": rate, "eta_human": eta}
os.makedirs(os.path.dirname(f), exist_ok=True)
with open(f, "w") as fh:
    json.dump(doc, fh)
PY
}

progress_waiting() { _progress_write waiting "${1:-}" 0 1; }
progress_step()   { _progress_write running "${3:-}" "${1:-0}" "${2:-1}"; }
progress_done()   { _progress_write done "${1:-}" 1 1; }
progress_blocked(){ _progress_write blocked "${1:-}" 0 1; }
