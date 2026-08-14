#!/usr/bin/env python3
"""PLAN §5 — GLM-5.2 teacher via pi run HEADLESS (owner directive 2026-08-15):
"run Pi as headless agent to get the model to do the work" — no Z.ai key needed.

Host-side driver (pi binary lives on the host, not in the dev container). Stdlib only.

- generate(task, prompt, payload=None): `pi -p --no-approve --thinking <level>` with
  piped stdin merged into the prompt; returns the response text.
- Every call is cached under outputs/corpus/pi_cache/ keyed by sha1(task|prompt|payload|
  model) — identical regenerations are free and runs are resumable.
- Every call is logged to outputs/corpus/pi_calls.jsonl (task, seed, model, wall, chars,
  cache_hit) for the §5 provenance record.
- verify_* helpers enforce machine-checkable ground truth on the OUTPUT side (planted-key
  presence, distractor absence, JSON schema); failures are discarded, never hand-repaired
  (PLAN §5: regenerate-or-discard).

Usage: python3 src/muse_longctx/corpus/pi_teacher.py --selftest
"""
import argparse
import hashlib
import json
import os
import subprocess
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
CACHE = os.path.join(ROOT, "outputs", "corpus", "pi_cache")
CALLS = os.path.join(ROOT, "outputs", "corpus", "pi_calls.jsonl")
DEFAULT_MODEL = os.environ.get("PI_TEACHER_MODEL", "")  # empty = session default (GLM-5.2)


def _key(task, prompt, payload, model):
    h = hashlib.sha1(f"{task}\x00{prompt}\x00{payload or ''}\x00{model}".encode())
    return h.hexdigest()[:16]


def generate(task, prompt, payload=None, thinking="low", model=None, timeout=900,
             allow_fail=True):
    model = model or DEFAULT_MODEL
    os.makedirs(CACHE, exist_ok=True)
    k = _key(task, prompt, payload, model)
    path = os.path.join(CACHE, k + ".txt")
    meta = {"task": task, "seed": k, "model": model or "(session-default)", "thinking": thinking}
    if os.path.exists(path):
        meta.update(cache_hit=True)
        text = open(path).read()
    else:
        cmd = ["pi", "-p", "--no-approve", "--thinking", thinking]
        if model:
            cmd += ["--model", model]
        cmd += [prompt]
        t0 = time.time()
        try:
            r = subprocess.run(cmd, input=payload or "", capture_output=True,
                               text=True, timeout=timeout, cwd=ROOT)
            if r.returncode != 0:
                raise RuntimeError(f"pi exited {r.returncode}: {r.stderr[-400:]}")
            text = r.stdout.strip()
            if not text:
                raise RuntimeError("empty pi response")
            with open(path, "w") as f:
                f.write(text)
            meta.update(cache_hit=False, wall_s=round(time.time() - t0, 1),
                        chars=len(text))
        except Exception as e:  # noqa: BLE001
            meta.update(error=str(e)[:300])
            with open(CALLS, "a") as f:
                f.write(json.dumps(meta) + "\n")
            if allow_fail:
                return None
            raise
    with open(CALLS, "a") as f:
        f.write(json.dumps(meta) + "\n")
    return text


# ---------------------------------------------------------- verification helpers
def verify_planted(text, must_contain, must_not_contain=()):
    """Planted-key verification: each key exactly once (uniqueness = checkable answer),
    distractors absent."""
    problems = []
    for k in must_contain:
        n = text.count(k)
        if n != 1:
            problems.append(f"key {k!r} occurs {n}x (want 1)")
    for k in must_not_contain:
        if k in text:
            problems.append(f"distractor {k!r} present")
    return (not problems), problems


def verify_json(text, schema_keys):
    try:
        start, end = text.find("{"), text.rfind("}")
        obj = json.loads(text[start:end + 1])
    except Exception as e:  # noqa: BLE001
        return False, [f"not-json: {e}"]
    missing = [k for k in schema_keys if k not in obj]
    return (not missing), [f"missing {k}" for k in missing] or []


# ------------------------------------------------------------------- selftest
SELFTEST_PROMPT = (
    "Write ONE paragraph (80-140 words) of realistic technical documentation about a "
    "fictional service. Requirements: mention the maintenance window exactly as "
    "'04:00-06:00 UTC' once and only once; mention the fallback region exactly as "
    "'eu-central-2' once and only once; do NOT mention 'us-east-1' or 'maintenance mode'. "
    "Output only the paragraph, no preamble."
)


def _selftest(args):
    ok = True
    text = generate("selftest-doc", SELFTEST_PROMPT, thinking="low")
    if text is None:
        print("[selftest] generation FAILED (see pi_calls.jsonl)")
        return 1
    good, problems = verify_planted(text, ["04:00-06:00 UTC", "eu-central-2"],
                                    ["us-east-1", "maintenance mode"])
    print(f"[selftest] chars={len(text)} verify={'OK' if good else problems}")
    print(f"[selftest] head: {text[:160]!r}")
    # cache round-trip
    again = generate("selftest-doc", SELFTEST_PROMPT, thinking="low")
    if again != text:
        print("[selftest] CACHE MISMATCH")
        ok = False
    print("[selftest]", "PASS" if good and ok else "FAIL (regeneration needed — this is "
          "the discard path, not an error)")
    return 0 if good and ok else 2


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    _ = ap.parse_args()
    raise SystemExit(_selftest(_))
