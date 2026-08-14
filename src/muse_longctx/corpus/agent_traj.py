#!/usr/bin/env python3
"""PLAN §5 — coding-agent trajectory generator: pi headless AS the agent (owner model).

pi (--mode json, cwd = a materialized training-only repo) genuinely explores the repo
with tools; we capture the full session (assistant text + tool calls + results) and build
SFT samples where the answer depends on information the agent observed EARLY in the
session — the GOAL.md "persistence across long tool-use trajectories" axis, as data.

Ground truth is machine-checkable WITHOUT trusting the model: facts are extracted from
repo files pre-run (unique-value check across the whole repo); a sample is accepted only
if the fact occurs exactly once in the captured transcript (i.e. the agent actually
observed it).

Output = serialize.py-compatible doc file:
  line 0: {"body": "", "ledger": null, "target_tokens": null}
  line i: {"prompt", "answer", "question", "axis": "agentmem"}

Usage (host): python3 src/muse_longctx/corpus/agent_traj.py --repo tiangolo/typer \
                  --out outputs/corpus/agent_v1/typer.jsonl --validate
"""
import argparse
import json
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from github_repos import fetch_repo  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
WORK = os.path.join(ROOT, "outputs", "corpus", "agent_work")

FACT_RE = re.compile(r'^\s*(version|name|port|app)\s*[:=]\s*["\']?([\w][\w.+-]{2,39})["\']?',
                     re.M)
DUNDER_RE = re.compile(r'^\s*__version__\s*=\s*["\']([\d][\w.+-]{2,39})["\']', re.M)
CONFIG_EXT = (".toml", ".cfg", ".ini", ".yaml", ".yml", ".json", ".txt")
SKIP_DIRS = ("test", ".github", "docs", "examples")

TASK_TMPL = (
    "You are exploring an unfamiliar repository to write an onboarding guide for new "
    "engineers. Work systematically with your tools (ls/read/grep): (1) map the repository "
    "structure; (2) record the key project metadata (version, entry points, notable "
    "configuration values) exactly as found in files; (3) inspect AT LEAST 8 different "
    "source modules spread across the tree — note each module's responsibility and one "
    "notable function/class with a one-line excerpt; (4) trace one cross-module dependency "
    "chain. Finish with the guide: a component map plus handover bullets restating the "
    "key metadata exactly as found. Use at least 14 tool calls; do not paraphrase file "
    "excerpts."
)


def extract_facts(files):
    """(label, value) pairs whose value is UNIQUE across the entire rendered repo."""
    full = "\n".join(files.values())
    cands = []
    for path, body in files.items():
        if any(path.startswith(s) for s in SKIP_DIRS):
            continue
        # tier A: __version__ anywhere (highest signal)
        for value in DUNDER_RE.findall(body):
            if full.count(value) == 1:
                cands.append(("version", value, path))
        # tier B: version/name/port/app from CONFIG files only (code assignments are noise)
        if os.path.splitext(path)[1] in CONFIG_EXT:
            for label, value in FACT_RE.findall(body):
                if full.count(value) == 1 and not value.lower().startswith(("true", "false", "null")):
                    cands.append((label, value, path))
    # tier A first (version facts), then shorter values (tighter, more checkable)
    tier = lambda c: 0 if (c[0] == "version" and "__version__" in files.get(c[2], "")) else 1
    cands.sort(key=lambda c: (tier(c), len(c[1])))
    return cands


def materialize(repo):
    slug = repo.replace("/", "__")
    work = os.path.join(WORK, slug)
    if not os.path.isdir(work):
        files, _lic = fetch_repo(repo)
        for path, body in files.items():
            # agent sessions run on the CODE tree (docs/tests/examples dropped: they
            # duplicate facts and dilute trajectories; fact uniqueness is judged over
            # what the agent can actually observe)
            if any(path.startswith(s) or path.split("/")[0] in SKIP_DIRS
                   for s in SKIP_DIRS):
                continue
            dst = os.path.join(work, path)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            with open(dst, "w") as f:
                f.write(body)
        open(os.path.join(work, ".gitignore"), "a").close()
    return work


def run_agent(work, timeout=1200):
    """Run pi headless in `work`; return transcript text (assistant + tool results).
    Fact-agnostic broad exploration; facts are harvested post-hoc from the transcript.
    --approve: the cwd is a materialized scratch copy of a permissive training-only
    repo — tool auto-approval is safe and REQUIRED (--no-approve denies tool use)."""
    task = TASK_TMPL
    cmd = ["pi", "--mode", "json", "-p", "--approve", "--thinking", "low", task]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=work)
    if r.returncode != 0:
        raise RuntimeError(f"pi exited {r.returncode}: {r.stderr[-300:]}")
    events = []
    for line in r.stdout.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    turns = [e for e in events if e.get("type") == "turn_end"]
    if not turns:
        raise RuntimeError("no turn_end events captured")
    parts = [f"<user>{task}</user>"]
    for i, t in enumerate(turns, 1):
        msg = t.get("message") or {}
        text = "".join(c.get("text", "") for c in msg.get("content", [])
                       if isinstance(c, dict))
        if text.strip():
            parts.append(f"<assistant-{i}>{text.strip()}</assistant-{i}>")
        for tr in t.get("toolResults", []):
            name = tr.get("toolName") or tr.get("name") or "tool"
            out = tr.get("output") or tr.get("result") or ""
            if not isinstance(out, str):
                out = json.dumps(out)
            parts.append(f'<tool-{i} name="{name}">{out[:6000]}</tool-{i}>')
    return "\n\n".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--max-samples", type=int, default=3)
    args = ap.parse_args()

    work = materialize(args.repo)
    # re-scan the materialized tree (single source of truth for facts)
    files = {}
    for dirpath, _dirs, fnames in os.walk(work):
        for fn in fnames:
            p = os.path.join(dirpath, fn)
            rel = os.path.relpath(p, work)
            try:
                files[rel] = open(p, encoding="utf-8", errors="ignore").read()
            except OSError:
                pass
    facts = extract_facts(files)
    if not facts:
        raise SystemExit(f"no unique facts found in {args.repo}")
    print(f"candidate facts: {len(facts)}; running pi agent in {work}")

    transcript = run_agent(work)
    print(f"transcript: {len(transcript):,} chars")

    # harvest facts the agent actually observed (≥1 occurrence; the value is UNIQUE in
    # the repo render so no same-value distractor can exist — occurrences beyond the
    # first are the agent correctly RESTATING it in its summary, which is the behavior
    # we want to reward; count recorded in meta)
    observed = [(label, value, path, transcript.count(value))
                for (label, value, path) in facts if value in transcript]
    if not observed:
        raise SystemExit("agent observed no candidate facts — discard/regenerate")
    print(f"observed facts: {len(observed)} (occurrences: "
          f"{[c for *_ , c in observed]})")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        f.write(json.dumps({"body": "", "ledger": None, "target_tokens": None}) + "\n")
        for label, value, path, count in observed[:args.max_samples]:
            q = (f"Session transcript follows.\n\n{transcript}\n\nQuestion: according "
                 f"to the session above, what is the project's {label}? Reply with the "
                 f"value only.")
            f.write(json.dumps({"prompt": q, "answer": value,
                                "question": f"{label}?", "axis": "agentmem",
                                "fact_source": path, "occurrences": count}) + "\n")
    n = min(len(observed), args.max_samples)
    print(f"wrote {1 + n} lines -> {args.out} (validate with serialize.py)")
    if args.validate:
        for label, value, _p, count in observed[:args.max_samples]:
            assert count >= 1
        print("agent_traj validation OK (harvested facts observed in transcript)")


if __name__ == "__main__":
    main()
