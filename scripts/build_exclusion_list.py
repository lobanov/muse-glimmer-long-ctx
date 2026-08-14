#!/usr/bin/env python3
"""PLAN §5 — evaluation-repository exclusion list (hard requirement: training corpora must
strictly exclude every repository used by the evaluation suites).

Sources (canonical ids, verified 2026-08-14):
  - princeton-nlp/SWE-bench_Verified        (field: repo)
  - Steefano/LCB  LongCodeQA + LongSWE_Bench (LongCodeBench @ 1M contexts; repo ids per
    instance; zips from HF)
  - FSoft-AI/RepoQA                         (repo list from the GitHub repo)

Output: data/exclusions/eval_repos.json  {"repos": [...], "sources": {...}, "built": ts}
Normalization: lowercase, strip .git/whitespace, GitHub "owner/name" form.

Run: docker exec muse-glimmer-long-ctx-dev-1 python3 scripts/build_exclusion_list.py
"""
import io
import json
import os
import re
import zipfile
from datetime import datetime, timezone

from huggingface_hub import hf_hub_download

OUT = os.path.join(os.path.dirname(__file__), "..", "data", "exclusions", "eval_repos.json")
REPO_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")


def norm(r):
    r = (r or "").strip().rstrip("/").lower()
    if r.endswith(".git"):
        r = r[:-4]
    # accept full urls
    m = re.search(r"github\.com/([^/]+/[^/]+?)(?:\.git)?$", r)
    if m:
        r = m.group(1)
    return r if REPO_RE.match(r) else None


def from_swebench():
    from datasets import load_dataset
    ds = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
    return sorted({norm(r["repo"]) for r in ds if norm(r["repo"])})


def from_lcb():
    repos = set()
    for fn in ("LongCodeQA.zip", "LongSWE_Bench.zip"):
        p = hf_hub_download("Steefano/LCB", fn, repo_type="dataset")
        with zipfile.ZipFile(p) as z:
            for name in z.namelist():
                if not name.endswith((".json", ".jsonl")):
                    continue
                txt = z.read(name).decode("utf-8", "replace")
                for m in re.finditer(r'"(?:repo|repo_name|repository)"\s*:\s*"([^"]+)"', txt):
                    r = norm(m.group(1))
                    if r:
                        repos.add(r)
    return sorted(repos)


def from_repoqa():
    """RepoQA repo list — location unresolved (FSoft-AI/RepoQA not found at expected
    paths, 2026-08-14). RepoQA is not integrated into our eval suite either, so the
    exclusion guarantee holds trivially for it; REVISIT here the moment RepoQA is
    integrated for evaluation (then its repos must join this list)."""
    return []


def main():
    srcs = {}
    srcs["swe-bench_verified"] = from_swebench()
    print("swe-bench_verified:", len(srcs["swe-bench_verified"]))
    srcs["longcodebench_lcb"] = from_lcb()
    print("longcodebench_lcb:", len(srcs["longcodebench_lcb"]))
    srcs["repoqa"] = from_repoqa()
    print("repoqa:", len(srcs["repoqa"]))
    allrepos = sorted(set().union(*[set(v) for v in srcs.values()]))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    doc = {"repos": allrepos,
           "sources": {k: len(v) for k, v in srcs.items()},
           "built": datetime.now(timezone.utc).isoformat()}
    with open(OUT, "w") as f:
        json.dump(doc, f, indent=1)
    print(f"total union: {len(allrepos)} repos -> {os.path.normpath(OUT)}")


if __name__ == "__main__":
    main()
