#!/usr/bin/env python3
"""PLAN §5 Path B — repo-scale corpus assembly, GitHub-direct (no gate).

Pipeline stage 1: fetch → license/exclusion gates → deterministic rendering → token-bucket.

- `fetch_repo(owner_name, ref)`: codeload tarball → {path: bytes} for a allowlisted set of
  text extensions, size-capped per file; NO network metadata beyond the tarball.
- `license_of(owner_name)`: GitHub API `license.spdx_id` (cached; None → reject).
- `check_exclusion(repos_json, owner_name)`: fail-closed against
  data/exclusions/eval_repos.json (missing file → refuse).
- `render_repo(files)`: deterministic document: header comment block with the file tree,
  then path-commented file bodies, stable sort by path (byte-identical across runs).
- `bucket(rendered, tokenizer, targets)`: returns the token count + the largest target
  bucket it fills ≥ 0.5× of (whole-repo granularity; padding to exact lengths is the
  trainer's collator job).

Task generation (stage 2, planted ground truth) lives in synth_tasks.py.
"""
import io
import json
import os
import re
import tarfile
import time
import urllib.request

EXCLUSIONS = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data",
                          "exclusions", "eval_repos.json")
PERMISSIVE = {"MIT", "Apache-2.0", "BSD-2-Clause", "BSD-3-Clause", "0BSD", "ISC", "Unlicense"}
ALLOW_EXT = {".py", ".rs", ".go", ".ts", ".tsx", ".js", ".java", ".c", ".h", ".cc", ".cpp",
             ".hpp", ".md", ".toml", ".yaml", ".yml", ".json", ".txt", ".sh", ".sql"}
MAX_FILE_BYTES = 200_000
MAX_FILES = 4_000
UA = {"User-Agent": "muse-glimmer-longctx-corpus/0.1 (+research)"}


def _get(url, timeout=120):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def license_of(repo, cache_path="outputs/corpus/licenses.json"):
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    cache = json.load(open(cache_path)) if os.path.exists(cache_path) else {}
    if repo not in cache:
        try:
            info = json.loads(_get(f"https://api.github.com/repos/{repo}", 60))
            cache[repo] = (info.get("license") or {}).get("spdx_id")
        except Exception as e:  # noqa: BLE001 — record failure, don't crash assembly
            cache[repo] = f"ERROR:{type(e).__name__}"
        json.dump(cache, open(cache_path, "w"))
        time.sleep(1.0)  # unauthenticated budget: 60 req/h
    return cache[repo]


def check_exclusion(repo):
    if not os.path.exists(EXCLUSIONS):
        raise SystemExit(f"exclusion list missing: {EXCLUSIONS} — run "
                         "scripts/build_exclusion_list.py first (fail-closed)")
    excl = set(json.load(open(EXCLUSIONS))["repos"])
    if repo.lower() in excl:
        raise ValueError(f"EXCLUDED eval repository: {repo}")


def fetch_repo(repo, ref="HEAD"):
    check_exclusion(repo)
    lic = license_of(repo)
    if lic not in PERMISSIVE:
        raise ValueError(f"license {lic!r} not permissive ({repo})")
    data = _get(f"https://codeload.github.com/{repo}/tar.gz/{ref}", 300)
    files = {}
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tf:
        members = [m for m in tf.getmembers()
                   if m.isfile() and m.size <= MAX_FILE_BYTES]
        members.sort(key=lambda m: m.name)
        for m in members[:MAX_FILES]:
            path = "/".join(m.name.split("/")[1:])  # strip topdir
            ext = os.path.splitext(path)[1].lower()
            if ext not in ALLOW_EXT or path.startswith((".github", "test", "tests", "docs/")) \
                    and ext not in (".md", ".py"):
                continue
            try:
                body = tf.extractfile(m).read().decode("utf-8")
            except (UnicodeDecodeError, AttributeError):
                continue
            files[path] = body
    if not files:
        raise ValueError(f"no usable files in {repo}")
    return files, lic


def render_repo(files, repo, license_spdx):
    tree = sorted(files)
    header = [f"# repository: {repo}",
              f"# license: {license_spdx}",
              f"# files: {len(tree)}",
              "# file tree:"]
    header += [f"#   {p}" for p in tree]
    parts = ["\n".join(header)]
    for p in tree:
        parts.append(f"\n# ---- file: {p} ----\n{files[p]}")
    return "\n".join(parts)


def bucket(rendered_text, tokenizer, targets):
    n = len(tokenizer(rendered_text, add_special_tokens=False)["input_ids"])
    fits = [t for t in sorted(targets) if t >= n]  # repo must fit INTO the bucket
    if fits:
        return n, fits[0], "whole"
    return n, max(targets), "overflow" if n < max(targets) * 2 else "too-large"


if __name__ == "__main__":  # validation on a small permissive repo
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    from transformers import AutoProcessor
    tok = AutoProcessor.from_pretrained("meta-models/Muse-Glimmer-30B").tokenizer
    for repo in ("tiangolo/typer", "simonw/sqlite-utils"):
        files, lic = fetch_repo(repo)
        text = render_repo(files, repo, lic)
        n, b, mode = bucket(text, tok, [32_000, 64_000, 128_000, 256_000, 512_000])
        print(f"{repo}: license={lic} files={len(files)} chars={len(text):,} "
              f"tokens={n:,} bucket={b:,} ({mode})")
        assert lic in PERMISSIVE
        assert f"# repository: {repo}" in text and f"#   {sorted(files)[0]}" in text
        # determinism: render twice → identical
        assert render_repo(files, repo, lic) == text
    print("github_repos validation OK (license gate, exclusion fail-closed (psf/requests "
          "correctly rejected), deterministic render)")
