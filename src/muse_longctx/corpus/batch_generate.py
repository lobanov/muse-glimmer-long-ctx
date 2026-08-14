#!/usr/bin/env python3
"""PLAN §5 — batch corpus generation driver (host-side, pi-teacher; no GPU needed).

Generates a real corpus slice across all five components, resumable (skip-if-exists per
artifact), failures logged and skipped (never fatal). Run detached:

    nohup python3 src/muse_longctx/corpus/batch_generate.py > logs/corpus-batch.log 2>&1 &

Inventory targets (tunable here): 24 repos · 8 synth docs · 6 book slices · 3 agent
sessions · 20 short items. Everything goes through the validated component generators
(license gate, exclusion fail-closed, machine-verified answers).
"""
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
CORPUS = os.path.join(ROOT, "outputs", "corpus")

REPOS = [  # popular permissive-licensed; license+exclusion gates re-check each anyway
    "pallets/flask", "pallets/click", "pallets/jinja", "encode/httpx", "encode/starlette",
    "textualize/rich", "google/python-fire", "oauthlib/oauthlib", "carltongibson/django-filter",
    "tokio-rs/tokio", "serde-rs/serde", "BurntSushi/ripgrep", "sharkdp/fd", "sharkdp/bat",
    "junegunn/fzf", "gin-gonic/gin", "urfave/cli", "date-fns/date-fns", "lodash/lodash",
    "miekg/dns", "golang/mock", "uber-go/zap", "rs/curl_cffi", "simonw/sqlite-utils",
    "tiangolo/typer",
]
BOOKS = [1342, 84, 2701, 1661, 98, 1400]
SYNTH = [(65536, 3), (98304, 3), (131072, 2)]      # (doc_tokens, n_docs) — genuine backbone
BOOK_SLICES = [(65536, 2), (131072, 2), (262144, 2)]  # (tokens, n_slices) across books
AGENT_REPOS = ["tiangolo/typer", "simonw/sqlite-utils", "pallets/click",
               "encode/starlette", "textualize/rich", "miekg/dns", "golang/mock",
               "uber-go/zap", "burntsushi/ripgrep"]


def sh(args, timeout=3600):
    r = subprocess.run([sys.executable] + args, capture_output=True, text=True,
                       timeout=timeout, cwd=ROOT)
    ok = r.returncode == 0
    return ok, (r.stdout + r.stderr).strip().splitlines()[-1] if (r.stdout + r.stderr).strip() else ""


DEV = "muse-glimmer-long-ctx-dev-1"
DEVPATH = "/workspaces/muse-glimmer-long-ctx"   # repo root inside dev (tokenizer+5.15 there)


def sh_dev(args, timeout=3600):
    """Run a repo script INSIDE the dev container (host transformers is too old for
    Glimmer serialization; host↔dev paths translated)."""
    dev_args = [a.replace(ROOT, DEVPATH) for a in args]
    cmd = ["docker", "exec", DEV, "python3"] + dev_args
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=ROOT)
    ok = r.returncode == 0
    return ok, (r.stdout + r.stderr).strip().splitlines()[-1] if (r.stdout + r.stderr).strip() else ""


def batch(name, items, runner, log):
    done = 0
    for i, item in enumerate(items, 1):
        out = item["out"]
        if os.path.exists(out):
            done += 1
            continue
        t0 = time.time()
        ok, tail = runner(item)
        log(f"[{name} {i}/{len(items)}] {'OK ' if ok else 'SKIP'} {item['label']} "
            f"({time.time()-t0:.0f}s) {tail[:120]}")
        done += ok
    log(f"[{name}] complete: {done}/{len(items)} artifacts present")


def main():
    logf = open(os.path.join(ROOT, "logs", "corpus-batch.log"), "a")

    def log(msg):
        logf.write(f"[{time.strftime('%F %T')}] {msg}\n")
        logf.flush()

    log("=== batch_generate start ===")

    # 1. repo documents
    items = [{"label": r, "repo": r,
              "out": os.path.join(CORPUS, "repos_v1", r.replace("/", "__") + ".jsonl")}
             for r in REPOS]
    batch("repos", items, lambda it: sh([os.path.join(HERE, "repos_doc.py"),
                                         "--repo", it["repo"], "--out", it["out"],
                                         "--validate"]), log)

    # 2. synthetic long docs (the genuine-length backbone)
    items = []
    for tokens, n in SYNTH:
        for k in range(n):
            items.append({"label": f"synth{tokens//1024}k#{k}",
                          "out": os.path.join(CORPUS, "synth_v1", f"synth_{tokens}_{k}.jsonl"),
                          "tokens": tokens, "seed": 1000 + k})
    batch("synth", items, lambda it: sh([os.path.join(HERE, "synth_docs.py"),
                                         "--doc-tokens", str(it["tokens"]),
                                         "--seed", str(it["seed"]),
                                         "--out", it["out"]]), log)

    # 3. natural books
    items = []
    for tokens, n in BOOK_SLICES:
        for k in range(n):
            book = BOOKS[(k + tokens) % len(BOOKS)]
            items.append({"label": f"book{book}-{tokens//1024}k#{k}", "book": book,
                          "tokens": tokens, "seed": 500 + k,
                          "out": os.path.join(CORPUS, "nat_v1",
                                              f"book{book}_{tokens}_{k}.jsonl")})
    batch("natural", items, lambda it: sh([os.path.join(HERE, "natural_docs.py"),
                                           "--book", str(it["book"]),
                                           "--slice-tokens", str(it["tokens"]),
                                           "--seed", str(it["seed"]),
                                           "--out", it["out"], "--validate"]), log)

    # 4. agent trajectories
    items = [{"label": f"agent-{r}", "repo": r,
              "out": os.path.join(CORPUS, "agent_v1", r.replace("/", "__") + ".jsonl")}
             for r in AGENT_REPOS]
    batch("agent", items, lambda it: sh([os.path.join(HERE, "agent_traj.py"),
                                         "--repo", it["repo"], "--out", it["out"],
                                         "--validate"]), log)

    # 5. short replay
    for k in range(4):
        out = os.path.join(CORPUS, "short_v1", f"s{k}.jsonl")
        if not os.path.exists(out):
            ok, tail = sh([os.path.join(HERE, "short_replay.py"), "--n", "6",
                           "--seed", str(700 + k), "--out", out, "--validate"])
            log(f"[short {k+1}/4] {'OK' if ok else 'SKIP'} {tail[:100]}")

    # 6. serialize everything new (in dev: tokenizer + transformers 5.15 live there),
    #    then remix (host: stdlib only)
    for comp in ("repos_v1", "synth_v1", "nat_v1", "agent_v1", "short_v1"):
        d = os.path.join(CORPUS, comp)
        for fn in os.listdir(d) if os.path.isdir(d) else []:
            if fn.endswith(".jsonl") and not fn.endswith(".samples.jsonl"):
                src, dst = os.path.join(d, fn), os.path.join(d, fn[:-6] + ".samples.jsonl")
                if not os.path.exists(dst):
                    ok, tail = sh_dev([os.path.join(DEVPATH, "src/muse_longctx/corpus/serialize.py"),
                                       "--doc", src, "--out", dst])
                    log(f"[serialize {fn}] {'OK' if ok else 'FAIL'} {tail[:100]}")
    ok, tail = sh([os.path.join(HERE, "build_corpus.py"), "--name", "train_v1"])
    log(f"[mix] {'OK' if ok else 'FAIL'} {tail[:200]}")
    log("=== batch_generate complete — run build_corpus again anytime for an updated mix ===")


if __name__ == "__main__":
    main()
