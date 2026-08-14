#!/usr/bin/env python3
"""Long-context perplexity probe (PLAN §3 "PPL where useful"; §4 arm comparison; §12 KV
quantization quality checks).

Method: vLLM /v1/completions with echo=true + logprobs=0 → per-token prompt logprobs →
PPL = exp(mean NLL) over the LAST `eval_span` tokens (early tokens excluded — they carry
near-zero entropy and mask long-range effects). Text: slices of the InfiniteBench/NoLiMa
corpora (real prose/code mix) at controlled token targets.

Notes:
- Uses the served model AS-IS: same arm/sidecar the eval grid runs against (VLLM_MODEL).
- `final_logit_softcapping` is part of forward → logprobs are self-consistent; compare
  within-model/config, not against external PPL implementations.
- Contexts >131k require the stock-524k arm (mechanical window extension) — that is the
  §3 "beyond nominal" measurement itself, so record `config_label`.

Usage (dev container, vLLM serving):
  python3 evals/ppl_probe.py --base-url http://vllm:8000/v1 --config-label stock-524k \
      --ctx 32000,131072,262144,393216,524288 --out outputs/eval/ppl_stock.jsonl
"""
import argparse
import json
import math
import os
import random
import time
import urllib.request

CORPORA = ["infbench", "nolima"]  # real books + shuffled prose (license: research use)


def get_corpus(name):
    if name == "nolima":
        from huggingface_hub import hf_hub_download
        parts = []
        for i in range(1, 6):
            p = hf_hub_download("amodaresi/NoLiMa",
                                f"haystack/rand_shuffle/rand_book_{i}.txt", repo_type="dataset")
            parts.append(open(p).read())
        return "\n".join(parts), 4.16
    if name == "infbench":
        from huggingface_hub import hf_hub_download
        import json as J
        rows = [J.loads(l) for l in open(hf_hub_download(
            "xinrongzhang2022/InfiniteBench", "longbook_qa_eng.jsonl",
            repo_type="dataset")) if l.strip()]
        return "\n\n".join(r["context"] for r in rows[:12]), 4.1
    raise ValueError(name)


def probe(base_url, model, prompt, max_tokens):
    body = {"model": model, "prompt": prompt, "max_tokens": 1, "echo": True,
            "logprobs": 0, "temperature": 0.0}
    req = urllib.request.Request(base_url.rstrip("/") + "/completions",
                                 json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t0 = time.time()
    r = json.load(urllib.request.urlopen(req, timeout=7200))
    return r, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", default="muse-glimmer")
    ap.add_argument("--config-label", default="stock")
    ap.add_argument("--ctx", default="32000,131072,262144,393216,524288")
    ap.add_argument("--corpus", default="nolima")
    ap.add_argument("--eval-span", type=int, default=8192,
                    help="PPL computed over the last N tokens")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    corpus, cpt = get_corpus(args.corpus)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    for target in [int(x) for x in args.ctx.split(",")]:
        for rep in range(args.reps):
            key = f"{args.config_label}|{args.corpus}|{target}|{rep}"
            done = set()
            if os.path.exists(args.out):
                for line in open(args.out):
                    try:
                        done.add(json.loads(line)["key"])
                    except Exception:
                        pass
            if key in done:
                continue
            rng = random.Random(1000 * target + rep)
            aim = int(target * cpt)
            start = rng.randrange(0, max(1, len(corpus) - aim - 1))
            text = corpus[start:start + aim]
            try:
                r, wall = probe(args.base_url, args.model, text, 1)
                toks = r["usage"]["prompt_tokens"]
                content = r["choices"][0]
                lps = content.get("logprobs", {}).get("tokens")
                tok_logprobs = content.get("logprobs", {}).get("token_logprobs") or []
                n_eval = min(args.eval_span, len(tok_logprobs) - 1)
                vals = [lp for lp in tok_logprobs[-n_eval:] if lp is not None]
                nll = -sum(vals) / len(vals) if vals else float("nan")
                row = {"key": key, "config": args.config_label, "corpus": args.corpus,
                       "target_ctx": target, "rep": rep, "prompt_tokens": toks,
                       "n_eval": len(vals), "ppl": math.exp(nll) if vals else None,
                       "wall_s": round(wall, 1)}
            except Exception as e:  # noqa: BLE001
                row = {"key": key, "config": args.config_label, "corpus": args.corpus,
                       "target_ctx": target, "rep": rep, "error": str(e)[:200]}
            with open(args.out, "a") as f:
                f.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
