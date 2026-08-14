#!/usr/bin/env python3
"""PLAN §0.2 parity spike: BF16 (vLLM) vs K-Quant GGUF (llama.cpp) needle retrieval.

Quant-noise floor measurement — identical prompts, greedy decoding (temp 0), no
chat_template_kwargs (identical template rendering on both engines). Generous max_tokens
because Glimmer is a reasoning model (AGENTS.md rule 5).

Cells: {32k, 64k, 128k} target ctx x {10%, 50%, 90%} needle depth x 3 reps = 27 requests.
Output: JSONL (append; resumable — already-present cells are skipped). Convert to Parquet
in the dev container afterwards (host python has no pyarrow).

Usage:
  python3 evals/parity_spike.py --engine vllm     --base-url http://localhost:8000/v1  --out outputs/parity_spike/vllm.jsonl
  python3 evals/parity_spike.py --engine llamacpp --base-url http://localhost:8081/v1 --out outputs/parity_spike/llamacpp.jsonl
"""
import argparse, json, random, time, urllib.request, urllib.error, os

MODEL = "muse-glimmer"  # served-model-name on both engines
CTX_TARGETS = [32_000, 64_000, 128_000]
DEPTHS = [0.10, 0.50, 0.90]
REPS = 3
MAX_TOKENS = 1024
CHARS_PER_TOKEN = 6.0          # measured for the log-entry haystack (23,859 tok / 144k chars)
FILL_FACTOR = 0.90             # headroom for template + generation (128k cell must fit 131072)

COLORS = ["amber", "teal", "violet", "crimson", "sapphire", "ochre"]
NAMES = ["aurora", "meridian", "kestrel", "cobalt", "harbor", "zenith"]

PARA = ("Log entry {n}: routine diagnostics completed across the western array; telemetry "
        "buffers rotated; two firmware patches queued for review; the night-shift handover "
        "mentions minor variance in coolant pressure but nothing outside tolerance bands.")


def make_case(rng, target_tokens, depth):
    color, name = rng.choice(COLORS), rng.choice(NAMES)
    code = f"{name}-{rng.randint(1000, 9999)}-{color}"
    needle = f" The {color} access code for the {name} project is {code}. "
    aim_chars = int(target_tokens * FILL_FACTOR * CHARS_PER_TOKEN)
    body, n = "", 0
    while len(body) < aim_chars:
        n += 1
        body += PARA.format(n=n)
    cut = int(len(body) * depth)
    prompt = (f"Context follows.\n\n{body[:cut]}{needle}{body[cut:]}\n\n"
              f"Question: What is the {color} access code for the {name} project? "
              f"Reply with the code only.")
    return prompt, code, color, name


def call(base_url, prompt):
    req = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,      # greedy: parity check (PLAN §2 rule)
        "max_tokens": MAX_TOKENS,
    }).encode()
    t0 = time.time()
    r = json.load(urllib.request.urlopen(urllib.request.Request(
        f"{base_url}/chat/completions", req,
        {"Content-Type": "application/json"}), timeout=1200))
    wall = time.time() - t0
    msg = r["choices"][0]["message"]
    content = (msg.get("content") or "")
    return {
        "prompt_tokens": r.get("usage", {}).get("prompt_tokens"),
        "completion_tokens": r.get("usage", {}).get("completion_tokens"),
        "finish_reason": r["choices"][0].get("finish_reason"),
        "content": content,
        "wall_s": round(wall, 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True, choices=["vllm", "llamacpp"])
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    done = set()
    if os.path.exists(args.out):
        with open(args.out) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    done.add((r["target_ctx"], r["depth"], r["rep"]))
                except Exception:
                    pass

    for target in CTX_TARGETS:
        for depth in DEPTHS:
            for rep in range(REPS):
                if (target, depth, rep) in done:
                    continue
                rng = random.Random(1000 * target + int(depth * 100) * 10 + rep)
                prompt, code, color, name = make_case(rng, target, depth)
                last_err = None
                for attempt in (1, 2):
                    try:
                        res = call(args.base_url, prompt)
                        last_err = None
                        break
                    except Exception as e:
                        last_err = str(e)[:200]
                        time.sleep(5)
                row = {"engine": args.engine, "target_ctx": target,
                       "depth": depth, "rep": rep, "code": code,
                       "needle_desc": f"{color}/{name}"}
                if last_err:
                    row.update(error=last_err, found=False)
                else:
                    row.update(res, found=code in res["content"],
                               content_head=res["content"][:80])
                    row.pop("content", None)
                with open(args.out, "a") as f:
                    f.write(json.dumps(row) + "\n")
                f_tok = row.get("prompt_tokens")
                print(f"[{args.engine}] ctx={target} depth={depth} rep={rep} "
                      f"found={row['found']} ptok={f_tok} wall={row.get('wall_s')}s"
                      + (f" ERR={last_err}" if last_err else ""), flush=True)


if __name__ == "__main__":
    main()
