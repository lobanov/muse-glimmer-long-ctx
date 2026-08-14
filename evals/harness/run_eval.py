#!/usr/bin/env python3
"""PLAN §2 evaluation runner.

Grid: task × context-length × evidence-depth × seed-rep, on any OpenAI-compatible engine.
Writes resumable JSONL (one row per request) and (optionally, if pyarrow is available)
converts to the common Parquet schema at the end of the run.

Usage (inside dev container):
  python3 evals/harness/run_eval.py \
      --engine vllm --base-url http://localhost:8000/v1 \
      --config-label stock --tasks niah,abstain \
      --ctx 32000,64000,128000 --depths 0,0.1,0.5,0.9,1.0 --reps 3 \
      --mode capability --out outputs/eval/stock_vllm.jsonl --write-parquet
"""
import argparse
import datetime as dt
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import client  # noqa: E402
import tasks as T  # noqa: E402

PLAN_CTX = [32_000, 64_000, 128_000, 192_000, 256_000, 384_000, 512_000]
PLAN_DEPTHS = [0.0, 0.10, 0.25, 0.50, 0.75, 0.90, 1.00]

COLS = ["run_id", "ts", "engine", "model", "config_label", "mode", "task",
        "target_ctx", "depth", "rep", "cell_id",
        "sampling", "prompt_tokens", "completion_tokens", "finish_reason",
        "wall_s", "ttft_s", "tok_per_s", "score", "detail",
        "expected", "response_head", "reasoning_head", "reasoning", "error"]


def make_row(**kw):
    row = {c: None for c in COLS}
    row.update(kw)
    return row


def load_done(out):
    done = set()
    if os.path.exists(out):
        with open(out) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    if not r.get("error"):
                        done.add(r["cell_id"])
                except Exception:
                    pass
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True, help="label, e.g. vllm | llamacpp")
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", default="muse-glimmer")
    ap.add_argument("--config-label", default="stock",
                    help="model-config arm: stock / qk4.1 / yarn4 / qlora-r32 ...")
    ap.add_argument("--tasks", default=",".join(T.TASKS))
    ap.add_argument("--ctx", default=",".join(map(str, PLAN_CTX)))
    ap.add_argument("--depths", default=",".join(map(str, PLAN_DEPTHS)))
    ap.add_argument("--reps", type=int, default=3,
                    help="independent instance resamples per cell (>=3 at extreme lengths)")
    ap.add_argument("--mode", choices=["capability", "parity"], default="capability")
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--timeout", type=float, default=7200)
    ap.add_argument("--plugin", default="",
                    help="comma list of task plugins in evals/harness/ (e.g. nolima)")
    ap.add_argument("--out", required=True, help="JSONL output path (host-visible)")
    ap.add_argument("--write-parquet", action="store_true")
    ap.add_argument("--force", action="store_true", help="re-run cells already present")
    args = ap.parse_args()

    task_list = [t.strip() for t in args.tasks.split(",") if t.strip()]
    ctx_list = [int(x) for x in args.ctx.split(",")]
    depth_list = [float(x) for x in args.depths.split(",")]
    run_id = f"{args.config_label}__{args.engine}__{dt.datetime.now().strftime('%Y%m%dT%H%M%S')}"
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    done = set() if args.force else load_done(args.out)
    import random
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    for plug in [p.strip() for p in args.plugin.split(",") if p.strip()]:
        mod = __import__(plug)
        mod.register(T)
    total = len(task_list) * len(ctx_list) * len(depth_list) * args.reps
    i = 0
    for task in task_list:
        if task not in T.BUILDERS:
            sys.exit(f"unknown task {task!r}; known: {T.TASKS}")
        for ctx in ctx_list:
            for depth in depth_list:
                for rep in range(args.reps):
                    i += 1
                    cell_id = f"{args.config_label}|{args.engine}|{args.mode}|{task}|{ctx}|{depth}|{rep}"
                    if cell_id in done:
                        continue
                    seed = T.cell_seed(task, ctx, depth, rep)
                    rng = random.Random(seed)
                    prompt, meta = T.BUILDERS[task](rng, ctx, depth)
                    row = make_row(
                        run_id=run_id, ts=dt.datetime.now().isoformat(timespec="seconds"),
                        engine=args.engine, model=args.model,
                        config_label=args.config_label, mode=args.mode, task=task,
                        target_ctx=ctx, depth=depth, rep=rep, cell_id=cell_id,
                        expected=meta)
                    try:
                        res = client.chat(
                            args.base_url, args.model,
                            [{"role": "user", "content": prompt}],
                            mode=args.mode, max_tokens=args.max_tokens,
                            seed=seed, timeout=args.timeout)
                        s, detail = T.score(task, res["content"], meta)
                        ct = res.get("completion_tokens")
                        row.update(
                            sampling=res["sampling"],
                            prompt_tokens=res.get("prompt_tokens"),
                            completion_tokens=ct,
                            finish_reason=res.get("finish_reason"),
                            wall_s=res["wall_s"], ttft_s=res["ttft_s"],
                            tok_per_s=round(ct / res["wall_s"], 2) if ct and res["wall_s"] else None,
                            score=s, detail=detail,
                            response_head=(res["content"] or "")[:160],
                            reasoning_head=res.get("reasoning_head"),
                            reasoning=res.get("reasoning"))
                    except Exception as e:  # noqa: BLE001 — record, keep going
                        row.update(error=str(e)[:300])
                    with open(args.out, "a") as f:
                        f.write(json.dumps(row) + "\n")
                    print(f"[{i}/{total}] {cell_id} score={row['score']} "
                          f"detail={row['detail']} ptok={row['prompt_tokens']} "
                          f"ttft={row['ttft_s']}s wall={row['wall_s']}s"
                          + (f" ERR={row['error']}" if row["error"] else ""), flush=True)

    if args.write_parquet:
        out_parquet = os.path.splitext(args.out)[0] + ".parquet"
        import to_parquet  # noqa: E402 (same dir)
        to_parquet.convert([args.out], out_parquet)
        print(f"[parquet] {out_parquet}")


if __name__ == "__main__":
    main()
