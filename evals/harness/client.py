#!/usr/bin/env python3
"""OpenAI-compatible streaming chat client with the PLAN §2 contract baked in.

Contract (docs/environment.md § "Eval-harness client contract"):
- always sends chat_template_kwargs.reasoning_strength (default "low")
- capability mode: temp 1.0 / top-p 0.95 / top-k 64 (Meta defaults)
- parity mode: greedy (temp 0) — deterministic comparisons only
- captures TTFT (prompt-ingestion wall-clock proxy via streaming), usage, finish_reason,
  content in `content`, thinking in `reasoning_content` (vLLM) or `reasoning`.
"""
import json
import time
import urllib.request


class ChatError(Exception):
    pass


def _extract_reasoning(delta):
    r = delta.get("reasoning_content") or delta.get("reasoning")
    return r or ""


def chat(base_url, model, messages, *, mode="capability", max_tokens=4096,
         seed=None, timeout=7200, retries=3):
    """One chat completion. Returns dict; raises ChatError after `retries` failures."""
    if mode == "capability":
        smp = dict(temperature=1.0, top_p=0.95, top_k=64)
    elif mode == "parity":
        smp = dict(temperature=0.0, top_p=1.0, top_k=1)
    else:
        raise ValueError(f"unknown mode {mode!r}")
    body = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": smp["temperature"],
        "top_p": smp["top_p"],
        "top_k": smp["top_k"],
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"reasoning_strength": "low"},
    }
    if seed is not None:
        body["seed"] = seed
    url = base_url.rstrip("/") + "/chat/completions"

    last_err = None
    for attempt in range(1, retries + 1):
        t0 = time.time()
        ttft = None
        parts, rparts = [], []
        usage, finish = None, None
        try:
            req = urllib.request.Request(
                url, json.dumps(body).encode(), {"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                for raw in resp:
                    line = raw.decode("utf-8", "replace").strip()
                    if not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if payload == "[DONE]":
                        break
                    try:
                        chunk = json.loads(payload)
                    except json.JSONDecodeError:
                        continue
                    if chunk.get("usage"):
                        usage = chunk["usage"]
                    for ch in chunk.get("choices", []):
                        delta = ch.get("delta") or {}
                        c = delta.get("content")
                        if c:
                            if ttft is None:
                                ttft = time.time() - t0
                            parts.append(c)
                        r = _extract_reasoning(delta)
                        if r:
                            rparts.append(r)
                        if ch.get("finish_reason"):
                            finish = ch["finish_reason"]
            wall = time.time() - t0
            content = "".join(parts)
            return {
                "content": content,
                "reasoning_head": "".join(rparts)[:200],
                "prompt_tokens": (usage or {}).get("prompt_tokens"),
                "completion_tokens": (usage or {}).get("completion_tokens"),
                "finish_reason": finish,
                "wall_s": round(wall, 2),
                "ttft_s": round(ttft, 2) if ttft is not None else None,
                "sampling": {**smp, "reasoning_strength": "low",
                             "max_tokens": max_tokens, "seed": seed, "mode": mode},
            }
        except Exception as e:  # noqa: BLE001 — retry any transport/HTTP error
            last_err = e
            time.sleep(min(60, 10 * attempt))
    raise ChatError(f"{url}: {last_err}")
