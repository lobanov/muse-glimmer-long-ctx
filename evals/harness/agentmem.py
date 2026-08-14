#!/usr/bin/env python3
"""PLAN §2 / GOAL.md — custom agentic-memory benchmark (`agentmem`).

Custom suite item: "persistence of relevant information across long tool-use trajectories".

Design: a simulated coding-agent session transcript (tool calls + outputs, realistic
rendering) in a single prompt. Critical facts (endpoint host/port, API key prefix,
feature-flag name, DB table) appear in EARLY tool outputs at depth-controlled positions;
mid-trajectory instructions reference them implicitly ("use the staging endpoint you
found earlier"); the final user turn asks to apply them (combine host:port, key prefix,
etc.). Every later tool output contains NEAR-MISS DISTRACTOR values (same shape, different
value — other hosts, ports, keys) so lexical matching without trajectory memory fails.

Scoring: exact planted values (partial credit per component). Fully synthetic — no
benchmark contamination possible; safe by construction for training-side reuse too.

Register: --plugin agentmem    Use: --tasks agentmem --ctx 32000,... --depths 0,0.1,0.5,0.9
"""
import random

COLORS = ["amber", "teal", "violet", "crimson", "sapphire"]
NAMES = ["kestrel", "harbor", "zenith", "palisade", "lantern"]
TOOLS = ["read", "grep", "run_tests", "curl", "db_query"]
FILES = ["config/staging.yaml", "services/api/main.py", "deploy/values.yaml",
         "tests/test_gateway.py", "docs/runbook.md", "src/lib/store.py",
         "scripts/migrate.sh", "infra/terraform/vars.tf"]

TOOLS_TMPL = [
    ("read", "read {file}\n```\n{body}\n```"),
    ("grep", "grep -rn \"{pat}\" src/\n```\n{body}\n```"),
    ("curl", "curl -s http://{host}:{port}/healthz\n```\n{{\"status\": \"{body}\"}}\n```"),
    ("db_query", 'db> SELECT * FROM "{table}" LIMIT 3;\n```\n{body}\n```'),
    ("run_tests", "pytest -q {file}\n```\n... {body} ... 12 passed\n```"),
]

FILLER_BODY = ("import logging\nlogger = logging.getLogger(__name__)\n"
               "# module: {mod}.handlers\n"
               "def handle_{fn}(event):\n"
               "    ctx = event.get(\"context\", {{}})\n"
               "    if ctx.get(\"retry\"):\n"
               "        return _retry(event, attempts=ctx.get(\"attempts\", 1))\n"
               "    logger.info(\"{fn} processed mid=%s\", ctx.get(\"mid\"))\n"
               "    return {{\"ok\": True, \"stage\": \"{mod}\"}}\n")


def _values(rng, name):
    """Planted key facts + confusable distractors of identical shape."""
    color, other = rng.sample(COLORS, 2)
    host = f"{name}-staging.internal"
    port = rng.randint(9100, 9699)
    key = f"{name[:3].upper()}-{rng.randint(10000, 99999)}"
    flag = f"enable_{name}_{color}"
    return {
        "host": host, "port": port, "key": key, "flag": flag,
        "d_host": f"{other}-canary.internal",
        "d_port": port + rng.choice([-111, 111, -7, 7]),
        "d_key": f"{other[:3].upper()}-{rng.randint(10000, 99999)}",
        "d_flag": f"enable_{other}_{color}",
    }


def _fact_output(name, v, kind):
    if kind == "config":
        body = (f"# config/staging.yaml (excerpt)\nservice: {name}\n"
                f"endpoint:\n  host: {v['host']}\n  port: {v['port']}\n"
                f"auth:\n  key_prefix: {v['key']}\nfeatures:\n  {v['flag']}: true")
    elif kind == "runbook":
        body = (f"# docs/runbook.md (excerpt)\nFor the {name} staging environment use\n"
                f"host {v['host']} port {v['port']}; the API key begins with {v['key']};\n"
                f"the rollout feature flag is `{v['flag']}`.")
    else:
        body = (f"$ env | grep {name.upper()}\n"
                f"{name.upper()}_HOST={v['host']}\n{name.upper()}_PORT={v['port']}\n"
                f"{name.upper()}_KEY={v['key']}\n{name.upper()}_FLAG={v['flag']}")
    return body


def _distractor_output(rng, name, v):
    kind = rng.choice(["config", "curl", "db", "grep"])
    if kind == "config":
        return (f"# deploy/values.yaml (excerpt)\nservice: {name}-canary\n"
                f"endpoint:\n  host: {v['d_host']}\n  port: {v['d_port']}\n"
                f"auth:\n  key_prefix: {v['d_key']}\nfeatures:\n  {v['d_flag']}: false")
    if kind == "curl":
        return f"{{\"status\": \"ok\", \"upstream\": \"{v['d_host']}:{v['d_port']}\"}}"
    if kind == "db":
        return (f" id | service   | endpoint\n"
                f"  1 | {name}-old | {v['d_host']}:{v['d_port']}\n"
                f"  2 | edge       | 10.0.{rng.randint(2, 250)}.{rng.randint(2, 250)}")
    return f"src/gateway.py:41: UPSTREAM = \"{v['d_host']}\"  # canary\nsrc/gateway.py:77: PORT = {v['d_port']}"


def _render_entry(rng, name, v, fn="step"):
    t, tmpl = TOOLS_TMPL[rng.randrange(len(TOOLS_TMPL))]
    body = _distractor_output(rng, name, v) if rng.random() < 0.4 else \
        FILLER_BODY.format(mod=name, fn=fn)
    return f"<tool name=\"{t}\">" + tmpl.format(
        file=rng.choice(FILES), pat=f"{name}_", host=v["d_host"],
        port=v["d_port"], body=body, table=f"{name}_events") + "</tool>"


def build_agentmem(rng, target_tokens, depth):
    name = rng.choice(NAMES)
    v = _values(rng, name)
    # transcript budget: chars ≈ tokens * 6 (matches harness haystack calibration)
    aim = int(target_tokens * 0.88 * 3.5)  # measured: transcript content ≈3.5 chars/token
    lines = [f"<session id=\"{name}-{rng.randint(1000, 9999)}\">"]
    total = 80
    # calibrate n_tools for THIS seed: average rendered entry size from a dry sample
    sample = [_render_entry(rng, name, v) for _ in range(60)]
    per_entry = sum(len(s) for s in sample) / len(sample)
    n_tools = max(8, int(aim / per_entry))
    # 1) opening user instruction (establishes the mission, no facts)
    lines += ["<user>Connect to the staging environment and prepare the rollout check "
              "for the next turn. Use only facts you have verified in this session.</user>"]
    # 2) the FACT tool output early at `depth`
    fact_line = f"<tool name=\"read\">{_fact_output(name, v, 'config')}</tool>"
    fact_pos = max(2, int(n_tools * depth))
    for i in range(1, n_tools + 1):
        if i == fact_pos:
            lines.append(fact_line)
        else:
            lines.append(_render_entry(rng, name, v))
        # a mid-trajectory user nudge that references the facts implicitly
        if i == n_tools // 2:
            lines.append(f"<user>Reminder: for the final report use the staging "
                         f"endpoint and credentials you verified earlier — not the "
                         f"canary ones.</user>")
        total += len(lines[-1])
    lines += [
        "</session>",
        "",
        "<user>Final task: state (1) the staging endpoint as host:port, (2) the API key "
        "prefix, (3) the rollout feature flag name — exactly as verified earlier in this "
        "session. One line each, formatted `host:port` / `KEY` / `flag`.</user>",
    ]
    prompt = "\n".join(lines)
    return prompt, {"host": v["host"], "port": v["port"], "key": v["key"],
                    "flag": v["flag"], "distractors": [v["d_host"], str(v["d_port"]),
                                                       v["d_key"], v["d_flag"]]}


def score_agentmem(content, meta):
    c = (content or "").lower()
    hits = 0
    detail = []
    for label, want in (("host", meta["host"].lower()), ("port", str(meta["port"])),
                        ("key", meta["key"].lower()), ("flag", meta["flag"].lower())):
        ok = want in c
        hits += ok
        detail.append(f"{label}={'y' if ok else 'n'}")
    full = f"{meta['host']}:{meta['port']}".lower() in c
    if full:
        detail.append("pair=y")
    return hits / 4.0, " ".join(detail)


def register(tasks_mod):
    tasks_mod.TASKS.append("agentmem")
    tasks_mod.BUILDERS["agentmem"] = build_agentmem
    orig_score = tasks_mod.score

    def score(task, content, meta):
        if task == "agentmem":
            return score_agentmem(content, meta)
        return orig_score(task, content, meta)

    tasks_mod.score = score
    print("agentmem plugin registered: simulated tool-use trajectory, 4 planted facts "
          "+ shape-identical distractors")


if __name__ == "__main__":  # selftest
    import sys
    sys.path.insert(0, __file__.rsplit("/", 1)[0])
    rng = random.Random(11)
    for target in (8_000, 64_000, 512_000):
        for d in (0.02, 0.5, 1.0):
            p, meta = build_agentmem(random.Random(target + int(d * 100)), target, d)
            assert meta["host"] in p and str(meta["port"]) in p
            assert all(x in p for x in meta["distractors"])  # distractors present
            assert len(p) < target * 5.0, len(p)  # ~3.5 chars/token bound
    p, meta = build_agentmem(rng, 8000, 0.1)
    good = (f"{meta['host']}:{meta['port']}\n{meta['key']}\n{meta['flag']}")
    s, det = score_agentmem(good, meta)
    assert s == 1.0 and "pair=y" in det, (s, det)
    bad = f"{meta['distractors'][0]}:{meta['distractors'][1]}\nx\ny"
    s2, det2 = score_agentmem(bad, meta)
    assert s2 == 0.0, (s2, det2)
    print("agentmem selftest OK (fill/distractors/scoring)")
