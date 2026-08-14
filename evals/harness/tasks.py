#!/usr/bin/env python3
"""Synthetic long-context tasks for the eval harness (PLAN §2).

Design rules:
- ground truth is machine-checkable; scorers are pure functions of (task, content, meta)
- evidence position (depth) is controllable; 0.0 = context start, 1.0 = context end
- no benchmark repositories or examples are used or copied (training-data exclusion is
  safe by construction: everything here is generated at run time)

Filler calibration (measured on the parity spike, outputs/parity_spike):
  ~6.0 chars/token for this haystack; FILL_FACTOR 0.90 leaves template+generation headroom.
"""
import hashlib
import re

CHARS_PER_TOKEN = 6.0
FILL_FACTOR = 0.90

COLORS = ["amber", "teal", "violet", "crimson", "sapphire", "ochre", "indigo", "jade"]
NAMES = ["aurora", "meridian", "kestrel", "cobalt", "harbor", "zenith", "palisade", "lantern"]

PARA = ("Log entry {n}: routine diagnostics completed across the western array; telemetry "
        "buffers rotated; two firmware patches queued for review; the night-shift handover "
        "mentions minor variance in coolant pressure but nothing outside tolerance bands.")

TASKS = ["niah", "niah_multi", "multihop", "counting", "semantic", "abstain"]


def _haystack(rng, target_tokens):
    aim = int(target_tokens * FILL_FACTOR * CHARS_PER_TOKEN)
    body, n = "", 0
    while len(body) < aim:
        n += 1
        body += PARA.format(n=n)
    return body


def _needle(color, name, code):
    return f" The {color} access code for the {name} project is {code}. "


def _code(rng, color, name):
    return f"{name}-{rng.randint(1000, 9999)}-{color}"


def _wrap(body, question):
    return (f"Context follows.\n\n{body}\n\nQuestion: {question}")


# ---------------------------------------------------------------- task builders

def build_niah(rng, target_tokens, depth):
    color, name = rng.choice(COLORS), rng.choice(NAMES)
    code = _code(rng, color, name)
    body = _haystack(rng, target_tokens)
    cut = int(len(body) * depth)
    prompt = _wrap(body[:cut] + _needle(color, name, code) + body[cut:],
                   f"What is the {color} access code for the {name} project? "
                   f"Reply with the code only.")
    return prompt, {"code": code, "needle": f"{color}/{name}"}


def build_niah_multi(rng, target_tokens, depth):
    """4 needles spread across the context; anchor depth from the grid + fixed spread."""
    body = _haystack(rng, target_tokens)
    pairs = []
    depths = [max(0.0, min(1.0, depth + off)) for off in (-0.30, -0.10, 0.10, 0.30)]
    for d in depths:
        color, name = rng.choice(COLORS), rng.choice(NAMES)
        code = _code(rng, color, name)
        cut = int(len(body) * d)
        body = body[:cut] + _needle(color, name, code) + body[cut:]
        pairs.append({"code": code, "needle": f"{color}/{name}"})
    q_names = ", ".join(f"{p['needle']}" for p in pairs)
    prompt = _wrap(body, f"Report the four access codes for: {q_names}. "
                         f"Reply with the four codes separated by commas.")
    return prompt, {"codes": [p["code"] for p in pairs], "needles": [p["needle"] for p in pairs]}


def build_multihop(rng, target_tokens, depth):
    """2-hop: pointer needle at `depth`; fact needle far away (mirrored depth)."""
    color, name = rng.choice(COLORS), rng.choice(NAMES)
    fact_name = rng.choice([n for n in NAMES if n != name])
    code = _code(rng, color, name)
    body = _haystack(rng, target_tokens)
    pointer = (f" The {color} access code is not stored here; it is recorded in the "
               f"{fact_name} project's maintenance note. ")
    fact = f" Maintenance note ({fact_name} project): the archived {color} access code is {code}. "
    d_pointer = depth
    d_fact = 1.0 - depth
    for d, frag in sorted(((d_fact, fact), (d_pointer, pointer)), reverse=True):
        cut = int(len(body) * d)
        body = body[:cut] + frag + body[cut:]
    prompt = _wrap(body, f"What is the {color} access code? "
                         f"Reply with the code only.")
    return prompt, {"code": code, "needle": f"{color}/{name}", "via": fact_name}


def build_counting(rng, target_tokens, depth):
    """Count k occurrences of a unique marker phrase spread through the context."""
    body = _haystack(rng, target_tokens)
    marker = f"SIGNAL-{rng.choice('KRXZ')}{rng.randint(10, 99)} telemetry beacon acknowledged"
    k = rng.randint(5, 12)
    for i in range(k):
        cut = int(len(body) * (i + rng.uniform(0.2, 0.8)) / k)
        body = body[:cut] + f" {marker}. " + body[cut:]
    prompt = _wrap(body, f"How many times does the exact phrase \"{marker}\" appear in the "
                         f"context above? Reply with just the number.")
    return prompt, {"marker": marker, "count": k}


def build_semantic(rng, target_tokens, depth):
    """NoLiMa-style: question shares (almost) no vocabulary with the evidence."""
    first, last = rng.choice(NAMES).upper(), rng.choice(COLORS).title()
    place = rng.choice(["the lakeside terminus", "the northern junction",
                        "the old freight yard", "the harbor-side platform"])
    traveler = f"{first} {last}"
    needle = (f"Passenger manifest update: {traveler} will transfer at {place} "
              f"before the final leg.")
    body = _haystack(rng, target_tokens)
    cut = int(len(body) * depth)
    prompt = _wrap(body[:cut] + f" {needle} " + body[cut:],
                   f"At which stop does the traveler named {traveler} change trains? "
                   f"Reply with the stop only.")
    return prompt, {"place": place, "traveler": traveler}


def build_abstain(rng, target_tokens, depth):
    """Needle absent — expected answer: acknowledge absence, do not fabricate."""
    color, name = rng.choice(COLORS), rng.choice(NAMES)
    body = _haystack(rng, target_tokens)  # no needle inserted
    prompt = _wrap(body, f"What is the {color} access code for the {name} project? "
                         f"If the context does not contain it, reply exactly: I don't know.")
    return prompt, {"needle_absent": f"{color}/{name}"}


BUILDERS = {
    "niah": build_niah,
    "niah_multi": build_niah_multi,
    "multihop": build_multihop,
    "counting": build_counting,
    "semantic": build_semantic,
    "abstain": build_abstain,
}


# ---------------------------------------------------------------- scorers

_FABRICATED = re.compile(r"[a-z]+-\d{3,5}-[a-z]+", re.I)
_ABSTAIN_OK = re.compile(
    r"i don'?t know|do not know|don'?t know|not (mentioned|present|contained|included|"
    r"found|provided|stored|listed)|no access code|cannot (find|locate|determine)", re.I)


def score(task, content, meta):
    """Return (score float 0..1, detail str)."""
    c = (content or "").strip()
    low = c.lower()
    if task == "niah":
        return (1.0, "hit") if meta["code"].lower() in low else (0.0, "miss")
    if task == "niah_multi":
        hits = sum(1 for code in meta["codes"] if code.lower() in low)
        return hits / len(meta["codes"]), f"{hits}/{len(meta['codes'])}"
    if task == "multihop":
        return (1.0, "hit") if meta["code"].lower() in low else (0.0, "miss")
    if task == "counting":
        m = re.search(r"\d+", c)
        if not m:
            return 0.0, f"no-int (k={meta['count']})"
        v = int(m.group())
        return (1.0, "exact") if v == meta["count"] else (0.0, f"got {v} want {meta['count']}")
    if task == "semantic":
        return (1.0, "hit") if meta["place"].lower() in low else (0.0, "miss")
    if task == "abstain":
        fabricated = bool(_FABRICATED.search(c))
        acknowledged = bool(_ABSTAIN_OK.search(c))
        if fabricated:
            return 0.0, "fabricated"
        if acknowledged:
            return 1.0, "acknowledged"
        return 0.5, "no-code-silent"
    raise ValueError(task)


def cell_seed(task, target_ctx, depth, rep):
    h = hashlib.sha1(f"{task}|{target_ctx}|{depth}|{rep}".encode()).hexdigest()
    return int(h[:8], 16)
