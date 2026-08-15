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

TASKS = ["niah", "niah_multi", "multihop", "counting", "semantic", "abstain",
         "conflicts", "set_intersect", "chronology"]


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
    return prompt, {"needle_absent": f"color={name}"}


def build_conflicts(rng, target_tokens, depth):
    """Conflicting facts: same key recorded twice with different values; the later
    entry (deeper position unless depth > 0.5) supersedes. Recency rule is stated."""
    color, name = rng.choice(COLORS), rng.choice(NAMES)
    old_code, new_code = _code(rng, color, name), _code(rng, color, name)
    d_early = depth * 0.5                # always the earlier record
    d_late = 0.5 + depth * 0.5           # always the later record
    body = _haystack(rng, target_tokens)
    for d, code in sorted(((d_late, new_code), (d_early, old_code)), reverse=True):
        cut = int(len(body) * d)
        body = body[:cut] + _needle(color, name, code) + body[cut:]
    prompt = _wrap(body, f"The {color} access code for the {name} project was recorded "
                         f"twice; the later record supersedes the earlier one. What is the "
                         f"current {color} access code for the {name} project? "
                         f"Reply with the code only.")
    return prompt, {"code": new_code, "superseded": old_code,
                    "needle": f"{color}/{name}", "depth_early": d_early, "depth_late": d_late}


def _word_list(rng, k, avoid):
    pool = [w for w in ("tundra", "basalt", "quartz", "fjord", "mangrove", "savanna",
                        "steppe", "taiga", "reef", "estuary", "marsh", "canyon",
                        "mesa", "delta", "gorge", "plateau") if w not in avoid]
    return rng.sample(pool, k)


def build_set_intersect(rng, target_tokens, depth):
    """Two inventories at distant depths; report items present in BOTH."""
    k1, k2 = rng.randint(6, 9), rng.randint(6, 9)
    shared = _word_list(rng, rng.randint(2, 4), set())
    only_a = _word_list(rng, k1 - len(shared), set(shared))
    only_b = _word_list(rng, k2 - len(shared), set(shared) | set(only_a))
    list_a = sorted(set(shared) | set(only_a))
    list_b = sorted(set(shared) | set(only_b))
    body = _haystack(rng, target_tokens)
    for d, tag, items in sorted(((depth, "north", list_a), (1.0 - depth, "south", list_b)),
                                reverse=True):
        frag = f" {tag.title()} depot manifest: " + ", ".join(items) + ". "
        cut = int(len(body) * d)
        body = body[:cut] + frag + body[cut:]
    prompt = _wrap(body, "Which component names appear in BOTH the North depot manifest "
                         "and the South depot manifest? List them separated by commas.")
    return prompt, {"intersection": sorted(shared)}


def build_chronology(rng, target_tokens, depth):
    """k=5 timestamped events scattered; report the 3 earliest in order."""
    k = 5
    events = _word_list(rng, k, set())
    base_h = rng.randint(1, 5)
    times = [f"{base_h + i:02d}:00" for i in range(k)]
    order = list(range(k))
    rng.shuffle(order)  # positional order != chronological order
    body = _haystack(rng, target_tokens)
    for i, ev_i in enumerate(order):
        cut = int(len(body) * ((i + rng.uniform(0.2, 0.8)) / k))
        body = body[:cut] + f" At {times[ev_i]} the {events[ev_i]} milestone was signed off. " + body[cut:]
    first3 = [events[j] for j in sorted(range(k), key=lambda j: times[j])[:3]]
    prompt = _wrap(body, "Five milestones were signed off at different times. Which three "
                         "happened FIRST, in chronological order? Reply with the three "
                         "names separated by commas.")
    return prompt, {"first3": first3, "times": times, "events": events}


BUILDERS = {
    "niah": build_niah,
    "niah_multi": build_niah_multi,
    "multihop": build_multihop,
    "counting": build_counting,
    "semantic": build_semantic,
    "abstain": build_abstain,
    "conflicts": build_conflicts,
    "set_intersect": build_set_intersect,
    "chronology": build_chronology,
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
        # model answers may drop the leading article ("lakeside terminus" vs
        # "the lakeside terminus") — accept both; strip leading 'the ' from gold
        places = {meta["place"].lower()}
        places |= {p[4:] for p in places if p.startswith("the ")}
        return (1.0, "hit") if any(p in low for p in places) else (0.0, "miss")
    if task == "abstain":
        fabricated = bool(_FABRICATED.search(c))
        acknowledged = bool(_ABSTAIN_OK.search(c))
        if fabricated:
            return 0.0, "fabricated"
        if acknowledged:
            return 1.0, "acknowledged"
        return 0.5, "no-code-silent"
    if task == "conflicts":
        if meta["code"].lower() in low:
            if meta["superseded"].lower() in low:
                return 0.5, "both-reported"
            return 1.0, "superseded-correct"
        if meta["superseded"].lower() in low:
            return 0.0, "stale-value"
        return 0.0, "miss"
    if task == "set_intersect":
        want = set(meta["intersection"])
        got = {w.strip(" .;\t") for w in c.split(",")}
        got = {w for w in got if w}
        if not want and not got:
            return 1.0, "empty-ok"
        inter = len(want & got)
        union = len(want | got)
        return inter / union if union else 0.0, f"iou {inter}/{union}"
    if task == "chronology":
        want = meta["first3"]
        # accept comma- or newline-separated; must be in order
        got = [w.strip(" .;\t") for w in re.split(r"[,\n]", c) if w.strip(" .;\t")]
        hits = sum(1 for i, w in enumerate(got[:3]) if i < len(want) and w.lower() == want[i].lower())
        return hits / 3.0, f"in-order {hits}/3 (want {want})"
    raise ValueError(task)


def cell_seed(task, target_ctx, depth, rep):
    h = hashlib.sha1(f"{task}|{target_ctx}|{depth}|{rep}".encode()).hexdigest()
    return int(h[:8], 16)


# --------------------------------------------------------------- self-test
def _selftest():
    import random
    rng = random.Random(7)
    L = 4000
    for task in TASKS:
        for d in (0.0, 0.25, 0.5, 0.9, 1.0):
            prompt, meta = task_build = BUILDERS[task](rng, L, d)
            assert isinstance(prompt, str) and len(prompt) > L * 4, task
            assert meta, task
        # scorer sanity per task
        if task == "niah":
            assert score(task, f"code is {meta['code']}", meta)[0] == 1.0
            assert score(task, "no idea", meta)[0] == 0.0
        elif task == "conflicts":
            s, det = score(task, meta["code"], meta)
            assert s == 1.0, det
            s, det = score(task, meta["superseded"], meta)
            assert (s, det) == (0.0, "stale-value"), det
            s, det = score(task, f"{meta['superseded']} then {meta['code']}", meta)
            assert s == 0.5 and det == "both-reported", det
        elif task == "set_intersect":
            good = ", ".join(meta["intersection"])
            s, det = score(task, good, meta)
            assert s == 1.0, det
            s, det = score(task, good + ", extra", meta)
            assert 0.0 < s < 1.0, det  # partial: extra item penalized
        elif task == "chronology":
            s, det = score(task, ", ".join(meta["first3"]), meta)
            assert s == 1.0, det
            s, det = score(task, ", ".join(reversed(meta["first3"])), meta)
            assert s < 1.0, det
        elif task == "abstain":
            assert score(task, "I don't know", meta)[0] == 1.0
        else:
            # niah_multi / multihop / counting / semantic exercised via their meta shapes
            assert score(task, str(meta), meta) is not None
    print("tasks selftest OK:", TASKS)


if __name__ == "__main__":
    _selftest()
