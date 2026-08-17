#!/usr/bin/env python3
"""RULER (NVIDIA/RULER, Apache-2.0) task recipes as a harness plugin — goal afe6584b.

Implements the public RULER synthetic recipes on this harness's filler/scoring
machinery (our `cwe` task already covers RULER-CWE top-1; niah/niah_multi cover the
basic multi-needle families). New tasks:
    ruler_vt  variable tracking — chains of variable references; report the value
              of the queried chain's end (RULER vt recipe, shortened chains)
    ruler_fwe frequent-word extraction — 20 candidate words with distinct counts;
              report the 10 most frequent (score = hits/10, RULER partial metric)
    ruler_niah_mk  multikey needle — needle kv + 3 look-alike distractor kv pairs
    ruler_niah_mv  multivalue needle — one key, 4 values scattered; report all 4
                  (score = found/4)

Attribution: task structure follows https://github.com/NVIDIA/RULER recipes;
filler and formatting are this repo's (eval-only; never training data).
Register: --plugin ruler    Use: --tasks ruler_vt,ruler_fwe,ruler_niah_mk,ruler_niah_mv
"""
import random

TASKS = ("ruler_vt", "ruler_fwe", "ruler_niah_mk", "ruler_niah_mv")
_WORDS = ["harbor", "cinder", "lantern", "quarry", "thistle", "cobalt", "marble",
          "falcon", "juniper", "saffron", "obsidian", "tallow", "bramble", "kestrel",
          "verdigris", "palisade", "rowan", "ashen", "loden", "wicken"]
_VARS = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel",
         "india", "juliet", "kilo", "lima", "mike", "november", "oscar", "papa",
         "quebec", "romeo", "sierra", "tango", "uniform", "victor", "whiskey",
         "xray", "yankee", "zulu"]


def _val(rng):
    return str(rng.randint(10000, 99999))


def _mk_builder():
    def build(rng, target_tokens, depth, tasks_mod):
        hay = tasks_mod._haystack
        wrap = tasks_mod._wrap
        body = hay(rng, target_tokens)
        cut = int(len(body) * depth)

        # ---- ruler_vt: 1 target chain (len 4) + 40 distractor chains -------------
        def vt():
            tgt = rng.choice(_VARS)
            v0 = _val(rng)
            names = [tgt] + [f"{rng.choice(_VARS)}-{i}" for i in range(40)]
            chain = [f" var {tgt}_1 = {v0}; "]
            for i in range(2, 5):
                chain.append(f" var {tgt}_{i} = {tgt}_{i - 1}; ")
            needle = "".join(chain)
            dist = []
            for n in names[1:]:
                dv = _val(rng)
                dist.append(f" var {n}_1 = {dv}; var {n}_2 = {n}_1; "
                            f"var {n}_3 = {n}_2; var {n}_4 = {n}_3; ")
            dfrac = rng.uniform(0.05, 0.95)
            pos = int(len(body) * dfrac)
            dchunk = "".join(dist)
            body2 = body[:pos] + dchunk + body[pos:]
            cut2 = int(len(body2) * depth)
            body2 = body2[:cut2] + needle + body2[cut2:]
            q = (f"Find all variables that are assigned the value of {tgt}_1 in the "
                 f"context above and reply the final assigned value of {tgt}_4. "
                 f"Reply with the value only.")
            return wrap(body2, q), {"gold": [v0], "chain": f"{tgt}_1..4"}

        # ---- ruler_fwe: 20 words, distinct counts, top-10 gold, score hits/10 ----
        def fwe():
            b = body
            words = rng.sample(_WORDS, 10) + [f"{w}-{i}" for i, w in
                                              enumerate(rng.sample(_WORDS, 10))]
            counts = rng.sample(range(5, 30), 20)
            top = sorted(zip(words, counts), key=lambda x: -x[1])[:10]
            gold = [w for w, _ in top]
            occ = [w for w, c in zip(words, counts) for _ in range(c)]
            rng.shuffle(occ)
            n = len(occ)
            step = max(1, (len(b) - 800) // (n + 2))
            slots = sorted(rng.sample(range(400, len(b) - 400, step), n))
            for p, w in zip(slots, occ):
                b = b[:p] + f" {w} " + b[p:]
            q = ("Below is a list of words. Which 10 words appear most frequently in "
                 "the list above? Reply with exactly the 10 words, comma-separated, "
                 "most frequent first.")
            return wrap(b, q), {"gold": gold, "counts": dict(zip(words, counts))}

        # ---- ruler_niah_mk: needle kv + 3 distractor kvs (RULER single_3 style) --
        def mk():
            keys = rng.sample(_VARS, 4)
            val = _val(rng)
            dis = [(k, _val(rng)) for k in keys[1:]]
            frag = (f" One of the keys below unlocks the maintenance vault. "
                    f"{keys[0]} = {val}; "
                    + " ".join(f"{k} = {v};" for k, v in dis) + " ")
            q = (f"Reply the value that the key {keys[0]} is assigned in the context "
                 f"above. Reply with the value only.")
            return wrap(body[:cut] + frag + body[cut:], q), {"gold": [val], "key": keys[0]}

        # ---- ruler_niah_mv: one key, 4 values; score found/4 ----------------------
        def mv():
            key = rng.choice(_VARS)
            vals = [_val(rng) for _ in range(4)]
            frags = [f" {key} = {v}; " for v in vals]
            ds = sorted(rng.uniform(0.05, 0.95) for _ in range(4))
            b = body
            for d, f in zip(ds, frags):
                p = int(len(b) * d)
                b = b[:p] + f + b[p:]
            q = (f"Find all values assigned to the key {key} in the context above. "
                 f"Reply with all of them, comma-separated.")
            return wrap(b, q), {"gold": vals, "key": key}

        return {"ruler_vt": vt, "ruler_fwe": fwe,
                "ruler_niah_mk": mk, "ruler_niah_mv": mv}

    return build


def register(tasks_mod):
    build = _mk_builder()
    for t in TASKS:
        tasks_mod.TASKS.append(t)
        tasks_mod.BUILDERS[t] = (lambda rng, tt, d, _t=t: build(rng, tt, d, tasks_mod)[_t]())
    orig_score = tasks_mod.score

    def score(task, content, meta):
        if task not in TASKS:
            return orig_score(task, content, meta)
        c = (content or "").strip()
        if not c:
            return 0.0, "empty"
        if task in ("ruler_vt", "ruler_niah_mk"):
            return (1.0, "contains") if any(g in c for g in meta["gold"]) else (0.0, "miss")
        if task in ("ruler_fwe", "ruler_niah_mv"):
            hits = sum(1 for g in meta["gold"] if g in c)
            n = len(meta["gold"])
            return hits / n, f"{hits}/{n}"
        return orig_score(task, content, meta)

    tasks_mod.score = score
    print("ruler plugin registered:", ", ".join(TASKS))
