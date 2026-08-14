#!/usr/bin/env python3
"""PLAN §4 — build config-only zero-shot arm directories.

Creates outputs/arms/<name>/ containing a patched config + symlinked weights so any
engine can serve the arm via a local path (no 60 GB copies; safetensors symlinked).

Arms (HF-research path only; GGUF deployment path for the winner goes through §11):
    qk4.1 / qk4.3 / qk4.6 / qk5.0  text_config.qk_scale_factor swept globally.
        NOTE: transformers applies qk_scale_factor as a single scalar on every layer
        (modeling line: query_states = qk_norm(q) * qk_scale_factor) — per-layer values
        are NOT supported, so the plan's "global layers only" preference is unavailable;
        the sweep hits SWA layers too (tolerable: they operate at ≤2048 relative distance).
    yarn4  text_config.rope_parameters = yarn, factor 4, original 131072;
           max_position_embeddings -> 524288. Standard transformers rope machinery incl.
           YaRN attention_scaling; only the 39 RoPE/SWA layers consume position
           embeddings (NoPE layers get position_embeddings=None), matching the plan's
           "YaRN through the existing RoPE mechanism only".

Usage: python3 scripts/make_zero_shot_arm.py --arm qk4.3 [--arm yarn4 ...]
"""
import argparse
import os
import shutil

from transformers import AutoConfig

QK_ARMS = {"qk4.1": 4.1, "qk4.3": 4.3, "qk4.6": 4.6, "qk5.0": 5.0}


def build(arm: str, base: str, out_root: str):
    out = os.path.join(out_root, arm)
    os.makedirs(out, exist_ok=True)
    cfg = AutoConfig.from_pretrained(base)
    tc = cfg.text_config
    if arm in QK_ARMS:
        tc.qk_scale_factor = QK_ARMS[arm]
    elif arm == "yarn4":
        tc.rope_parameters = {"rope_type": "yarn", "factor": 4.0,
                              "original_max_position_embeddings": 131072,
                              # explicit: yarn init reads rope_theta from this dict and
                              # the 10k default would silently rewrite SWA frequencies
                              "rope_theta": float(tc.rope_parameters["rope_theta"])}
        tc.max_position_embeddings = 524288
    elif arm == "stock-524k":
        # MECHANICAL window extension only (max_position_embeddings 131072 -> 524288).
        # No adaptation method: NoPE layers take no positions; RoPE lives on SWA layers
        # at <=2048 relative distance. Required for engines to ACCEPT >131k prompts
        # when evaluating stock Glimmer beyond its nominal limit (PLAN §3 "where the
        # runtime permits").
        tc.max_position_embeddings = 524288
    elif arm == "stock":
        pass
    else:
        raise SystemExit(f"unknown arm {arm}")
    cfg.save_pretrained(out)

    # symlink everything else from the base snapshot (weights, tokenizer, template)
    snap = os.path.realpath(os.path.join(
        os.environ.get("HF_HOME", "/cache/huggingface"),
        "hub", "models--" + base.replace("/", "--"), "snapshots"))
    snap = os.path.join(snap, os.listdir(snap)[0])
    for f in os.listdir(snap):
        if f == "config.json" or os.path.exists(os.path.join(out, f)):
            continue
        os.symlink(os.path.join(snap, f), os.path.join(out, f))
    print(f"[arm {arm}] -> {out}  (qk_scale_factor={getattr(tc, 'qk_scale_factor')}, "
          f"rope_parameters={getattr(tc, 'rope_parameters', None)}, "
          f"max_pos={tc.max_position_embeddings})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", action="append", required=True,
                    choices=[*QK_ARMS, "yarn4", "stock", "stock-524k"])
    ap.add_argument("--base", default="meta-models/Muse-Glimmer-30B")
    ap.add_argument("--out-root", default="outputs/arms")
    a = ap.parse_args()
    for arm in a.arm:
        build(arm, a.base, a.out_root)
