# MODEL — Muse Glimmer 30B Reference

> Architecture reference for the Muse Glimmer 30B long-context adaptation project.
> Sources: official `config.json` (revision `a4e59da`, Aug 2026), Hugging Face release blog
> ("Meta is back with Muse Glimmer", Aug 10 2026), the Meta GGUF repository, the vLLM recipe,
> and the DFlash paper page (arXiv:2602.06036). Derived/estimated values are marked **[est.]**.

## 1. Overview

| | |
|---|---|
| Full name | Muse Glimmer 30B (`meta-models/Muse-Glimmer-30B`) |
| Release | Meta (Meta Superintelligence Labs), ~Aug 10 2026 |
| License | Apache 2.0 (weights, quants, drafter, perception encoder) |
| Type | Dense multimodal VLM, distilled from Muse |
| Total params | ~29.6B including vision encoder **[est. from published specs]** |
| Composition | 28B text decoder + 2B vision encoder + projector + optional DFlash drafter |
| Native context | 131,072 tokens (`max_position_embeddings`; Meta markets "131,072+") |
| Positioning | Local agentic/coding model for 24–32 GB systems; strong tool use |

## 2. Text Decoder (the part this project adapts)

### 2.1 Global structure

| Parameter | Value |
|---|---|
| `num_hidden_layers` | 52 |
| `hidden_size` | 6,656 |
| `intermediate_size` | 19,968 (gated MLP, SiLU) |
| `num_attention_heads` | 32 |
| `num_key_value_heads` | **2** (16× GQA sharing) |
| `head_dim` | 128 |
| `vocab_size` | 202,048 |
| `tie_word_embeddings` | false |
| `attention_bias` | false |
| `rms_norm_eps` | 1e-5 |
| `post_norm_eps` | 1e-8 |
| `final_logit_softcapping` | 20.0 (Gemma-2-style) |
| `max_position_embeddings` | 131,072 |

### 2.2 Attention architecture — hybrid SWA + NoPE-global

Layer pattern: **(SWA, SWA, SWA, Full) × 13** = 39 sliding-window layers + 13 global full-attention layers.

| Property | SWA layers (39) | Global layers (13) |
|---|---|---|
| Attention scope | sliding window, 2,048 tokens | full context |
| Positional encoding | RoPE, `rope_theta = 500,000` | **NoPE** (`layer_rope_theta = 0`) |
| KV cache growth | fixed (~window-sized, if runtime uses iSWA) | grows linearly with context |

The per-layer `layer_rope_theta` array in `config.json` alternates `500000, 500000, 500000, 0`
through all 52 entries — the `0` entries are the NoPE global layers. Design intent (per Meta):
RoPE layers retain relative order/distance locally; NoPE layers preserve information globally.

**Consequence for context extension:** this is not a conventional RoPE-extension problem.
RoPE only ever operates at ≤ 2,048 relative distance; the extrapolation burden falls entirely
on the 13 NoPE global layers as distractor count grows.

### 2.3 QK normalization and the native attention-temperature knob

Before the attention dot product, every query and key head is RMS-normalized; queries are then
multiplied by an extra scale factor:

| Parameter | Value |
|---|---|
| `qk_scale_factor` | **3.87** |
| `output_multiplier` | 0.19611613… = 1/√26 |

Meta describes the extra query scaling as behaving like an **inverse softmax temperature**.
This is the architecture's built-in knob for attention entropy — the primary zero-shot lever
for the 512k adaptation (see PLAN §4a). QK-RMSNorm also means cached K vectors are normalized,
which is favorable for KV-cache quantization (no K-norm outlier problem).

### 2.4 Parameter budget [est.]

- Embeddings: 202,048 × 6,656 ≈ 1.34B; untied input+output ≈ 2.7B
- Attention per layer: q 6,656×4,096 + k 6,656×256 + v 6,656×256 + o 4,096×6,656 ≈ 58M
- MLP per layer: 3 × 6,656 × 19,968 ≈ 399M
- Per layer ≈ 457M; × 52 ≈ 23.8B; text total ≈ 26.5B **[est.; Meta states 28B — delta likely
  in norms, per-head QK-norm weights, and rounding of component sizes]**

### 2.5 Tensor families

Conventional HF naming (verify against `model.safetensors.index.json` before writing loaders):

- Per text layer `N` in 0..51: `q_proj/k_proj/v_proj/o_proj` (attention), `gate_proj/up_proj/down_proj`
  (MLP), input/post RMSNorm weights, per-head QK-norm weights → 52 × ~7 linear weights plus norms
- `embed_tokens` and `lm_head` (untied)
- Vision tower and projector tensors (see §3) — **frozen in this project**
- GGUF side: separate `mmproj-*.gguf` (vision+projector) and `dflash-*.gguf` (drafter) files;
  `Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf` is the text model

LoRA targets for adaptation: `q_proj, k_proj, v_proj, o_proj` of the **text decoder submodule only**.

## 3. Vision Encoder (frozen; context for GGUF packaging)

| Parameter | Value |
|---|---|
| Type | ViT-like "Perception Encoder" style, ~2B params |
| Layers | 50, pattern (window_attention ×3, full_attention) — note the final two layers are both full attention |
| Hidden / intermediate | 1,536 / 8,960 (GELU MLPs) |
| Heads | 16 |
| Patch | 14 × 14, `patch_temporal = 2` (2 frames per patch group) |
| Positional | learned absolute table, 32 × 32, interpolated; max 1,024 positions |
| Attention RoPE | 2D, `rope_theta = 10,000` (window/full layers as above) |
| Token reduction | pixel shuffle `merge_size = 2` → 2×2 spatial concat → 4× fewer image tokens |
| Projector | hidden 4,096, GELU, `out_hidden_size` 6,144 |

Image input: patches of shape [2 frames × 3 ch × 14 × 14]. Video: sampled at ~2 fps, capped at
96 frames per clip, injected as timestamped placeholders ("Time: 0.0s <|video|> × N").
Image token id 200,092; video token id 200,091.

## 4. KV-Cache Memory Model

Only the 13 global layers grow with context (assuming the runtime uses an interleaved-SWA cache
for the 39 windowed layers — llama.cpp does; verify in logs, `--swa-full` must not be set).

Per-token cost, global layers: `13 × 2 heads × 128 dim × 2 (K,V) × 2 B = 13,312 B/token` at F16.
Fixed SWA cost: `39 × 2,048 × 1,024 B ≈ 82 MB`.

| Context | F16 KV | Q8_0 KV | turbo3 (~4.9×) |
|---|---:|---:|---:|
| 128k | 1.75 GB | 0.9 GB | ~0.36 GB |
| 256k | 3.5 GB | 1.75 GB | ~0.7 GB |
| 512k | 7.0 GB | 3.5 GB | ~1.4 GB |
| 1M | 14 GB | 7 GB | ~2.9 GB |

With ~17 GB K-Quant weights, F16 KV at 512k fits a 32 GB RTX 5090 (~25–27 GB total incl. compute
buffers). TurboQuant: fork-only in llama.cpp (`Madreag/turbo3-cuda` is RTX 5090/sm_120-validated;
CUDA 12.8, FA required), native in vLLM (`--kv-cache-dtype turboquant_4bit_nc`). Glimmer's
QK-RMSNorm + head_dim 128 + only-13-KV-layers profile matches TurboQuant's best-case regime, but
validate with full-depth needle tests (community data shows turbo3 V-cache can degrade generation
trajectories even when PPL looks clean).

## 5. DFlash Speculative-Decoding Drafter

| | |
|---|---|
| What | Optional lightweight **block-diffusion** drafter (parallel token drafting, conditioned on target-model context features) |
| Paper | arXiv:2602.06036 (Z Lab); claims >6× lossless acceleration, up to 2.5× over EAGLE-3 |
| Block size | 16 tokens = 1 anchor + 15 proposed (`--spec-draft-n-max` values > 15 are clamped) |
| transformers | `MuseGlimmerAssistantModel` + `generate(..., assistant_model=…, speculation_type="dflash")` |
| llama.cpp | `--spec-type draft-dflash --spec-draft-n-max 15` (also `-md <dflash gguf> -ngld 99`) |
| Quantized targets | Author-confirmed working with quantized models |
| Project note | After merge + requant, re-check drafter load and **draft acceptance rate** (PLAN §11.6); drafter memory competes with context at 512k (PLAN §12) |

## 6. Tokenizer, Chat Template, Sampling

- Vocabulary 202,048; BOS 200,000, EOS 200,001; image/video token ids 200,092 / 200,091
- Chat template supports `reasoning_strength` (low/…; balances quality vs speed) and tool calling,
  including multimodal tool calls; native open-ended object-detection output format (JSON)
- **Recommended sampling (Meta): temperature 1.0, top-p 0.95, top-k 64** — pin these in all
  evaluation runs; fix and log `reasoning_strength`
- Multilingual

## 7. Official Artifacts & Engine Support

### Repositories

| Artifact | Repo | Size / notes |
|---|---|---|
| BF16 reference | `meta-models/Muse-Glimmer-30B` | 59.58 GB |
| GGUF quants | `meta-models/Muse-Glimmer-30B-GGUF` | `Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf` + `mmproj-…Q4_K_M.gguf` + `dflash-…Q4_K_M.gguf`; no BF16 GGUF published |
| ExecuTorch | `meta-models/Muse-Glimmer-30B-ExecuTorch-PTE` | pre-exported PTE for NVIDIA CUDA & Apple Silicon |
| Drafter (HF) | `meta-models/Muse-Glimmer-30B-assistant` | `MuseGlimmerAssistantModel` |
| Community quants | `RedHatAI/…-FP8-block` 32.78 GB; `Inferact/…-NVFP4-W4A4` 25.42 GB; Unsloth quants | for vLLM-class serving |

### Engine matrix (as of Aug 2026)

| Engine | Status | Key details |
|---|---|---|
| transformers | ✅ day-0, **requires v5** (`5.15.0.dev0`+) | `AutoModelForMultimodalLM` + `AutoProcessor`; CUDA/ROCm/XPU |
| llama.cpp | ✅ day-0, **build ≥ 10353** | text + mmproj + DFlash; `-c N --jinja`; iSWA cache for the 39 SWA layers |
| vLLM | ✅ day-0 | `--model-impl transformers --tool-call-parser muse_glimmer --reasoning-parser muse_glimmer`; native TurboQuant KV; DGX Spark GB10 recipe published |
| SGLang | ✅ branch-based | `muse-glimmer` branch; DFlash loads directly |
| ExecuTorch / MLX / Ollama / LM Studio / Jan | ✅ | optimized integrations rolling out post-launch |
| TRL | ✅ examples shipped | LoRA SFT BF16 measured at ~1×80 GB H100 minimum → DGX Spark 128 GB unified is viable for LoRA-on-BF16 |

### llama.cpp reference invocation

```bash
./llama-server \
    -m Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf \
    --mmproj mmproj-Muse-Glimmer-30B-Q4_K_M.gguf \
    -a muse-glimmer-30B -ngl 99 -c 131072 -np 4 \
    --jinja --temp 1.0 --top-p 0.95 --top-k 64
# + speculative decoding: -md dflash-Muse-Glimmer-30B-Q4_K_M.gguf -ngld 99 \
#                        --spec-type draft-dflash --spec-draft-n-max 15
```

## 8. Published Benchmarks (Muse Glimmer 30B, High Reasoning)

| Benchmark | Score |
|---|---:|
| MCP Atlas | 75.5 |
| DeepSearch QA | 74.6 |
| GAIA2 | 43.3 |
| SWE-Bench Pro | 51.2 |
| SWE-Bench Verified | 76.0 |
| TerminalBench 2.1 | 51.7 |
| **Beam 128K** (long-context) | **65.1** |
| AIME 2026 | 94.7 |
| GPQA Diamond | 83.5 |
| HLE (text, no tools) | 22.0 |
| MMMU Pro | 74 |
| ScreenSpot Pro | 75.4 |

(Compared favorably vs Gemma4-31B-Thinking and Qwen3.6-27B-Thinking on most agentic axes.)

## 9. Long-Context Behavior Notes

- Meta publishes a **Beam 128K = 65.1** score — the model is already benchmarked beyond short context.
- Community report (r/LocalLLaMA, Aug 2026): needle retrieval verified at **1M context zero-shot**
  (anecdotal, 3/3 retrieval; not a formal eval). Consistent with the NoPE-global design
  extrapolating better than RoPE-only architectures.
- Attention temperature (`qk_scale_factor`) is the built-in mechanism most likely to govern
  NoPE-layer behavior under distractor load — primary zero-shot experiment (PLAN §4a).
- YaRN-style RoPE rescaling is expected to be near-inert here: the only RoPE layers are windowed
  to 2,048 relative distance with θ = 500k.

## 10. Properties That Matter for This Project (summary)

1. **2 KV heads** → tiny global KV cache → F16@512k fits 32 GB; TurboQuant optional, not required.
2. **NoPE global layers** → position-ID tricks (PoSE etc.) mostly inert; genuine long sequences
   carry the training signal; attention-temperature is the interesting knob.
3. **VLM packaging** → LoRA on text decoder only; keep `mmproj` + DFlash loadable after requant.
4. **QK-RMSNorm + qk_scale_factor** → favorable KV-quant regime + a config-level temperature knob
   that must survive HF→GGUF conversion (verify metadata in Step 0).
5. **transformers v5 requirement** → pin early; eval harness and trainer must agree on it.
