# GOAL — Muse Glimmer 30B Long-Context Adaptation

## Objective

Adapt **Muse Glimmer 30B** to provide a **reliably usable 512k-token context window** while remaining deployable on a **32 GB VRAM GPU** using the approximately **17 GB K-Quant GGUF** variant.

Training and experimentation will be performed on a **DGX Spark**.

## Target State

- **Qualified maximum context:** 524,288 tokens
- **Recommended operating range:** 256k–384k tokens
- **Baseline/native context:** 131,072 tokens
- **Context-extension method:** YaRN, initially at **4×**
- **Deployment weights:** approximately 17 GB K-Quant GGUF
- **Primary KV-cache target:** F16 at 512k
- **Optional KV-cache fallback:** Q8 if additional memory headroom is required
- **Training method:** QLoRA/LoRA on the Hugging Face model, followed by merge and requantization
- **Primary workload:** long-horizon coding and agentic workloads

## Architectural Constraint

Glimmer has a mixed attention architecture:

- 52 transformer layers
- 39 local sliding-window attention layers
- 13 global full-attention layers
- local window: approximately 2,048 tokens
- only the local layers use RoPE
- the global layers are NoPE

Therefore, this is **not simply a conventional RoPE-extension problem**.

YaRN can stabilize positional extrapolation in the local RoPE path, but the more important challenge may be maintaining effective retrieval and reasoning in the global NoPE layers as the number of distractor tokens grows.

## Core Research Questions

1. How far can stock Glimmer extrapolate beyond 128k without retraining?
2. Does YaRN materially improve 256k–512k behaviour?
3. If adaptation is required, how little training is needed?
4. Does long-context adaptation preserve short- and native-context capability?
5. Can the resulting model remain practical within a 32 GB inference budget?
6. Does the model remain useful on realistic repository-scale and agentic workloads, rather than merely passing synthetic needle tests?

## Success Criteria

The project is successful if the final quantized model:

1. Loads and runs at **512k context within 32 GB VRAM**.
2. Shows controlled degradation rather than collapse from 128k to 512k.
3. Retains strong retrieval across the full context, including evidence near the beginning and end.
4. Performs meaningful multi-hop reasoning across widely separated evidence.
5. Improves or maintains repository-scale coding performance at 256k–512k.
6. Shows minimal regression at <=128k.
7. Produces materially better long-context results than stock Glimmer at the same context length.
8. Retains the practical inference advantages of the approximately 17 GB K-Quant deployment artifact.

## Evaluation Philosophy

Do not optimize for a single aggregate benchmark.

Measure separately:

- positional/context extrapolation
- retrieval under distraction
- semantic rather than lexical retrieval
- multi-hop integration
- repository-level code comprehension
- repository-level software repair
- long agent trajectory memory
- short-context regression

Keep all evaluation benchmark repositories and examples strictly excluded from training data.

## Primary Evaluation Suite

- **RULER** — controlled effective-context curves
- **NoLiMa** — semantic retrieval under low lexical overlap
- **LongCodeBench / LongCodeQA** — repository-scale code understanding
- **LongSWE-Bench** — repository-scale code repair
- **LongBench v2** — realistic long-context reasoning
- **∞Bench** — broad >100k-context evaluation
- **HELMET** — general long-context quality and <=128k regression
- **Custom agentic-memory benchmark** — persistence of relevant information across long tool-use trajectories

## Preferred Outcome

The desired result is not merely “Glimmer accepts 512k tokens.”

The desired result is:

> **A 17 GB-class Glimmer model that can use a 512k context meaningfully for long-horizon coding and agentic inference on a 32 GB GPU.**
