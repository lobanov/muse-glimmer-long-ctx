# PLAN — Muse Glimmer 30B 512k Context Extension

## 1. Establish the Reproducible Environment

Use:

- NVIDIA NGC PyTorch container on DGX Spark
- PyTorch
- Hugging Face Transformers
- PEFT
- bitsandbytes
- Accelerate
- Hugging Face Datasets / PyArrow / Parquet
- SGLang for research inference
- llama.cpp for final GGUF/K-Quant deployment

Pin all versions and commits once a known-good Glimmer configuration is established.

Avoid introducing DeepSpeed, FSDP, Axolotl, LLaMA-Factory or NeMo unless later experiments demonstrate a concrete need.

---

## 2. Build the Evaluation Harness First

Before training, create a common runner and normalized result format.

Evaluate at:

- 32k
- 64k
- 128k
- 192k
- 256k
- 384k
- 512k

Where practical, vary evidence position across approximately:

- 0%
- 10%
- 25%
- 50%
- 75%
- 90%
- 100%

Integrate:

1. RULER
2. NoLiMa
3. LongCodeBench / LongCodeQA
4. LongSWE-Bench
5. LongBench v2
6. ∞Bench
7. HELMET
8. custom agentic-memory tests

Store results in a common Parquet schema so configurations can be compared directly.

---

## 3. Establish the Stock Glimmer Baseline

Run unmodified Glimmer beyond its nominal 128k context limit where the runtime permits.

Test:

- 128k
- 192k
- 256k
- 384k
- 512k

Measure:

- task accuracy
- retrieval position sensitivity
- multi-hop degradation
- code/repository performance
- perplexity where useful
- prompt-ingestion latency
- peak memory
- decode speed

This determines whether Glimmer's Local-RoPE / Global-NoPE architecture already extrapolates substantially without training.

Do not assume YaRN is necessary until this baseline is measured.

---

## 4. Create a Zero-Shot YaRN-4 Configuration

Configure:

- maximum context: 524,288
- YaRN factor: 4
- original context: 131,072
- original local RoPE theta retained
- standard YaRN beta/scaling defaults initially

Apply YaRN only through the model's existing RoPE mechanism; do not force rotary embeddings into the NoPE global layers.

Run the full baseline evaluation again.

Compare:

**Stock Glimmer vs YaRN-4 zero-shot**

This isolates the benefit or harm from positional rescaling before training.

---

## 5. Prepare the Training Corpus

Target mixture:

| Component | Approx. share |
|---|---:|
| Whole code repositories | 35% |
| Synthetic long-context reasoning | 30% |
| Long natural documents | 15% |
| Coding-agent trajectories | 10% |
| Short-context replay | 10% |

### Repository Data

Use repository-level samples from sources such as The Stack v2 / Software Heritage.

Prefer repositories spanning:

- 32–64k tokens
- 64–128k
- 128–256k
- 256–512k

Generate tasks requiring evidence across multiple files and distant locations.

Strictly exclude repositories used by evaluation suites such as LongCodeBench, RepoQA and SWE-bench-derived held-out sets.

### Synthetic Long-Context Data

Generate tasks analogous in capability to RULER and NoLiMa, without copying benchmark examples or templates.

Include:

- single retrieval
- multi-needle retrieval
- conflicting facts
- variable/entity tracking
- multi-hop chains
- aggregation/counting
- set intersection
- chronological reconstruction
- semantic retrieval with little lexical overlap

Control both task difficulty and evidence position.

### Natural Long Documents

Use a limited amount of:

- books
- technical manuals
- scientific papers
- related-document collections
- documentation corpora

Prefer tasks requiring cross-section synthesis rather than simple extraction.

### Coding-Agent Trajectories

Generate tool-using coding sessions over training-only repositories.

Retain:

- tool calls
- tool results
- failed hypotheses
- tests
- intermediate discoveries
- long command output

Construct tasks where later decisions depend on information observed tens or hundreds of thousands of tokens earlier.

### Short Replay

Retain approximately 10% high-quality ordinary instruction/coding data to reduce regression.

---

## 6. Build a Custom Context/Position Sampler

Implement a small reusable component producing:

- `input_ids`
- `position_ids`
- labels
- loss masks
- evidence positions
- physical sequence length
- virtual context length

Support at least:

1. normal positions
2. uniform positional offsets
3. randomized segments
4. PoSE-style skipped positions
5. Randomized-YaRN-style virtual ranges
6. genuine long sequences

This component is central to the experiment and should remain independent from the trainer.

---

## 7. Run the First Adaptation Experiment

Start with QLoRA rather than full fine-tuning.

Initial configuration:

- 4-bit NF4 base
- BF16 compute
- LoRA rank 16–32
- gradient checkpointing
- single-process Accelerate
- attention projections first:
  - `q_proj`
  - `k_proj`
  - `v_proj`
  - `o_proj`

Do not initially adapt all MLP layers.

Use a training-length mixture approximately like:

- 50–60%: 32–64k physical sequences with virtual positions spanning up to 512k
- 25–35%: genuine 96–128k sequences
- 10–20%: ordinary short-context replay

Include some genuinely long sequences because virtual positions alone cannot reproduce the distractor load experienced by the 13 global full-attention layers.

---

## 8. Evaluate the First Trained Model

Compare three states:

1. stock Glimmer
2. YaRN-4 zero-shot
3. YaRN-4 + QLoRA adaptation

For every state, measure the same context-length buckets and benchmark subsets.

Primary questions:

- Does YaRN help at all?
- Does training improve 256k–512k performance?
- Is retrieval failure positional or primarily attention/selectivity related?
- What capability is lost at <=128k?
- Where does degradation become steep?

---

## 9. Perform Targeted Ablations

Only after the first end-to-end result, test:

### LoRA location

Compare:

- local RoPE layers only
- global NoPE layers only
- all attention layers

This can reveal whether the limiting factor is positional adaptation or global retrieval/selectivity.

### LoRA capacity

Compare, as needed:

- rank 8
- rank 16
- rank 32
- rank 64

### Training-position strategy

Compare:

- genuine long sequences only
- PoSE/randomized virtual positions only
- mixed strategy

### YaRN factor

If useful, compare:

- 2× / 256k
- 3× / ~384k
- 4× / 512k

The goal is not necessarily the largest nominal context; it is the strongest useful context under the deployment constraint.

---

## 10. Add Training Only Where the Diagnostics Indicate

If local positional errors dominate:

- refine YaRN parameters
- increase position-randomized training
- concentrate LoRA capacity on local attention layers

If global retrieval/selectivity dominates:

- increase multi-needle and distractor-heavy data
- increase genuine long-context examples
- concentrate adaptation on global NoPE attention layers
- generate harder semantic and multi-hop retrieval tasks

If short-context regressions appear:

- increase replay proportion
- reduce LoRA rank
- reduce learning rate or training duration
- constrain adaptation to fewer modules

---

## 11. Merge and Export the Best Adapter

Once a clear checkpoint wins:

1. reload the original Hugging Face model
2. merge the LoRA adapter
3. validate the merged BF16 checkpoint
4. export to GGUF
5. quantize to the target approximately 17 GB K-Quant format

Do not assume BF16 evaluation results transfer perfectly through quantization.

---

## 12. Qualify the Final 32 GB Deployment Artifact

Run the final GGUF in llama.cpp on the target 32 GB GPU.

Test:

- 128k
- 256k
- 384k
- 512k

Measure:

- total VRAM
- KV-cache memory
- prompt-processing speed
- decode speed
- benchmark accuracy
- stability
- quantization regression

Start with F16 KV.

If 512k memory margin is insufficient, test Q8 KV and quantify any quality/performance change.

---

## 13. Optional 1M Follow-On

Only attempt 1M after 512k is robust.

Potential configuration:

- same approximately 17 GB weights
- Q8 KV cache
- extended YaRN/position training

Treat 1M as a separate research target because full-attention compute in the 13 global layers is likely to become more important than raw VRAM capacity.

---

## 14. Final Deliverables

Produce:

1. reproducible environment specification
2. training-data generation scripts
3. context/position sampler
4. QLoRA training configuration
5. benchmark harness
6. baseline and ablation results
7. merged Hugging Face checkpoint
8. approximately 17 GB K-Quant GGUF
9. 32 GB / 512k deployment configuration
10. concise report describing:
   - effective context
   - memory use
   - throughput
   - benchmark results
   - regressions
   - remaining limitations

## Decision Rule

The final model should be selected on **useful long-context performance under the 32 GB deployment constraint**, not on nominal context length alone.
