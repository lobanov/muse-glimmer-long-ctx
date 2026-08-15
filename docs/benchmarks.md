# Evaluation Benchmarks — Complete Reference (PLAN §2 / GOAL.md suite)

One harness (`evals/harness/run_eval.py`), one result schema (Parquet, 24 cols), one
sampling/template contract for every benchmark: **temp 1.0 / top-p 0.95 / top-k 64**
(capability scores), `chat_template_kwargs.reasoning_strength=low`, generous `max_tokens`,
scoring on `message.content` (reasoning captured as diagnostic). Greedy (temp 0) only for
deterministic parity checks. Evidence depth is controllable on all synthetic tasks
(0% = context start, 100% = end). Every task is machine-scored; community suites use
official gold answers.

Groups: **A** synthetic core (10 tasks, contamination-safe by construction) ·
**B** official community suites (5 plugins, eval-only, never training data) ·
**C** operational probes (PPL, parity).

---

## A. Synthetic core — `evals/harness/tasks.py`

All generated at runtime; `target_ctx` ≈ 90% fill (calibrated chars/token); instances
resampled per seed (sha1 of cell identity). Scores are exact-match unless noted.

### A1. `niah` — needle in a haystack (retrieval baseline)
- **Construct**: one planted fact ("The {color} access code for the {name} project is
  {code}") inside a log-entry haystack, at controlled depth.
- **Question**: asks for the code; answer = the code.
- **Measures**: single-fact retrieval vs distractor load; position sensitivity.
- **Score**: 1.0 if code appears in answer.

### A2. `niah_multi` — multi-needle retrieval (4 needles, spread)
- **Construct**: 4 needles at depths {d−0.30, d−0.10, d+0.10, d+0.30} around the anchor.
- **Measures**: multi-instance retrieval with intra-context interference.
- **Score**: fraction of 4 codes present (partial credit).

### A3. `multihop` — 2-hop chained retrieval (mirrored depths)
- **Construct**: pointer needle at depth d ("code is recorded in the {other} project's
  maintenance note"); the actual code in a *different* needle at depth 1−d. Answering
  requires following the pointer across widely separated context regions.
- **Score**: exact code.

### A4. `counting` — aggregation (k occurrences)
- **Construct**: k=5–12 identical marker phrases spread evenly; question asks "how many".
- **Measures**: aggregation under distractor load — the measured weak axis (stock:
  0.952→0.476 over 32k→128k, every miss an exact **off-by-one undercount** = attention
  dilution on the NoPE-global layers).
- **Score**: exact integer.

### A5. `cwe` — common-word extraction (RULER-CWE-style; hardest aggregation)
- **Construct**: 8 candidate words with machine-verified distinct occurrence counts
  (top ≥ runner-up + 3) distributed through the context; question lists candidates and
  asks for the most frequent.
- **Measures**: *comparative* aggregation (must count and compare across candidates) —
  discriminates even at 32k (stock 0.778; wrong-word errors, not undercounts).
- **Score**: exact word.

### A6. `semantic` — NoLiMa-style retrieval (low lexical overlap)
- **Construct**: a traveler-change-trains fact phrased with no vocabulary shared with the
  question ("At which stop does {name} change trains?" vs "will transfer at {place}
  before the final leg").
- **Measures**: semantic rather than lexical retrieval.
- **Score**: place string (leading-article tolerant).

### A7. `abstain` — needle absent (no fabrication)
- **Construct**: question about a code that was never inserted; instructed to reply
  "I don't know".
- **Score**: 1.0 acknowledged absence · 0.0 fabricated a code-shaped string · 0.5 silent.

### A8. `conflicts` — superseded fact (recency resolution)
- **Construct**: same key recorded twice (early at depth·0.5, late at 0.5+depth·0.5),
  different values; question states the later record supersedes.
- **Score**: 1.0 current value only · 0.5 both reported · 0.0 stale value.

### A9. `set_intersect` — two distant lists
- **Construct**: two depot manifests at mirrored depths; report items in BOTH.
- **Score**: IoU over reported set.

### A10. `chronology` — temporal ordering
- **Construct**: 5 timestamped events scattered (positional order ≠ chronological);
  report the 3 earliest in order.
- **Score**: in-order hits / 3.

### A11. `agentmem` — agentic memory (custom; `agentmem.py`)
- **Construct**: simulated coding-agent session transcript (read/grep/curl/db/test tool
  outputs); 4 planted facts (host, port, key prefix, feature flag) in one early tool
  output at controlled depth; ~40% of later outputs carry **shape-identical distractors**
  (canary host/port/key); a mid-session nudge references the facts only implicitly; the
  final turn requires all four.
- **Measures**: persistence of information across long tool-use trajectories (GOAL.md's
  custom-suite item).
- **Score**: fraction of 4 facts exactly recalled (pair bonus recorded).

---

## B. Official community suites (plugins; eval-only)

Licenses noted; all strictly excluded from training data (`data/exclusions/eval_repos.json`
+ fail-closed gates; 6 production rejections logged).

### B1. NoLiMa — `nolima.py` (plugin `--plugin nolima`)
- **Source**: `amodaresi/NoLiMa` (ICML 2025; Adobe Research License, non-commercial
  research). Official needle set (58 instances: 10 tasks × tests × {onehop, twohop}) and
  official word-shuffled book haystacks (10 books, ≈1.9M tokens).
- **Protocol**: official task templates, official "contains" metric, needle at controlled
  depth; corpus fill tokenizer-calibrated (~90%).
- **Measures**: retrieval requiring latent associations with minimal lexical overlap —
  world-knowledge two-hop is the hardest category (stock interim: 0.22–0.75 by length).
- **Canonical lengths** 250–32K; we extend to 512k (labelled extension).

### B2. LongBench v2 — `longbench_v2.py` (plugin `longbench_v2`)
- **Source**: `THUDM/LongBench-v2` (503 MC instances; CC-BY-NC). Contexts 10k–4.3M
  Glimmer tokens (median 97k, measured & cached).
- **Protocol**: official "answer with the option's letter directly"; first standalone
  A–D letter scoring. Instance selection filters to [0.5, 0.92]×target (pools:
  79/104/90/44 at 32k/128k/256k/512k). Depth fixed by instance.
- **Measures**: deep understanding/reasoning across realistic multitasks — single-doc QA,
  multi-doc QA, long ICL, dialogue history, code-repo understanding, structured data.
  Human experts: 53.7% under 15 minutes.

### B3. LongCodeQA — `longcodeqa.py` (plugin `longcodeqa`)
- **Source**: `Steefano/LCB` (LongCodeBench @1M-contexts suite; MIT). 443 MC instances in
  official buckets 32K/64K/128K/256K/512K/1M (counts match paper: 113/76/92/65/47/50).
- **Protocol**: official prompt verbatim; `correct_letter` scoring; bucket = largest ≤
  target. All repos verified present in the exclusion list.
- **Measures**: repository-scale code comprehension (GOAL criterion 5 axis).

### B4. ∞Bench (InfiniteBench) — `infbench.py` (plugin `infbench`)
- **Source**: `xinrongzhang2022/InfiniteBench` (CC-BY-NC). Curated deterministic subset:
  `infb_kv` (kv_retrieval, 500 — exact UUID match from ~50k-token key-value tables),
  `infb_bookmc` (longbook_choice_eng, 229 — full-option-text match),
  `infb_codedebug` (code_debug, 394 — find the broken function).
- **Protocol**: official gold answers, containment scoring; lengths 74k–200k Glimmer
  tokens (slots into 128k/256k; no 512k instances exist).
- **Measures**: broad >100k evaluation (retrieval, book understanding, code debugging).

### B5. LongSWE-Bench — *not yet integrated*
- Requires a test-execution harness; deliberately sequenced after the first end-to-end
  trained result (PLAN §2 note). Repo-scale code repair axis (GOAL criterion 5).
  SWE-bench_Verified repos are already exclusion-listed.

---

## C. Operational probes

### C1. PPL probe — `evals/ppl_probe.py`
Long-context perplexity on real corpora (last-8k-token span; echo+logprobs via vLLM;
self-consistent with `final_logit_softcapping`). Curves: stock (32k–524k) and every
trained/merged config. Length-doubling PPL deltas are the canary before task scores move.

### C2. Parity checks — greedy (temp 0), the only sanctioned non-capability mode
- **BF16 vs K-Quant** (§0 gate; re-run contract: `reasoning_strength=low`, 4096 tokens):
  quant-noise floor — K-Quant 3/3 @128k/90% (closed).
- **Merged vs adapter-on-base** (§11 stage 2): merge correctness.
- **BF16-merged vs new GGUF** (stage9 mini-suite): export correctness, 5-pt noise floor.

---

## Grids & analysis

- Default grid: 7 ctx × 7 depths × ≥3 resamples (t-CIs; `summarize.py`).
- `retention.py`: per-task retention vs 128k + the ≥85% decision rule verdict.
- `compare.py`: config × task × ctx tables, Δ vs reference with significance markers.
- `diagnose.py`: §10 failure-mode classifier (positional / selectivity / short-regression
  / reasoning-drain → recommended actions).
- Counting error-anatomy (in `collect_results.sh`): off-by-one undercount tracking — the
  most sensitive single regression instrument (§8).

## Coverage map (GOAL.md evaluation philosophy → instruments)

| Axis | Benchmarks |
|---|---|
| Positional / context extrapolation | niah, niah_multi, multihop @ depths 0–100%, 32k–512k |
| Retrieval under distraction | niah*, counting, cwe, infb_kv, agentmem |
| Semantic (non-lexical) retrieval | semantic, NoLiMa |
| Multi-hop integration | multihop, NoLiMa twohop |
| Repo-scale code comprehension | LongCodeQA, LongBench v2 (code domain), infb_codedebug |
| Repo-scale code repair | LongSWE-Bench (pending) |
| Long agent-trajectory memory | agentmem |
| Short-context regression | 32k cells of every grid + short-replay corpus eval |
| Realistic long reasoning | LongBench v2, NoLiMa, ∞Bench |
| Quantization/merge integrity | parity probes (C2), PPL |
