# Verification Report — GLM-5.3 Adversarial Review (2026-08-15)

**Provenance.** Review run headless (`pi -p --thinking high`, z.ai `glm-5.3`, ~9 min)
over a 63 KB corpus (GOAL, PLAN, STATUS, phase-3/4/5/7 docs, deliverables audit).
Reviewer **did have repo read access** despite `--no-approve` (read-only tools are
auto-approved; its verbatim quote of `tasks.py:106` and stored `reasoning_head` quotes
match disk exactly). Its `--model` invocation and review text: `outputs/review/`.

**Method of this report.** Every checkable claim re-verified against code/data on disk
(zero GPU, zero changes to running queues — per instruction, nothing was actioned).

---

## 1. Claim-by-claim verdicts

### R1 — armed training run "unconditional, unpowered, starved" → **TRUE (all three), one number worse than claimed**

| Sub-claim | Verified |
|---|---|
| 174 visible rows / 10.87M tok at bucket 131072; **81% of token mass dropped** | ✅ exact (manifest: 259 raw / 57.0M → 174 / 10.87M) |
| ~21 optimizer steps (mb1/accum8/1 epoch) | ✅ 174//8 = 21 |
| 96–128k band = 40 rows; **zero** sequences >131k | ✅ |
| Agent component 7 rows / 25,337 tok (3.6k/row) vs "10% target" and GOAL's primary workload | ✅ exact |
| stage6 trains regardless of §4 outcome ("no winner → stock knobs", never "no training") | ✅ code-verified; the only gates are dry-run OK + corpus size + batch done |
| Winner rule detects only huge effects | ✅ **worse than claimed**: arm CI at n=9 (binary, sd≈0.505) = ±0.536 → detectable effect ≈ **+57 pts**, not +35 |

### R2 — dilution hypothesis unverified; evidence contradicts it → **PARTIALLY TRUE (its quotes are real; one proposed mechanism is quantitatively impossible; it exposed a stale claim of ours)**

- Reviewer's reasoning-channel quotes ("Actually entry 31", enumerating 7 of 12 entries): ✅ real, byte-matched against stored `reasoning_head`.
- **Our claim "every miss an exact off-by-one undercount" is FALSE at full scale** — erratum: 12/17 misses are k−1; the other 5 include an **over-count (13 vs 12)** and four under-counts of 2–3. The claim was true for the first 4 misses observed mid-grid and was never re-checked after 128k completed. STATUS/phase-3 prose carries the stale statement (snapshot's anatomy table is computed dynamically and is correct).
- Enumeration-with-uncertainty visible in miss traces: ✅ (entry-number enumeration + self-correction in ~60% of miss heads).
- Reviewer's marker-fusion mechanism ("two markers can land nearly adjacent"): ❌ **quantitatively impossible** — insertion spacing guarantees ≥ 0.4/k of the body between consecutive markers = **2.3k–3.8k tokens** minimum. Markers cannot be adjacent; fusion-by-proximity cannot occur. (The glued-paragraph rendering quirk it flagged is real but cosmetic.)
- Net: alternative hypotheses (enumeration/arithmetic slips, decode strategy at `reasoning_strength:low`) are **live and untested**; "dilution" remains unproven but not refuted. The §4a/stage6 *causal framing* rests on it — a genuine weakness.

### R3 — §4 sweep omits its sharpest instrument and its harm check → **TRUE**

- `cwe` absent from stage4 (added only to stage7/§8): ✅.
- No ≤64k cells in the arm sweep (128k is the shortest): ✅ — short-context damage (GOAL criterion 6) invisible until after arms/training.
- niah/semantic saturated at 1.000 → ~2/3 of arm-sweep cells can only measure *damage*: ✅.

### R4 — criterion 7 unfalsifiable as instrumented → **PARTIALLY TRUE (one sub-claim false)**

- NoLiMa: different instances per length (n=9/cell) → cross-length trend invalid; 0.222@384k vs 0.750@512k is noise: ✅ (builder draws a random instance per cell).
- Corpus↔eval family contamination risk (synthetic corpus trains the same task families the weak-axis instruments measure): ✅ — a real circularity risk, documented as "disjoint surface templates" but never audited.
- **LBv2 512k pool "n≤3": ❌ FALSE** — calibration cache shows **44** instances in [0.5,0.92]×512k (58 @384k). Reviewer error.
- LQA pools 65/47 at 256k/512k (small but real): ✅.

### R5 — regression detection ceiling-blind → **TRUE (in substance)**

Stock is 1.000 on 5/6 ≤128k tasks, so those cells detect only gross damage; the
sensitive instruments (counting/cwe/NoLiMa) are exactly the contaminated/underpowered
ones; HELMET/RULER-official deferred by design. ✅

### R6 — chain fragility details → **TRUE (both), one ordering claim true**

- **Rep correlation**: measured common prefix between reps of one cell = **50% of the prompt** (haystack text is deterministic; only needle/markers differ) → CIs optimistic: ✅.
- `finish_reason=length` logged but not gated in compare paths: ✅.
- Counting runs 5th of 6 in the remaining >128k grid (decision cells last): ✅ (`--tasks niah,semantic,multihop,abstain,counting,niah_multi`).

---

## 2. Compressions & experiments (assessed, NOT executed)

| Item | Assessment |
|---|---|
| C1 conditional train1 (20–42h) | Sound *if* the decision rule is honored; contradicts current armed chain (see R1b) |
| C2 trimmed §4 sweep (~30h saved; +15pt detectable) | Arithmetic checks out (81→~25 cells/arm); the power critique is valid (R1c) |
| C3 reorder counting first (~6–8h earlier gate data) | Valid; requires relaunching remaining grid (resume-safe) |
| C4 YaRN→20-min probe (~8h) | Consistent with PLAN's own "expected near-inert" framing |
| C5 trim suites (~6–8h) | Mixed: LBv2/LQA 512k pools are bigger than reviewer thought (44, not ≤3), so "uninterpretable" is overstated; depth-trim of synth3 valid |
| C6 drop PPL probe | Reasonable (softcapped-logit PPL on a reasoning VLM is weakly interpretable) |
| C7 §8 decision-cells (~8h) | Valid if training runs |
| C8 identity-merge export rehearsal on real 30B | Best de-risk item: no schedule cost, converts §12 from debugging to measurement |
| E1 miss forensics (0 GPU) | **Done during verification** (results above) — inconclusive between dilution vs decode-arithmetic; needs the full-reasoning re-run (E2) to split |
| E2 greedy/enumeration-prompt re-run (~1.5h) | Highest information-per-hour of the GPU items; directly falsifies/validates the decode-strategy hypothesis |
| E3 entropy probe + per-layer qk monkeypatch | Fair reframing: our doc said "not supported natively" (true); 5-line monkeypatch + GGUF-side per-layer absorption make global-only sweeping cheap and deployable — strengthens §4a if pursued |
| E4 powered trimmed qk sweep | Follows from C2 |
| E5 full export rehearsal on real model | See C8 |

## 3. Reviewer errors found

1. **LBv2 512k pool "n≤3"** — false (44 instances; calibration cache).
2. **Marker fusion by adjacency** — quantitatively impossible (min gap 2.3k–3.8k tokens).
3. Winner-rule detectable effect understated (≈+57 pts, not +35) — direction favors its argument.
4. Implied task-design flaw from glued paragraphs — cosmetic, not causal.

## 4. Errata in OUR documents found during verification (not yet fixed)

1. STATUS/phase-3: "every miss an exact off-by-one undercount" → **12/17** (stale mid-grid observation; anatomy table in snapshots is computed and correct).
2. phase-3 "dilution" mechanism stated as established; it is one live hypothesis among ≥3.
3. STATUS "v1 BUILT 173 rows/31.5M" vs final v1.1 259/57.0M raw (visible 174/10.87M) — superseded rows in the ledger read confusingly.

## 5. Disposition (for decision — nothing actioned)

The verified core of the review stands: **the armed chain trains unconditionally on a
21-step corpus under an instrument that cannot detect realistic qk effects, on a
mechanism (dilution) that our own traces now complicate.** Options, in the reviewer's
priority order: gate train1 (C1); run E2 + trimmed sweep (C2/E4); reorder counting (C3);
export rehearsal (C8). Against: corpus and queues are complete and running; gating adds
a decision dependency on human review; the §4 sweep as armed still yields harm-check +
dose-response value if slow. Owner call.
