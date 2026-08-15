# STATUS — Muse Glimmer 30B 512k Adaptation

Short-form progress ledger, updated as phases progress. Detail lives in `docs/` (per-phase
reports); decisions and results of record live in `PLAN.md`/`CONTRIBUTING.md`.
Last updated: 2026-08-15 09:30.

## Interim results (as of 2026-08-15 09:23; interim = grids still filling)

**§3 stock baseline — the defining finding so far** (n=21/cell ≤128k, n=9/cell >128k,
capability contract, zero truncations, zero transport errors):

| task | 32k | 64k | 128k | 192k | 256k | 384k | 512k |
|---|---|---|---|---|---|---|---|
| niah | 1.000 | 1.000 | 1.000 | **1.000** | **1.000** | **1.000** | **1.000** |
| semantic (NoLiMa-style) | 1.000 | 1.000 | 1.000 | **1.000** | **1.000** | **1.000** | 1.000³ |
| niah_multi | 1.000 | 1.000 | 1.000 | queued | queued | queued | queued |
| multihop | 1.000 | 1.000 | 1.000 | queued | queued | queued | queued |
| abstain | 1.000 | 1.000 | 1.000 | queued | queued | queued | queued |
| counting | 0.952 | 0.762 | 0.476 | queued | queued | queued | queued |

¹ complete cells (378/378, `docs/phase3-stock-baseline.md`) · ² >128k grid 66/216 in flight
³ partial (3/3 so far @463k-token prompts)

- **Stock Glimmer retrieves perfectly zero-shot at 4× nominal context** (every depth incl.
  0%/100%). Decision rule (≥85% retention on retrieval) trending "training optional /
  targeted" → project pivots toward PLAN's strengthen-qualify-deploy branch; counting
  (>128k cells queued) is the axis that decides whether adaptation is still needed.
- **Counting = sole degradation axis ≤128k** (0.952→0.476), every miss an exact
  **off-by-one undercount** — systematic attention dilution on the 13 NoPE-global layers,
  exactly PLAN §10's mode B and the §4a qk-scale target. Position-insensitive otherwise.
- **Official NoLiMa suite (31/54)**: 0.444 / 0.556 / 0.444 @32/64/128k, 0.750 @256k(partial)
  — hard semantic benchmark behaving per its leaderboard (~50–70% for best models);
  a genuine difficulty axis our synthetic tasks don't saturate.
- **§0 caveat CLOSED**: K-Quant GGUF 3/3 at 128k/90% under the low-reasoning contract.
- Scoring integrity: one scorer bug (semantic leading-article) caught + fixed live; 63
  cells re-scored from stored responses, flagged `rescored`; scorer validated on 63/63.

## Phase status (PLAN.md numbering)

| Phase | Status | Where |
|---|---|---|
| §0 compat/memory spike | ✅ complete — all gates passed | `docs/phase0-compat-memory-spike.md` |
| §1 environment pinning | ✅ complete | `docs/environment.md` |
| §2 eval harness | ✅ core + NoLiMa + LongBench v2 + LongCodeQA + ∞Bench + custom agentmem; RULER-official/HELMET/LongSWE deferred by design | `docs/phase2-eval-harness.md` |
| §3 stock baseline | 🔄 ≤128k **COMPLETE** (report: docs/phase3-stock-baseline.md — 5/6 tasks 1.000, counting sole axis 0.95→0.48); >128k grid running; §0 caveat closed | `docs/phase3-stock-baseline.md` |
| §4 zero-shot arms | 🔧 arms built & validated (qk 4.1/4.3/4.6/5.0, yarn4, stock-524k); GGUF-metadata spike answered; runs queued behind §3 | `docs/phase4-zeroshot-arms.md`, `outputs/arms/` |
| §5 training corpus | ✅ **v1 BUILT**: 173 rows / 31.5M tokens genuine-length (repos 34.7% · synth 34.1% · natural 16.2% · short 11.6% · agent 3.5% honest shortfall); pi-headless teacher; stage6 G3 gate PASS
| §6 position sampler | ✅ complete + selftested | `docs/phase6-position-sampler.md` |
| §7 QLoRA trainer | 🔧 skeleton complete; `--dry-run` awaits free GPU | `docs/phase7-qlora-trainer.md` |
| §8–§11, §14 | 🔧 pre-staged & armed: §8 auto-eval (stage7), §9 ablate.sh (location/rank arms, one command), §10 diagnose.py (failure-mode classifier → recommended actions), §11 export chain (stage8, regression-guarded); §12 awaits RTX 5090 | `scripts/ablate.sh`, `evals/harness/diagnose.py`, `scripts/export_pipeline.sh` |

## Automation currently running (GPU serial queue)

1. overnight queue (`logs/overnight-queue.log`): §3 ≤128k grid → parquet → llama.cpp
   caveat re-run (closes phase-0 action) → vLLM @524288 (`stock-524k`) → §3 >128k grid
2. suite queue (`logs/suite-queue.log`): NoLiMa / LBv2 / LongCodeQA / ∞Bench / synth-3
   grids at 32k–512k
3. stage3 queue (`logs/stage3-queue.log`): agentmem grid + stock PPL probe (32k–524k)

## Open items / owners

- §5 corpus build at volume: pi-teacher verified; batch generators (synthetic reasoning,
  agent trajectories over training-only repos) are the next build item — no external
  dependency remains.
- §12 on-device qualification: awaits RTX 5090 hardware (all memory numbers decided
  analytically from GB10 measurements in the meantime).
- Review: 3 synthetic tasks (conflicts/set_intersect/chronology) joined after the ≤128k
  grid started — fill-in grid queued (suite queue step E).

## Next-session runbook (check in order)

1. `bash scripts/collect_results.sh` → `docs/results-snapshot.md` (safe any time; auto-covers whatever finished)
2. `for f in logs/*queue*.log; do echo "== $f"; tail -3 $f; done` — chain position + any ERROR/WARN lines
3. Markers: `ls logs/*.done logs/*.launched` — stage order: overnight → suite → stage3 → stage4 → stage5 (dry-run) → stage6 (train1) → stage7 (§8) → stage8 (§11 export) → stage9 (quant parity)
4. If `logs/train1.launched`: `tail -20 logs/train-run1.log` (loss every 10 steps)
5. When §3 grids complete: write `docs/phase3-stock-baseline.md` from the snapshot (retention + decision rule ≥ 85%) — the first real report
6. When §4 arms complete: `evals/harness/compare.py` verdict already in snapshot; update `docs/phase4-zeroshot-arms.md` with results
7. If stage8 BLOCKED (≤128k regression): decide fallback per PLAN §7/§10 (bf16_lora arm is pre-staged) — do NOT blindly rerun
8. §9 after §8: `bash scripts/ablate.sh location` then `rank`; §10: `evals/harness/diagnose.py` output is in the snapshot
9. §12 when RTX 5090 arrives: `docs/phase12-deployment-config.md` checklist (step 1 includes re-verifying llama-server flag spellings)
10. Completion only when every `docs/deliverables.md` row is ✅ with fresh evidence

## Session log (condensed)

- 2026-08-14: §0 gates closed + docs; §1 pins; §2 harness core + smoke; §3 grids launched;
  §4 arms + metadata spike; §6 sampler; §7 skeleton; overnight/suite/stage3 queues armed.
- 2026-08-15: §5 unblocked (pi headless teacher, selftest PASS); GitHub-direct repo
  assembly validated (license/exclusion/determinism/bucketing); Stack-v2 correction
  recorded; STATUS.md instituted per owner request; synth long-doc generator validated
  (pi-written verified sections → grounded var/agg training samples, 30% corpus component);
  serialize bridge validated (chat-template-faithful, loss-on-answer, genuine positions);
  agent-trajectory generator validated (pi as genuine agent with tools via --mode json,
  repo-unique fact ground truth, code-tree materialization); natural-docs + short-replay
  + repo-doc samples + corpus mixer validated (train_v1 manifest: target-vs-actual
  weights, dedupe, genuine-only lengths noted); §5 pipeline complete; batch_generate
  driver launched detached (24 repos · 8 synth docs · 6 book slices · 3 agent sessions ·
  short items → serialize → remix); exclusion gate rejecting eval repos in production
  (jinja/httpx correctly skipped).
- 2026-08-15 (cont.2): §9 lora-scope + §8 compare tool + §11 staged export pipeline
  pre-staged; critical compose-parse fix (comments inside folded command block —
  would have killed tonight's queue restart); stage4 §4-arm sweep armed (qk arms given
  mechanical 524288 window); phase-12 deployment config + qualification checklist;
  corpus repos stage done 15/25 (10 correctly rejected by license/exclusion gates).
- 2026-08-15 (cont.3): export-toolchain rehearsal on tiny/known models — caught & fixed:
  --outfile (b10428 has no --out; would have failed stage 3), llama-imatrix missing from
  image (rebuilt Dockerfile target), sentencepiece added (converter SPM path insurance);
  verified: convert_hf_to_gguf w/ PYTHONPATH shadow → valid GGUF v3, llama-imatrix
  produces real imatrix, quantize accepts --imatrix format. Residual (documented):
  bf16→Q4_K_M-with-imatrix on a runtime-loadable BF16 fixture — real run uses fresh
  convert output (well-formed) + first-class Glimmer arch; stage8 fails observably and
  is idempotent if it still trips.
- 2026-08-15 (cont.4): chain hardening — client non-streaming fallback on 4xx (mock-server
  validated; insurance for first-ever streaming call vs llama-server), stage6 exits with
  blocked marker on dry-run FAILED (was infinite wait); CONTRIBUTING §5 gotchas recorded;
  next-session runbook added.
- live §3 signal (00:50): counting degrades with length — 0.952 @32k → 0.625 @64k (n=8,
  partial) — while niah/niah_multi/multihop stay 1.000 everywhere. First non-trivial
  baseline result: aggregation/counting is the weak axis, retrieval is robust ≤128k.
  Feeds phase-3 report + §10 (expected mode B: distractor-load, not positional).
- counting-miss anatomy (00:52): ALL misses are exact off-by-one UNDERCOUNTS (8/9, 11/12,
  9/10, 6/7) — systematic attention dilution (drops 1 of k markers), not random collapse
  and not positional. Direct §10 implication: aggregation-under-distractor data +
  global-layer focus; also the cleanest §8 regression metric for run1.
- 2026-08-15 (cont.5, 03:40): corpus length-mixture audit caught a hidden assumption —
  seq_bucket=131072 would silently drop 68/173 rows (26M of 31.5M tok; repos sit at
  186k-1.5M). Fixes: manifest now reports per-bucket stats; stage6 G3 gate is
  bucket-aware (checks what the trainer sees). Band fill: +12 synth (slow path),
  +32 book slices, +5 repos → v1.1: 258 raw / @131072 173 rows, 10.87M tok visible,
  genuine 96-128k = 40 rows, 39.1% of visible tokens. Deviation from §7's 55-70%
  documented (synth=30%/doc is pi-bound at ~30s/section; activation memory caps bucket
  at 131k, more conservative than PLAN's 128-256k fallback); §8/§10 diagnostics decide
  whether v2 needs more. Source shares: natural 20.2% (over 15% target, from band
  filling — accepted, recorded).
- 2026-08-15 03:53 MILESTONES: §3 ≤128k grid COMPLETE (378/378): niah/niah_multi/multihop/
  semantic/abstain ALL 1.000±0.000 (n=21 each); counting sole degradation axis
  (0.952/0.762/0.476 @32/64/128k, all off-by-one undercounts). §0 caveat CLOSED
  (K-Quant 3/3 under low-reasoning contract — phase-0 action item done). Scorer bug
  fixed live (semantic article) — 63/63 re-scored hit, flagged rescored. vLLM @524288
  on stock-524k serving (window fix validated live); >128k grid started — first cell
  niah@192k = 1.0 hit @173k prompt tokens.
- 2026-08-15 04:21: cwe task added (RULER-style aggregation — extends measured weak axis).
  NOTE for next session: cwe is in NO armed grid (their task lists are baked; scripts are
  mid-flight — do not edit). Follow-up when GPU serves stock-524k again: run
  `run_eval.py --tasks cwe --ctx 32000,64000,128000,256000 --depths 0,0.5,1.0 --reps 3`
  for stock; and add the same grid for run1 during §8 (stage7's standard grid predates
  cwe). Live: niah@256k 3/3 perfect so far.
- 2026-08-15 04:35: agent-component expansion attempted over 8 fact-bearing repos:
  +2 sessions (fd, aiohttp earlier) → 7 rows; Go/Rust code trees lack unique version
  facts (duplicated across modules) — extractor ceiling reached at ~2.7% vs 10% target.
  Accepted + documented (mixer records actuals); v2 options: synthetic-trajectory
  augmentation or a context-signature fact extractor, only if §8 diagnostics flag the
  agentmem axis as limiting. Corpus v1.1 final: 259 raw / 174 visible @131072.
- 2026-08-15 04:35: GRID-DESIGN FIX — stage4 arm sweep now includes counting (the weak axis;
  qk hypothesis is specifically about attention dilution — untestable without it) and
  stage7 §8 grid adds counting+cwe. Watchers killed by pid (stale ones held file offsets
  into edited scripts) and re-armed fresh (04:34). Stock >128k grid untouched (mid-flight;
  its task list predates cwe — manual follow-up already noted).
- live (05:00): niah@384k 3/3 (347k-ptok prompts) — stock retrieval extrapolation holds
  to 3× nominal; weak-axis cells (counting@192k+) approaching in task-major order.
