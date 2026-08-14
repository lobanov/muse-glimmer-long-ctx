# STATUS — Muse Glimmer 30B 512k Adaptation

Short-form progress ledger, updated as phases progress. Detail lives in `docs/` (per-phase
reports); decisions and results of record live in `PLAN.md`/`CONTRIBUTING.md`.
Last updated: 2026-08-15.

## Phase status (PLAN.md numbering)

| Phase | Status | Where |
|---|---|---|
| §0 compat/memory spike | ✅ complete — all gates passed | `docs/phase0-compat-memory-spike.md` |
| §1 environment pinning | ✅ complete | `docs/environment.md` |
| §2 eval harness | ✅ core + NoLiMa + LongBench v2 + LongCodeQA + ∞Bench + custom agentmem; RULER-official/HELMET/LongSWE deferred by design | `docs/phase2-eval-harness.md` |
| §3 stock baseline | 🔄 running — 109+/378 ≤128k cells; niah/niah_multi 100.0±0.0 so far; **plugin suite live-verified** through the real runner (nolima/lb-v2/lqa/infbench/agentmem) — infb_kv 1.0 @124k-ptok UUID retrieval, agentmem 4/4; lb-v2/lqa single-cell misses = real difficulty; >128k + suites + §4 + §7 dry-run queued | `docs/results-snapshot.md` |
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
