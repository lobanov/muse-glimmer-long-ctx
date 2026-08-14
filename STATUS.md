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
| §5 training corpus | ✅ **v1 BUILT**: 151 rows / 30.7M tokens genuine-length (repos 60/23.8M · synth 52/3.6M · natural 26/3.3M · agent 6 · short 7); pi-headless GLM-5.2 teacher; stage6 G3 gate PASS; v2 scale-up targets: short+agent components | `outputs/corpus/train_v1/manifest.json`, `src/muse_longctx/corpus/` |
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
