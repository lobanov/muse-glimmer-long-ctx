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
| §3 stock baseline | 🔄 running — ≤128k grid 64+/378 cells (0 errors, perfect so far); >128k + suites queued overnight | `logs/eval-*.log`, `outputs/eval/` |
| §4 zero-shot arms | 🔧 arms built & validated (qk 4.1/4.3/4.6/5.0, yarn4, stock-524k); GGUF-metadata spike answered; runs queued behind §3 | `docs/phase4-zeroshot-arms.md`, `outputs/arms/` |
| §5 training corpus | ✅ PIPELINE COMPLETE — all 5 components validated end-to-end (repos 35% · synth 30% · natural 15% · agent 10% · short 10%), mixer emits train.jsonl + manifest; scale-up = runtime | `docs/phase5-corpus-plan.md`, `src/muse_longctx/corpus/`, `outputs/corpus/train_v1/` |
| §6 position sampler | ✅ complete + selftested | `docs/phase6-position-sampler.md` |
| §7 QLoRA trainer | 🔧 skeleton complete; `--dry-run` awaits free GPU | `docs/phase7-qlora-trainer.md` |
| §8–§14 | ⏸ sequenced after §3/§4/§5–§7 (eval, ablations, merge/export, 5090 qualification, deliverables) | PLAN.md |

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
  weights, dedupe, genuine-only lengths noted); §5 pipeline complete, scale-up = runtime.
