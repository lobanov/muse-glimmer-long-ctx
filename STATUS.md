# STATUS — Muse Glimmer 30B 512k Adaptation

Short-form progress ledger, updated as phases progress. Detail lives in `docs/` (per-phase
reports); decisions and results of record live in `PLAN.md`/`CONTRIBUTING.md`.
Last updated: 2026-08-16 09:45.

## Interim results (REGENERATED from disk 2026-08-15 23:40 — computed, not hand-typed;
per-cell n in parens; n<9 = partial; — = queued)

**§3 stock baseline** (capability contract, zero truncations, zero transport errors):

| task | 32k | 64k | 128k | 192k | 256k | 384k | 512k |
|---|---|---|---|---|---|---|---|
| niah | 1.000(21) | 1.000(21) | 1.000(21) | 1.000(9) | 1.000(9) | 1.000(9) | 1.000(9) |
| niah_multi | 1.000(21) | 1.000(21) | 1.000(21) | 1.000(9) | 1.000(9) | 1.000(9) | 1.000(9) |
| multihop (2-hop) | 1.000(21) | 1.000(21) | 1.000(21) | 1.000(9) | 1.000(9) | 1.000(9) | 1.000(9) |
| semantic (NoLiMa-style) | 1.000(21) | 1.000(21) | 1.000(21) | 1.000(9) | 1.000(9) | 1.000(9) | 1.000(9) |
| abstain | 1.000(21) | 1.000(21) | 1.000(21) | 1.000(9) | 1.000(9) | 1.000(9) | 1.000(9) |
| counting | 0.952(21) | 0.762(21) | 0.476(21) | 0.667(9) | 0.667(9) | 0.667(9) | **0.222(9)** |
| cwe (8-word freq compare) | 0.778(9) | 0.889(9) | 0.889(9) | — | 0.778(9) | — | — |
| official NoLiMa | 0.444(9) | 0.556(9) | 0.444(9) | — | 0.667(9) | 0.222(9) | 0.444(9) |

**§3 grids COMPLETE (378 + 216 cells)** — report: `docs/phase3-stock-baseline.md`.
Decision-rule verdict: 100% retrieval retention through 512k → strengthen-qualify-deploy
formally taken. k-matched counting (80/80): no clean length trend at matched k (128k is a
local dip; 256k recovers; capability≈greedy) → aggregation fragility confirmed,
dilution-as-length-mechanism weakened; the 512k=0.222 cell likely overstates the length
effect (uncontrolled k-draws).

- **Stock Glimmer retrieves AND 2-hop-reasons perfectly zero-shot at 4× nominal context**
  (every depth incl. 0%/100%, up to 463k-token prompts). §3's ≥85%-retention rule is
  met on every completed retrieval column → strengthen-qualify-deploy pivot stands;
  **GOAL criterion 7 now hinges entirely on the weak axes.**
- **Weak-axis picture (ERRATUM 2026-08-15/23:40, audit-verified — replaces earlier
  narrative): counting/cwe = aggregation fragility: k-difficulty-dependent (k=5: 1.00 →
  k=12: 0.33), partly stochastic (E2: greedy fixed 5/17 capability misses), NO clean
  length trend beyond 128k (0.667 flat 192–384k, vs 0.476@128k — non-monotone; k-drift
  confound documented). "Attention dilution" is DOWNGRADED to one unconfirmed
  hypothesis — §4a is hypothesis exploration ("does attention sharpening move
  aggregation scores at all?"), not mechanism-targeted treatment.** E2 paired anatomy:
  11/17 misses enumeration-resistant (retrieval-side component real); k-matched grid
  (strata k=6/11, capability+greedy) running to de-confound length from difficulty.
- NoLiMa non-monotone (0.44/0.56/0.44/0.67/0.22/0.44) — instances differ per length;
  treat as per-length difficulty, not a trend. Corroborator suites at n=3/cell (LBv2/LQA)
  cannot detect <~50pt effects — powering decision pending (audit F6).
- **§0 caveat CLOSED**: K-Quant GGUF 3/3 at 128k/90% under the low-reasoning contract.
- Scoring integrity: semantic leading-article scorer bug caught + fixed live (63 cells
  re-scored, flagged); counting-miss-anatomy and E2 claims errata'd twice as data
  landed — numbers in this table are computed from disk.

## Phase status (PLAN.md numbering)

| Phase | Status | Where |
|---|---|---|
| §0 compat/memory spike | ✅ complete — all gates passed | `docs/phase0-compat-memory-spike.md` |
| §1 environment pinning | ✅ complete | `docs/environment.md` |
| §2 eval harness | ✅ core + NoLiMa + LongBench v2 + LongCodeQA + ∞Bench + custom agentmem; RULER-official/HELMET/LongSWE deferred by design | `docs/phase2-eval-harness.md` |
| §3 stock baseline | 🔄 ≤128k **COMPLETE** (report: docs/phase3-stock-baseline.md — 5/6 tasks 1.000, counting sole axis 0.95→0.48); >128k grid running; §0 caveat closed | `docs/phase3-stock-baseline.md` |
| §4 zero-shot arms | 🔧 arms built & validated (qk 4.1/4.3/4.6/5.0, yarn4, stock-524k); GGUF-metadata spike answered; runs queued behind §3 | `docs/phase4-zeroshot-arms.md`, `outputs/arms/` |
| §5 training corpus | ✅ **v1.1 final**: 259 raw / 174 trainer-visible @131072 / 10.87M visible tokens (repos ~31% · synth 34% · natural 20% · short 12% · agent 2.7% honest shortfall); pi-headless teacher; bucket-aware G3 PASS |
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
- counting-miss anatomy (ERRATUM 2026-08-15, post-review): early claim "ALL misses
  off-by-one" was stale — full data: 12/17 are k−1, plus one OVER-count (13 vs 12) and
  four under-counts of 2–3. Undercount-biased, not perfectly systematic. Mechanism NOT
  established: dilution vs enumeration/arithmetic slips both live → E2 forensics lane
  running (`outputs/eval/e2_counting_forensics.jsonl`).
  (Superseded mid-grid note removed: early 4-miss sample had shown only k−1.)
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
  (0.952/0.762/0.476 @32/64/128k, undercount-biased errors — see ERRATUM above). §0 caveat CLOSED
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
- live (11:19): cwe@32k 0.600 (n=5, interim) — the harder aggregation task discriminates
  at SHORT context (counting is 0.95 there): wrong-word misses (marsh→savanna), not
  just undercounts. Sharper §8 regression instrument than counting, as intended.
  Lanes: gt128k 82/216 (semantic 512k column ~done), nolima 39/54, cwe 5/36.
- live (12:25): multihop@256k 1.000 (3/3, 232k-ptok prompts) — 2-hop reasoning ALSO
  holds zero-shot at 2x nominal. gt128k 90/216 (semantic+niah columns complete at
  1.000; multihop running), nolima 42/54, cwe 7/36. All watchers verified.
- live (14:51): multihop@512k 1.000 first cells (463k-ptok prompts) — 2-hop reasoning
  extrapolates zero-shot to 4x nominal. gt128k 100/216 (niah/semantic/multihop columns
  effectively complete at 1.000; abstain+counting+niah_multi remain), nolima 48/54,
  cwe 12/36 (@64k cells running).
- 2026-08-15 16:30 REVIEW ACTIONS (docs/review-glm53-verification.md): (1) stage6
  auto-train KILLED + replaced by approval-gated v2 (verified: 21-step corpus,
  +57pt winner floor — unconditional training was indefensible); (2) stage4 sweep
  trimmed/repowered 405→~83 cells (counting+cwe focused, 5 reps, +64k harm-check,
  yarn4→probe); (3) counting-mechanism downgraded to live hypothesis + errata in
  STATUS/phase-3 (12/17 k−1, one over-count); (4) >128k grid reordered counting-first
  (decision data 6-8h earlier); (5) E2 forensics lane launched (17 miss instances ×
  greedy/enumeration — adjudicates dilution vs decode-strategy). Reviewer errors noted:
  LBv2-512k pool is 44 not ≤3; marker-adjacency impossible (min gap 2.3k tok).
  Open follow-ups: NoLiMa id-pinned variant; corpus↔eval family circularity → §8 must
  corroborate counting/cwe gains on NoLiMa/LQA/LBv2; CI-prefix caveat; export
  rehearsal (C8) queued behind grid (no schedule cost).
- 2026-08-15 16:40 PLAN-ALIGNMENT (owner commit 4c875bc): §4 arms officially repurposed to
  zero-training treatment for criterion 7 (extrapolation rescue dead — stock 1.000 to
  512k). stage4 → v2.1: weak-axis primary grids (counting/cwe/nolima @128k+256k ×5,
  cell-seed-PAIRED vs stock — NoLiMa instances deterministic per cell so paired reads
  are valid; cross-length trends still are not), niah@64k harm check, 512k extension
  only on ≥+10pt paired signal, qk bracket {4.3,5.0}, yarn4 control now includes
  weak-axis cells (can falsify the inertness prediction). If a qk arm fixes weak axes
  harmlessly → deployment artifact = stock weights + config re-convert (no LoRA/merge —
  cheapest §11 path); else §7 training carries criterion 7 (approval-gated stage6-v2).
  PLAN §4 text erratum applied (12/17 k−1, not "every" off-by-one). First decision data
  landing: counting@192k = 0.0 (2 cells) — weak axis confirmed beyond nominal.
  e2 forensics: 1/34 rows so far. All 10 watchers verified.
- E2 forensics interim (17:30, 3 miss-instances × 2 conds): greedy REPRODUCES the misses
  (not sampling noise) and explicit enumeration does NOT fix them → evidence AGAINST
  decode-strategy hypothesis, FOR retrieval/attention (dilution). Sharpest case:
  want 12, model enumerated 11 entries then answered 9 — one marker never surfaced.
  Caveat: n=3 pairs, enum-token-counting is crude; full 34-row run continues.
- E2 VERDICT (22:07, complete n=17 paired): enum-fix-rate 0.35 / greedy-miss 0.71 →
  script verdict "inconclusive", but the PAIRED anatomy is informative:
  * 11/17 BOTH-MISS — explicit entry-by-entry enumeration does NOT rescue; in 5 pairs
    the model listed FEWER entries than k even when told to enumerate methodically
    (e.g. want 12, listed 11 twice under both conditions) → markers genuinely not
    surfacing = RETRIEVAL-SIDE (dilution) evidence.
  * Several miss cells resolved under BOTH conditions (5 both-correct + 1 FIXED) →
  original capability-mode misses were partly stochastic decode variance.
  * Note: 'numbers_listed' counts any digit token (entry numbers ≈ marker count k, so
    correct enumerations list ~k-2k digits) — crude, interpreted with that caveat.
  NET: mechanism picture = mixed retrieval-undercount (dominant, enumeration-resistant)
  + decode variance (minor). qk hypothesis remains the live causal test (§4 v2.1).
  phase-3/PLAN language already downgraded — consistent with this.
- cwe grid COMPLETE (22:58): 0.778@32k / 0.889@64k / 0.889@128k / 0.778@256k (n=9) —
  non-monotone (shallow dip at 32k, recovery, decline at 256k), clearly NOT a length
  cliff; consistent with difficulty-driven errors, not context-collapse. §4 sweep's
  cwe@128k/256k cells pair directly against these stock values.
- 2026-08-15 23:50 AUDIT ACTIONS (independent review, verified then actioned):
  (F4 FIXED) stage4 weak_signal was dead code — per-task `len(pairs)>=10` unreachable
  (stock has 3 reps → max 6 pairs); now pooled counting+cwe, floor 5, both deltas
  logged; watcher replaced. (F3) k-matched counting grid LAUNCHED (strata k=6/11 ×
  32k-256k × 5 reps × capability+greedy; builder verified 20 seeds — marker stem
  collision with question found+fixed). (F1/F2/F5 ERRATA) interim table REGENERATED
  from disk (no more hand-typed cells): counting non-monotone >128k (0.667 flat
  192-384k vs 0.476@128k), E2 headline corrected (greedy fixed 5/17 — ~30% sampling
  share; 11/17 enumeration-resistant), nolima@512k=0.444(n=9), cwe@64k=0.889,
  counting@192k "0.0 (2 cells)" flash-read superseded. §5 row → v1.1 final numbers.
  (minor) phase0 3:1 pattern erratum + 25% padding note; stage6 winner lists →
  bracket {4.3,5.0}; stage7 short-regression +cwe@32k, header corrected; PLAN
  nolima number fixed. (F6 PENDING) corroborator powering decision (LBv2/LQA 10-15
  draws at 128/256k vs downgrade claim) — queued with the train1 approval evidence.
  Mechanism reframed everywhere: aggregation fragility (k-dependent, partly
  stochastic, no clean length trend); §4a = hypothesis exploration. E2 verdict rule
  now 3-way. train1 stays approval-gated; approval evidence = k-matched grid +
  E2-complete + §4 paired reads.
- 2026-08-16 08:45 MILESTONE: §3 COMPLETE (both grids). Final >128k: retrieval/reasoning
  1.000 everywhere (niah/niah_multi/multihop/semantic/abstain, all lengths through 512k);
  counting 0.667/0.667/0.667/0.222 — non-monotone plateau then real drop only at 4×
  nominal. Decision rule: 100% retention → strengthen-qualify-deploy formally taken
  (phase-3 report updated, commit f8f9f5c). k-matched grid COMPLETE (80/80):
  k-controlled counting shows NO clean monotone length trend at matched difficulty
  (k=6: 1.0/1.0/0.6/0.8 cap, 1.0/1.0/0.6/1.0 greedy; k=11: 1.0/1.0/0.8/0.8 cap,
  0.8/0.8/0.6/0.6 greedy) — capability≈greedy at matched k (small stochastic share),
  and 128k is a LOCAL dip in both strata while 256k partially recovers. Confirms
  audit reframing: aggregation fragility (k-dependent, noisy, weakly length-linked);
  the 512k=0.222 main-grid cell likely overstates the length effect (k-draws there).
  §4 sweep is now the sole pending decision input for train1 approval.
- 2026-08-16 09:50: LQA "stall" was a false alarm — the grid COMPLETED at 15 rows
  (5 ctx × 3 reps; my /45 counter used the wrong denominator). Final: LongCodeQA
  0.67/0.67/0.67/1.00/0.67 @32k-512k (repo-scale coding holds at 512k — GOAL crit 5
  baseline in reach; n=3 → wide CI, powering decision still pending). Lane now on
  synth3 (148/189; conflicts/set_intersect/chronology landing). §4 sweep next after
  stage3, then the train1 approval decision.
- 2026-08-16 11:20 SUITE LANE COMPLETE — full stock baseline (all suites on disk,
  snapshot regenerated). New: conflicts 1.000 everywhere; set_intersect 0.96-1.00;
  chronology 0.95-1.00; infb_kv 1.000 (UUID KV @128k+256k); infb_codedebug 0.67@128k →
  0.00@256k (real degradation axis found! repo-scale bug-finding collapses at 256k);
  infb_bookmc 0.33→0.00. LBv2 0.33-1.00 (n=3, wide). Weak-axis shortlist for §4/§7:
  counting/cwe (aggregation), infb_codedebug+bookmc @256k (long-doc code/reasoning),
  NoLiMa (per-length ~0.22-0.67). GPU chain: stage3 (agentmem+PPL) → §4 sweep →
  approval decision.

## 2026-08-16 18:49 — GLM-5.3 adversarial audit of the armed chain; all FIX-NOW findings actioned

Headless `pi -p --provider z.ai --model glm-5.3` review (`logs/adversarial-review-glm53.log`)
of setup/progress/upcoming phases, every claim re-verified on disk before acting. Notable
verified findings + fixes (commit follows):
- **F-1.1** the §4 ≥+10pt pooled gate required a perfect 12/12 (stock 3 reps, 10/12 pairs
  ceiling-bound; recomputed from disk) → stage4 now enriches stock to 5 paired reps first
  (`stock_weak5.jsonl`, n=20 pairs).
- **F-1.2** niah@64k harm check was logged but consumed by nothing → now vetoes both the
  512k extension (stage4) and the qk training override (stage6).
- **F-1.3** arm grids were marked `.done` even on failure (permanent silent skip on
  re-arm) → markers now conditional on expected rows (47+20 / 9), `.failed` otherwise.
- **F-2.2** stage6 stock globs dropped `stock_cwe`/`suite_nolima` → override + winner-info
  now see the full weak-axis stock evidence.
- **F-4.1** snapshot §3 table silently empty for N runs (summarize.py took one path;
  stderr swallowed) → summarize accepts multiple paths; stderr visible; verified live.
- **F-5.1/5.2** arms/run1 had no infbench/nolima/LQA/niah_multi coverage (criterion 7
  adjudicated on in-family tasks only) → stage4 arm grids add infb_codedebug+infb_bookmc
  ×3; stage7 adds nolima@128/256k, LQA@128k, niah_multi corroborator lanes.
- **F-5.3** stage8 guard was a fixed 3pt on n=3 cells (near-certain noise BLOCK) → now
  `max(3pts, stock-cell CI)` (T975 actually used).
- **F-7.1** export audit never asserted 512k context or artifact size →
  `muse-glimmer.context_length = 524288` grep + ≤19GiB gate in stage8.
- **F-7.2** stage9 parity read nan→PASS on missing GGUF cells → missing/incomplete = FAIL.
- Dashboard: marker prefixes {blocked,failed,skipped} now distinct states; arm grids +
  run1_short + stock_weak5 in GRIDS; run1 total 162→177 (corroborators).
- MONITOR items (stall timers, G3 floor, budget pinning, phase-4 doc staleness) logged in
  the review; phase-4 doc got an ERRATUM pointer to v2.1.
Stage4 watcher was killed mid-arm-switch (vLLM cold-starting qk4.3, no eval rows, no
markers) — re-armed with v2.2 script; it will serve stock first for enrichment.

### Addendum 19:00 — mislabeled-enrichment race caught & fixed (post-audit hardening)

While re-arming stage4 with the audit fixes, two chained bugs briefly launched the
stock-label enrichment against the **qk4.3** arm (leftover vLLM cold start from the
killed watcher): (1) the enrichment guard only checked *reachability*, not the served
model; (2) the enrichment block was inserted before `serve_arm`'s definition
(undefined → instant 127 → "ERROR" logged but execution continued). Both caught within
seconds; **zero rows written** (`stock_weak5.jsonl` never created; run_eval killed
pre-first-cell twice, file verified absent). Fixes: `serve_arm` now polls until the
served `/v1/models` root matches the requested arm path; enrichment pre-check verifies
root == `/arms/stock-524k`; block moved below function definitions. Gotcha recorded:
`pkill -f` in a compound command self-matches the wrapper shell's cmdline (bracket
trick insufficient when the literal appears elsewhere in the same command) — split
kill/start into separate tool calls.

### Addendum 19:40 — idle-watcher hardening (audit MONITOR items); enrichment pairing verified

- stage5/6/7 watchers patched & re-armed (all were idle in wait loops; kill-by-pid +
  re-arm): stage5 writes a `dry-run FAILED` marker + blocked on verify-env failure
  (was: bare exit → stage6 waited forever); stage6 G1/G3 failures now loud (blocked),
  G2 also checks the dev container for batch_generate, and the approval gate refuses
  an empty `train1.approved` or a non-clean stage4 completion (audit 2.1/3.2/3.4);
  stage7 detects trainer-crash (launched ≥15 min ago, trainer gone, no adapter) →
  blocked instead of waiting forever (audit 3.1). Bash gotcha found: a multi-line
  `{ … }` group after `<<'PY' ||` breaks heredoc parsing — keep such groups one-line.
- Dashboard: any marker containing FAILED renders non-done (was prefix-match only).
- **Enrichment pairing verified from disk**: stock_weak5 vs original stock rows share
  6 (task,ctx,rep) cells → 0/6 cell_seed mismatches (identical instances, paired
  reads valid); 5/6 score agreement (one stochastic flip at temp 1.0 — expected;
  enrichment re-measures under fresh sampling, pooled deltas unaffected).
- Stage4 enrichment in flight: 8/20 (counting@256k exact on all reps so far).

### Erratum (2026-08-16 22:00) — depth-0.5 weak-axis stock baselines revised by n=5 enrichment

The §4 gate pairing claim above ("cwe@128k/256k cells pair directly against these stock
values", n=3/n=9 depth-averaged) is superseded at depth 0.5 by `stock_weak5.jsonl`
(2026-08-16, seed-paired with arm reps, 0/6 cell_seed mismatches):
- cwe@128k 1.000 (n=3) → **1.000 (n=5)**; cwe@256k 1.000 (n=3) → **0.600 (n=5)**
- counting@128k 0.333 (n=3) → **0.800 (n=5)**; counting@256k 1.000 (n=3) → **0.600 (n=5)**
The n=3 draws were optimistic on 2 of 4 cells (and pessimistic on counting@128k).
Implications: (a) the §4 pooled gate input is now 15/20 = 0.750 (arm needs 17/20);
(b) "cwe@256k clean 1.000" is no longer supportable — cwe has real depth-0.5 headroom
at 256k, making it a genuine weak axis for criterion 7; (c) depth-averaged doc tables
(phase3 report) are unaffected (they pool depths 0.0/0.5/1.0, n=9). Snapshot regenerated.

## 2026-08-17 06:05 — §4 COMPLETE (negative); stage5 green; chain at train1 approval gate

- **§4 verdict (all 3 arms, paired, evidence in docs/phase4-evidence.md)**: NO zero-training
  rescue. qk4.3 pooled −5 pts (counting worse, cwe@256k +40 in isolation, p=0.5);
  qk5.0 pooled −20 pts with dose-dependent counting harm (3.87→4.3→5.0 gives
  4/5→2/5→0/5 @128k — qk sharpening actively destroys aggregation); yarn4 inert
  (position is not the lever). Harm checks clean (niah 5/5 both arms). No 512k
  extension triggered; no qk override → if train1 runs, it trains on STOCK knobs.
- **Mechanism update**: attention-temperature hypothesis for aggregation fragility is
  now WEAKENED (sharper attention ≠ better pooling; possibly worse). Training with
  aggregation-targeted SFT (§7 corpus) is the remaining lever for criterion 7.
- stage5: dev refresh + verify-env PASSED; trainer dry-run initially failed on two
  real bugs — LORA_TARGET_RE missed the `model.` wrapper prefix (peft fullmatch) and
  triton couldn't find libcuda (stale ldconfig cache vs compat/lib) — both fixed
  (commit 06e8ae9; TRITON_LIBCUDA_PATH env in stage5/stage6 launches). Dry-run green:
  forward+backward OK, 58.8M trainable / 208 modules. GPU free, training-ready.
- stage6: G1 (dry-run OK) passed; blocked at `logs/train1.approved` as designed.
  Approval brief: §4 negative (training carries criterion 7 alone) + corpus math
  (174 rows @131k ≈ 21 optimizer steps at mb1×accum8; winner-detection floor ~+57pt).
- stage7–9 armed behind train1.

### Erratum (2026-08-17, goal afe6584b) — InfBench ">128k cliff" was a length-cache artifact

The suite-lane finding "infb_codedebug 0.667@128k→0.000@256k, infb_bookmc 0.33→0.00"
(STATUS 11:20 entry, suite_infbench.jsonl) is **retracted as stated**: target_ctx did
not control actual prompt length. `infbench_lengths.json` (v1) was keyed by bare
dataset id — ids collide across the three ∞Bench subsets, so each task's calibration
pass overwrote the others (final pass = kv's ~124k tables). True lengths (v3,
task/id-keyed, sanity-asserted): the "128k" cells actually served 143k/223k/266k
tokens, the "256k" cells 131k–163k. Re-attributed (infb_forensics.jsonl): bookmc is
non-monotone in true length (difficulty-dominated, no cliff); codedebug 160–200k is
0/9 across stock+arms (the one possibly-real zone — honest-length band runs in
flight: 100–140k 2/3, 140–170k 0/3, 170–200k 0/3 so far). kv never varied length.
docs/perf-gt128k-quantification.md is the authoritative write-up (regenerates from
disk via scripts/perf_doc.py). Final-report §2 and deliverables #4 wording to be
updated when bands complete.

## 2026-08-18 22:50 — goal d56ed95d (RULER+MRCR): BLOCKED-STOP on GPU lanes (partial)

Delivered: MRCR v2 plugin (official GCS dataset, Apache-2.0, eval-only; mrcr2/mrcr4,
strict exact-match + lenient flags; selftest + harness-integration PASS, commit ddaa880);
RULER plugin from yesterday stands (selftest PASS). Dashboard tracks both (workload
ledger + grids + /api/workloads verified). Lane chain armed: ruler (48 cells) →
mrcr (12 cells) ∥ synth_512k remainder ∥ bookmc true bands; all hardened after the
error-row incident (orphans wrote 9 error rows with vLLM down at 19:46; markers were
falsely satisfied by wc -l — lanes now count '"error": null' only; dashboard
readJSONL/countRows exclude error rows; synth purged+re-chained; run_eval resume
verified error-skipping; commit 8f1f149).
BLOCKER: vLLM cold start OOM-killed 3× (exact ~5:00 after shard-load begins;
State.OOMKilled=true). GPU holds: ~37 GiB by co-running autoresearch work + 80.3 GiB
vLLM@0.66 budget > 121.69 unified; host RAM during load spikes below the 55.46 GiB
shread. Temporary compose override at .devcontainer/docker-compose.override.yml
(util 0.66, gitignored, REMOVE when GPU free; verified file untouched).
Self-heal: ruler lane now starts+polls vLLM itself (up -d + 5-min poll), so the whole
chain fires automatically when the co-running work releases memory. Fixed ruler↔synth
wait deadlock (synth waits ruler-done; ruler now gates on server, not on synth).
train1 approval gate untouched (stage6 watcher alive, blocked as designed).
UNMET on goal: RULER 48 cells, MRCR 12 cells, synth_512k, bookmc bands, their doc
rows — all pending GPU availability. Goal stays active (not complete).
