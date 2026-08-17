# Deliverables & Completion Audit — GOAL.md ↔ PLAN §14

Working manifest: every GOAL success criterion and PLAN §14 deliverable mapped to its
evidence artifact and status. The goal is complete only when every row reads ✅ with
fresh evidence. Updated as phases close.

## PLAN §14 deliverables

| # | Deliverable | Evidence | Status |
|---|---|---|---|
| 1 | Reproducible environment spec (llama.cpp build, CUDA, transformers v5 pin) | `docs/environment.md`, `.devcontainer/*`, `pyproject.toml`, `scripts/verify-env.sh` | ✅ |
| 2 | Training-data generation scripts (teacher pipeline + verification passes) | `src/muse_longctx/corpus/` (pi_teacher + 5 generators + serialize + mixer + batch driver); provenance `outputs/corpus/pi_calls.jsonl`; corpus v1.1 final: 259 raw / 174 @131k / 10.87M tok (`outputs/corpus/train_v1/manifest.json`) | ✅ |
| 3 | Context/position sampler | `src/muse_longctx/position_sampler.py` (selftested) + `docs/phase6-position-sampler.md` | ✅ |
| 4 | QLoRA training configuration (+ BF16-LoRA fallback) | `src/muse_longctx/train_qlora.py`, `docs/phase7-qlora-trainer.md` | ✅ config; dry-run GREEN 2026-08-17 (fwd+bwd OK, 58.8M trainable/208 modules; commit 06e8ae9); ⏳ first run approval-gated (`logs/train1.approved`) |
| 5 | Benchmark harness with pinned sampling/template controls | `evals/harness/` (client contract, 6+5 task modules, runner, Parquet schema, summarize/retention/compare); plugins live-verified 5/5 | ✅ |
| 6 | Baseline, zero-shot (qk, YaRN), and ablation results | baseline `docs/phase3-stock-baseline.md` (594 cells, retrieval 1.000 →512k, decision rule met); zero-shot `docs/phase4-evidence.md` (paired, 3 arms, harm-checked: qk negative w/ dose-dependent counting harm, yarn4 inert); snapshot auto-incl. enriched weak-axis baselines | ✅ baseline+zero-shot; ⏳ §9 ablation (post-train) |
| 7 | Merged Hugging Face checkpoint | `scripts/export_pipeline.sh` stage 1 (merge, vision-safe asserts) | ⏳ awaits winning adapter (§7→§8) |
| 8 | ~17 GB K-Quant GGUF (long-context imatrix, verified metadata, DFlash-compatible) | `scripts/export_pipeline.sh` stages 3–7 (metadata audit greps, imatrix from corpus, Q4_K_M, dflash+mmproj smoke) | ⏳ staged; awaits §7 |
| 9 | 32 GB / 512k deployment configuration (RTX 5090) | `docs/phase12-deployment-config.md` (memory table from measured components, KV order, qualification checklist) | ✅ config; ⏳ on-device qualification awaits hardware |
| 10 | Concise report (effective context, memory, throughput, regressions, limits) | to be written from `docs/results-snapshot.md` + phase reports | ⏳ awaits §8+ results |

## GOAL.md success criteria → required evidence

| # | Criterion | Evidence needed | Status |
|---|---|---|---|
| 1 | Loads & runs at 512k within 32 GB VRAM | §12 checklist items 2–3 (buffer totals + iSWA) on the 5090; analytic fit already: 22.13 GB @512k F16 (`docs/phase0-...md`) | ⏳ hardware |
| 2 | Controlled degradation (not collapse) 128k→512k | §3 >128k grid COMPLETE: every retrieval/reasoning task 100.0% retention at 192–512k (n=9/cell; `docs/phase3-stock-baseline.md`); counting degrades but is weak-everywhere (not length-collapse) | ✅ stock; ⏳ §8 re-check post-train |
| 3 | Strong retrieval across full context incl. start/end | depth 0%/100% @≥256k: 63/63 hits (niah/niah_multi/multihop/agentmem, n=9/cell/task/depth; recomputed from JSONLs 2026-08-17) | ✅ stock; ⏳ §8 re-check |
| 4 | Multi-hop over separated evidence | multihop 120/120 across 32k–512k (all depths, n=21 ≤128k / n=9 >128k); infb_kv 6/6; infb_codedebug 0.67@128k→0.00@256k is the genuine axis (weak-axis shortlist) | ✅ stock; ⏳ §8 re-check |
| 5 | Repo-scale coding maintained/improved at 256k–512k | LQA stock: 3/3 @256k, 2/3 @512k (n=3, wide CI — powering decision F6 pending); no collapse; §8 vs trained pending | ✅ stock (n=3 caveat); ⏳ §8 |
| 6 | Minimal regression at ≤128k | stock baseline in place (§3 378-cell ≤128k grid); arm/regression reads wired: stage7 short grid (54 cells) + stage8 CI-aware guard (audit F-5.3) | ⏳ post-train |
| 7 | Materially better than stock at same length | §4 zero-shot lever EXHAUSTED (negative, paired evidence); remaining lever = §7 training (approval-gated); §8 compare + corroborators armed | ⏳ post-training (sole open lever) |
| 8 | Retains 17 GB artifact practicality | §11 quant parity mini-suite + §12 throughput w/ DFlash | ⏳ |

## Hard constraints honored (ongoing)

- Eval repos excluded from training: `data/exclusions/eval_repos.json` + fail-closed gates
  (3 production rejections logged); NoLiMa/LBv2/∞Bench/LQA used eval-only.
- Sampling/template controls pinned in every call (client contract; logged per row).
- Per-phase commits: see `git log` (one phase per commit message prefix).
- Environment changes via pyproject only; verify-env after rebuilds (stage5 re-checks).

## Known pending decisions (from evidence, not blockers)

- ~~§4 winner (qk value) feeds §7 `--config-override`~~ RESOLVED 2026-08-17: no override — §4 negative (qk4.3 −5 pts pooled; qk5.0 −20 with dose-dependent counting harm; yarn4 inert); train1 would use stock knobs.
- Decision rule (§3): ≥85% retrieval retention at 256k+ → training optional/targeted.
- If QLoRA lift weak over zero-shot → BF16-LoRA fallback arm (pre-staged).
