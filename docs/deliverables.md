# Deliverables & Completion Audit — GOAL.md ↔ PLAN §14

Working manifest: every GOAL success criterion and PLAN §14 deliverable mapped to its
evidence artifact and status. The goal is complete only when every row reads ✅ with
fresh evidence. Updated as phases close.

## PLAN §14 deliverables

| # | Deliverable | Evidence | Status |
|---|---|---|---|
| 1 | Reproducible environment spec (llama.cpp build, CUDA, transformers v5 pin) | `docs/environment.md`, `.devcontainer/*`, `pyproject.toml`, `scripts/verify-env.sh` | ✅ |
| 2 | Training-data generation scripts (teacher pipeline + verification passes) | `src/muse_longctx/corpus/` (pi_teacher + 5 generators + serialize + mixer + batch driver); provenance `outputs/corpus/pi_calls.jsonl` | ✅ code; ⏳ corpus volume accruing (`batch_generate`) |
| 3 | Context/position sampler | `src/muse_longctx/position_sampler.py` (selftested) + `docs/phase6-position-sampler.md` | ✅ |
| 4 | QLoRA training configuration (+ BF16-LoRA fallback) | `src/muse_longctx/train_qlora.py`, `docs/phase7-qlora-trainer.md` (first-run command, fallback arm, dry-run gate) | ✅ config; ⏳ dry-run fires in stage5; first run pending GPU + corpus |
| 5 | Benchmark harness with pinned sampling/template controls | `evals/harness/` (client contract, 6+5 task modules, runner, Parquet schema, summarize/retention/compare); plugins live-verified 5/5 | ✅ |
| 6 | Baseline, zero-shot (qk, YaRN), and ablation results | `docs/results-snapshot.md` (auto); grids: §3 running, §4 queued (stage4), §9 configs pre-staged (`--lora-scope`) | ⏳ runs queued |
| 7 | Merged Hugging Face checkpoint | `scripts/export_pipeline.sh` stage 1 (merge, vision-safe asserts) | ⏳ awaits winning adapter (§7→§8) |
| 8 | ~17 GB K-Quant GGUF (long-context imatrix, verified metadata, DFlash-compatible) | `scripts/export_pipeline.sh` stages 3–7 (metadata audit greps, imatrix from corpus, Q4_K_M, dflash+mmproj smoke) | ⏳ staged; awaits §7 |
| 9 | 32 GB / 512k deployment configuration (RTX 5090) | `docs/phase12-deployment-config.md` (memory table from measured components, KV order, qualification checklist) | ✅ config; ⏳ on-device qualification awaits hardware |
| 10 | Concise report (effective context, memory, throughput, regressions, limits) | to be written from `docs/results-snapshot.md` + phase reports | ⏳ awaits §8+ results |

## GOAL.md success criteria → required evidence

| # | Criterion | Evidence needed | Status |
|---|---|---|---|
| 1 | Loads & runs at 512k within 32 GB VRAM | §12 checklist items 2–3 (buffer totals + iSWA) on the 5090; analytic fit already: 22.13 GB @512k F16 (`docs/phase0-...md`) | ⏳ hardware |
| 2 | Controlled degradation (not collapse) 128k→512k | §3 >128k grid (stock) + §8 compare — retention curves w/ CIs (`retention.py`) | ⏳ grid queued tonight |
| 3 | Strong retrieval across full context incl. start/end | depths 0%/100% rows across niah/niah_multi/multihop/agentmem/nolima @256k–512k | ⏳ |
| 4 | Multi-hop over separated evidence | multihop + infb tasks ≥ thresholds TBD from §3 stock baseline (report defines "meaningful") | ⏳ |
| 5 | Repo-scale coding maintained/improved at 256k–512k | longcodeqa 256k/512k buckets, §8 vs stock | ⏳ (LQA pools: 65/47) |
| 6 | Minimal regression at ≤128k | short replay eval + ≤128k grid across arms (`compare.py` Δ) | ⏳ |
| 7 | Materially better than stock at same length | §8 compare: trained vs stock with significance markers | ⏳ post-training |
| 8 | Retains 17 GB artifact practicality | §11 quant parity mini-suite + §12 throughput w/ DFlash | ⏳ |

## Hard constraints honored (ongoing)

- Eval repos excluded from training: `data/exclusions/eval_repos.json` + fail-closed gates
  (3 production rejections logged); NoLiMa/LBv2/∞Bench/LQA used eval-only.
- Sampling/template controls pinned in every call (client contract; logged per row).
- Per-phase commits: see `git log` (one phase per commit message prefix).
- Environment changes via pyproject only; verify-env after rebuilds (stage5 re-checks).

## Known pending decisions (from evidence, not blockers)

- §4 winner (qk value) feeds §7 `--config-override` — decided by stage4 results.
- Decision rule (§3): ≥85% retrieval retention at 256k+ → training optional/targeted.
- If QLoRA lift weak over zero-shot → BF16-LoRA fallback arm (pre-staged).
