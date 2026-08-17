# Final Report — Muse Glimmer 30B: 512k Context on 32 GB (PLAN §14 #10)

Status: **DRAFT — sections fill as phases close; every number disk-computed, never
hand-typed.** Sources: `docs/results-snapshot.md` (auto), phase reports, progress
markers. Pending sections are marked ⏳ with their gate.

## 1. Environment & reproducibility

Pinned stack (NGC PyTorch base, CUDA 12.9/13.x dual-path, transformers v5.15 pin,
llama.cpp b10428 sm_120/121): `docs/environment.md`, `.devcontainer/*`,
`scripts/verify-env.sh` (health gate, ALL CHECKS PASSED after every rebuild; last
verified 2026-08-17 by stage5). Cold-start and cache figures: CONTRIBUTING.md §7.

## 2. Effective context — stock baseline (§3)

- **Retrieval/reasoning: 1.000 at every length through 512k** (niah, niah_multi,
  multihop, semantic, abstain; n=9/cell; 594 cells total; `docs/phase3-stock-baseline.md`).
- Edge-of-context (depth 0%/100%) @≥256k: **63/63**. No primacy/recency effect.
- Decision rule (≥85% retention) met with maximal margin → strengthen-qualify-deploy.
- **Weak axes (GOAL criterion 7 target)**: aggregation family — counting
  0.95@32k→0.22@512k (depth-avg), cwe non-monotone 0.6–0.9, nolima 0.22–0.67
  non-monotone, infb_codedebug 0.67@128k→0.00@256k, infb_bookmc 0.33→0.00.
  Mechanism: aggregation fragility (k-dependent), NOT length collapse — no clean
  length trend at matched k; ~30% stochastic share (E2 forensics); retrieval-side
  core (11/17 enumeration-resistant).

## 3. Zero-shot arms (§4) — negative, decisive

Paired reads (cell_seed-identical instances; `docs/phase4-evidence.md`):

| arm | pooled counting+cwe Δ | counting@128k | harm niah@64k | verdict |
|---|---|---|---|---|
| qk4.3 | **−5 pts** (n=20) | 2/5 vs stock 4/5 | 5/5 clean | no signal |
| qk5.0 | **−20 pts** (n=20) | 0/5 vs 4/5 | 5/5 clean | harmful (dose-dependent) |
| yarn4 | −17 pts (n=6, control) | 2/3 = stock | — | inert as predicted |

Attention-temperature sharpening *degrades* aggregation monotonically
(3.87→4.3→5.0 ⇒ 4/5→2/5→0/5 @128k); positional rescaling is inert. Enriched
(n=5, seed-paired) stock baselines: counting@128k 0.80, counting@256k 0.60,
cwe@128k 1.00, cwe@256k 0.60.

## 4. Training (§7) — ⏳ approval-gated

Trainer verified end-to-end (dry-run: forward+backward OK; 58.8M trainable across
208 modules; QLoRA NF4 r32 all-scope; corpus v1.1: 174 rows @131k, 10.87M tokens,
21 optimizer steps at mb1×accum8). Winner-detection floor ~+57pt at n=3 — run1 is
evidence-gathering, not a final claim. Gate: `logs/train1.approved`.

## 5. §8 evaluation of run1 — ⏳ (stage7 armed)

Decision subset (niah/semantic/multihop/abstain/counting/cwe @128k–512k ×3 reps,
162 cells) + ≤32k short-regression (18) + corroborators (NoLiMa 128k/256k, LQA@128k,
niah_multi — audit F-5.2). Regression guard: CI-aware (max(3pts, stock CI)), blocks
export on any beyond-CI ≤128k drop.

## 6. Artifact (§11) — ⏳ (stage8/9 armed)

Export chain rehearsed: merge → convert → metadata audit (now asserting
`context_length = 524288`, block/window/pattern/θ=500k/softcap/template) →
imatrix (corpus-calibrated) → Q4_K_M (~17 GB, size-gated ≤19 GiB) → DFlash/mmproj
smoke. Quant parity: BF16 vs GGUF paired reads, missing cells = FAIL (not PASS).

## 7. Deployment fit (§12) — ⏳ hardware

RTX 5090 32 GB qualification checklist + memory table: `docs/phase12-deployment-config.md`.
Analytic fit at 512k: 22.13 GB F16 KV (phase0) with iSWA order options. On-device
throughput + buffer totals pending hardware arrival.

## 8. Throughput & limits

GB10 measured (lower bounds; 5090 ≈ 6× bandwidth): prefill ~600–2700 tok/s effective
(prefix caching), niah@32k 30 s wall → @512k ~1.0–1.4 ks ttft. GGUF parity latency +
DFlash acceptance: pending stage9. Known limits: aggregation family remains weak
absent a training effect; corroborator powering (LBv2/LQA n=3) is a recorded caveat (F6).

## 9. Constraint compliance

Eval-only suite licenses honored; exclusion list fail-closed (3 rejections logged);
sampling contract pinned per row (temp 1.0/top-p 0.95/top-k 64, low reasoning,
`message.content` only; greedy only for parity/confirmation); no eval data in
training corpus; per-phase commits; `.env`/cache/weights never committed.
