# Project Docs — Muse Glimmer 30B 512k Context Adaptation

Index of working artifacts. Short-form progress ledger: `STATUS.md`. Completion audit:
`docs/deliverables.md`. Human-oriented project docs: `GOAL.md` → `PLAN.md` → `MODEL.md`;
environment: `CONTRIBUTING.md`; agent rules: `AGENTS.md`.

| Phase | Report | Status |
|---|---|---|
| §0 spike | [phase0-compat-memory-spike.md](phase0-compat-memory-spike.md) | complete — gates passed |
| §1 environment | [environment.md](environment.md) | complete — pinned |
| §2 eval harness | [phase2-eval-harness.md](phase2-eval-harness.md) | core + NoLiMa + LBv2 + LongCodeQA + ∞Bench + agentmem (all live-verified); RULER/HELMET/LongSWE sequenced post-§3 |
| §4 zero-shot arms | [phase4-zeroshot-arms.md](phase4-zeroshot-arms.md) | prep complete; runs queued (GPU) |
| §6 position sampler | [phase6-position-sampler.md](phase6-position-sampler.md) | complete |
| §7 QLoRA trainer | [phase7-qlora-trainer.md](phase7-qlora-trainer.md) | skeleton complete + §9 scope flag; `--dry-run` awaits free GPU |
| §5 corpus | [phase5-corpus-plan.md](phase5-corpus-plan.md) | **pipeline complete** — pi-headless GLM-5.2 teacher (owner directive); all 5 components validated; batch scale-up running |
| §3 baseline | (in progress — `outputs/eval/stock_vllm_le128k.jsonl`, `logs/eval-stock-le128k.log`) | running; overnight → suite → stage3–stage9 chains armed |
| §12 deployment config | [phase12-deployment-config.md](phase12-deployment-config.md) | config complete; qualification awaits RTX 5090 |
| Deliverables audit | [deliverables.md](deliverables.md) | every GOAL criterion ↔ evidence artifact + status |

## Queued automation

GPU chain: overnight (§3 grids + caveat) → suite (community suites) → stage3 (agentmem +
PPL) → stage4 (§4 arm sweep) → stage5 (dev refresh + §7 dry-run) → stage6 (§7 run1
launch, gated) → stage7 (merge + §8 eval + regression subset + PPL) → stage8 (§11 export
chain w/ regression guard GO/BLOCK → Q4_K_M artifact) → stage9 (§11 quant-parity
mini-suite: BF16-merged vs GGUF @128k greedy + DFlash acceptance check).
Host-side: corpus batch (§5 scale-up) running concurrently.
