# Project Docs — Muse Glimmer 30B 512k Context Adaptation

Index of working artifacts. Human-oriented project docs: `GOAL.md` → `PLAN.md` → `MODEL.md`;
environment: `CONTRIBUTING.md`; agent rules: `AGENTS.md`.

| Phase | Report | Status |
|---|---|---|
| §0 spike | [phase0-compat-memory-spike.md](phase0-compat-memory-spike.md) | complete — gates passed |
| §1 environment | [environment.md](environment.md) | complete — pinned |
| §2 eval harness | [phase2-eval-harness.md](phase2-eval-harness.md) | core complete; NoLiMa + LongBench v2 integrated; RULER/∞Bench/HELMET sequenced post-§3 |
| §4 zero-shot arms | [phase4-zeroshot-arms.md](phase4-zeroshot-arms.md) | prep complete; runs queued (GPU) |
| §6 position sampler | [phase6-position-sampler.md](phase6-position-sampler.md) | complete |
| §7 QLoRA trainer | [phase7-qlora-trainer.md](phase7-qlora-trainer.md) | skeleton complete + §9 scope flag; `--dry-run` awaits free GPU |
| §5 corpus | [phase5-corpus-plan.md](phase5-corpus-plan.md) | **pipeline complete** — pi-headless GLM-5.2 teacher (owner directive); all 5 components validated; batch scale-up running |
| §3 baseline | (in progress — `outputs/eval/stock_vllm_le128k.jsonl`, `logs/eval-stock-le128k.log`) | running; overnight → suite → stage3 → stage4 chains armed |

| §12 deployment config | [phase12-deployment-config.md](phase12-deployment-config.md) | config complete; qualification awaits RTX 5090 |

## Queued automation

GPU chain: overnight (§3 grids + caveat) → suite (community suites) → stage3 (agentmem +
PPL) → stage4 (§4 arm sweep: qk4.3/qk5.0/qk4.1/qk4.6/yarn4 × niah/semantic @ 128k–512k) →
stage5 (dev refresh + verify-env, GPU freed, §7 dry-run → training-ready).
Host-side: corpus batch (§5 scale-up) running concurrently.
