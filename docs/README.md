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
| §7 QLoRA trainer | [phase7-qlora-trainer.md](phase7-qlora-trainer.md) | skeleton complete; blocked: §5 data + GPU |
| §5 corpus | [phase5-corpus-plan.md](phase5-corpus-plan.md) | design complete; **decision requested**: Z.ai key (Path A) or approve interim Path B |
| §3 baseline | (in progress — `outputs/eval/stock_vllm_le128k.jsonl`, `logs/eval-stock-le128k.log`) | running; overnight + suite queues armed |

Open blockers: **§5 corpus generation needs a Z.ai API key** (not in `.devcontainer/.env`;
interim Stack-v2-slice path verified viable, awaiting owner approval); §12 on-device
qualification awaits RTX 5090 hardware.
