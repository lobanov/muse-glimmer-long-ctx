# §4 zero-shot arms — evidence pack (generated; do not hand-edit)

All reads paired on cell_seed-identical instances.
Per-cell n=5 → ±22pt binomial SE; read p before believing any single cell.

## arm qk4.3

| task@ctx | arm | stock | Δpts | discord (arm+/arm−) | McNemar p | flag |
|---|---|---|---|---|---|---|
| counting@128k | 2/5 | 4/5 | -40 | 0/2 | 0.50 | **greedy-confirm** |
| counting@256k | 2/5 | 3/5 | -20 | 0/1 | 1.00 |  |
| cwe@128k | 5/5 | 5/5 | +0 | 0/0 | 1.00 |  |
| cwe@256k | 5/5 | 3/5 | +40 | 2/0 | 0.50 | **greedy-confirm** |
| infb_bookmc@128k | 1/3 | 1/3 | +0 | 0/0 | 1.00 |  |
| infb_bookmc@256k | 0/3 | 0/3 | +0 | 0/0 | 1.00 |  |
| infb_codedebug@128k | 2/3 | 2/3 | +0 | 0/0 | 1.00 |  |
| infb_codedebug@256k | 0/3 | 0/3 | +0 | 0/0 | 1.00 |  |
| niah@64k | 3/3 | 3/3 | +0 | 0/0 | 1.00 |  |
| nolima@128k | 1/3 | 0/3 | +33 | 1/0 | 1.00 |  |
| nolima@256k | 0/3 | 0/3 | +0 | 0/0 | 1.00 |  |

- **gate pool**: arm 0.700 vs stock 0.750 (-5.0 pts, n=20; needs ≥ +10.0)
- **harm niah@64k**: 1.000 (n=3)

## arm qk5.0

| task@ctx | arm | stock | Δpts | discord (arm+/arm−) | McNemar p | flag |
|---|---|---|---|---|---|---|
| counting@128k | 0/5 | 4/5 | -80 | 0/4 | 0.12 | **greedy-confirm** |
| counting@256k | 2/5 | 3/5 | -20 | 1/2 | 1.00 | **greedy-confirm** |
| cwe@128k | 5/5 | 5/5 | +0 | 0/0 | 1.00 |  |
| cwe@256k | 4/5 | 3/5 | +20 | 2/1 | 1.00 | **greedy-confirm** |
| infb_bookmc@128k | 1/3 | 1/3 | +0 | 0/0 | 1.00 |  |
| infb_bookmc@256k | 0/3 | 0/3 | +0 | 0/0 | 1.00 |  |
| infb_codedebug@128k | 2/3 | 2/3 | +0 | 0/0 | 1.00 |  |
| infb_codedebug@256k | 0/3 | 0/3 | +0 | 0/0 | 1.00 |  |
| niah@64k | 3/3 | 3/3 | +0 | 0/0 | 1.00 |  |
| nolima@128k | 0/3 | 0/3 | +0 | 0/0 | 1.00 |  |
| nolima@256k | 0/3 | 0/3 | +0 | 0/0 | 1.00 |  |

- **gate pool**: arm 0.550 vs stock 0.750 (-20.0 pts, n=20; needs ≥ +10.0)
- **harm niah@64k**: 1.000 (n=3)

## arm yarn4

| task@ctx | arm | stock | Δpts | discord (arm+/arm−) | McNemar p | flag |
|---|---|---|---|---|---|---|
| counting@128k | 2/3 | 2/3 | +0 | 0/0 | 1.00 |  |
| cwe@128k | 2/3 | 3/3 | -33 | 0/1 | 1.00 |  |
| niah@128k | 3/3 | 3/3 | +0 | 0/0 | 1.00 |  |

- **gate pool**: arm 0.667 vs stock 0.833 (-16.7 pts, n=6; needs ≥ +10.0)
- harm: none

---
Rules: 512k extension fired only on pooled ≥ +10 pts AND harm ≥ 0.9; qk training override needs ≥2 cells with ≥5-rep wins > +15 pts AND harm ok. greedy-confirm cells must be re-run greedily before any approval cites them.
