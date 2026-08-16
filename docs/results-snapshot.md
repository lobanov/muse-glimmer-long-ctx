# Results snapshot — 2026-08-16 11:17:25

Data files present:
- `outputs/eval/e2_counting_forensics.jsonl` (34 rows)
- `outputs/eval/kmatched_counting.jsonl` (80 rows)
- `outputs/eval/parity_caveat_llamacpp.jsonl` (3 rows)
- `outputs/eval/parity_caveat_vllm.jsonl` (3 rows)
- `outputs/eval/plugin_smoke_agentmem.jsonl` (1 rows)
- `outputs/eval/plugin_smoke_infbench.jsonl` (1 rows)
- `outputs/eval/plugin_smoke_longbench_v2.jsonl` (1 rows)
- `outputs/eval/plugin_smoke_longcodeqa.jsonl` (1 rows)
- `outputs/eval/plugin_smoke_nolima.jsonl` (1 rows)
- `outputs/eval/smoke_vllm.jsonl` (4 rows)
- `outputs/eval/stock_cwe.jsonl` (36 rows)
- `outputs/eval/stock_vllm_gt128k.jsonl` (216 rows)
- `outputs/eval/stock_vllm_le128k.jsonl` (378 rows)
- `outputs/eval/suite_agentmem.jsonl` (2 rows)
- `outputs/eval/suite_infbench.jsonl` (18 rows)
- `outputs/eval/suite_longbench_v2.jsonl` (12 rows)
- `outputs/eval/suite_longcodeqa.jsonl` (15 rows)
- `outputs/eval/suite_nolima.jsonl` (54 rows)
- `outputs/eval/suite_synth3.jsonl` (189 rows)

## §3 stock baseline — score by task × ctx (mean ± 95% CI (n))
```
```
## §3 retention vs 128k + decision rule
```
# Retention analysis — stock (stock)
rows: 594; reference ctx = 128000

## stock
| task           |         32000 |         64000 |        128000 |        192000 |        256000 |        384000 |        512000 |
|---|---|---|---|---|---|---|---|
| abstain        | 100.0± 0.0 (21) | 100.0± 0.0 (21) | 100.0± 0.0 (21) | 100.0± 0.0 (9) | 100.0± 0.0 (9) | 100.0± 0.0 (9) | 100.0± 0.0 (9) |
| counting       |  95.2± 9.3 (21) |  76.2±18.7 (21) |  47.6±21.9 (21) |  66.7±32.7 (9) |  66.7±32.7 (9) |  66.7±32.7 (9) |  22.2±28.8 (9) |
| multihop       | 100.0± 0.0 (21) | 100.0± 0.0 (21) | 100.0± 0.0 (21) | 100.0± 0.0 (9) | 100.0± 0.0 (9) | 100.0± 0.0 (9) | 100.0± 0.0 (9) |
| niah           | 100.0± 0.0 (21) | 100.0± 0.0 (21) | 100.0± 0.0 (21) | 100.0± 0.0 (9) | 100.0± 0.0 (9) | 100.0± 0.0 (9) | 100.0± 0.0 (9) |
| niah_multi     | 100.0± 0.0 (21) | 100.0± 0.0 (21) | 100.0± 0.0 (21) | 100.0± 0.0 (9) | 100.0± 0.0 (9) | 100.0± 0.0 (9) | 100.0± 0.0 (9) |
| semantic       | 100.0± 0.0 (21) | 100.0± 0.0 (21) | 100.0± 0.0 (21) | 100.0± 0.0 (9) | 100.0± 0.0 (9) | 100.0± 0.0 (9) | 100.0± 0.0 (9) |

| retention      |         32000 |         64000 |        128000 |        192000 |        256000 |        384000 |        512000 |
|---|---|---|---|---|---|---|---|
| abstain        |        100.0% |        100.0% |        100.0% |        100.0% |        100.0% |        100.0% |        100.0% |
| counting       |        200.0% |        160.0% |        100.0% |        140.0% |        140.0% |        140.0% |         46.7% |
| multihop       |        100.0% |        100.0% |        100.0% |        100.0% |        100.0% |        100.0% |        100.0% |
| niah           |        100.0% |        100.0% |        100.0% |        100.0% |        100.0% |        100.0% |        100.0% |
| niah_multi     |        100.0% |        100.0% |        100.0% |        100.0% |        100.0% |        100.0% |        100.0% |
| semantic       |        100.0% |        100.0% |        100.0% |        100.0% |        100.0% |        100.0% |        100.0% |

- **stock @ 256000: retrieval retention 100.0% ≥ 85% → training OPTIONAL/targeted**
- **stock @ 384000: retrieval retention 100.0% ≥ 85% → training OPTIONAL/targeted**
- **stock @ 512000: retrieval retention 100.0% ≥ 85% → training OPTIONAL/targeted**
```
## suite: nolima
```
| config     engine    task            ctx |  score  ±95%   n | ttft med wall med finish-length |
-------------------------------------------------------------------------------------------------
| stock      vllm      nolima       512000  |  0.444 0.405   9 |   1399.5   1454.4             0 |
| stock      vllm      nolima       384000  |  0.222 0.339   9 |   1155.4   1156.0             0 |
| stock      vllm      nolima       256000  |  0.667 0.384   9 |    554.7    557.7             0 |
| stock      vllm      nolima       128000  |  0.444 0.405   9 |    495.6    498.7             0 |
| stock      vllm      nolima        64000  |  0.556 0.405   9 |    189.3    219.7             0 |
| stock      vllm      nolima        32000  |  0.444 0.405   9 |    310.0    310.3             0 |
```
## suite: longbench_v2
```
| config     engine    task            ctx |  score  ±95%   n | ttft med wall med finish-length |
-------------------------------------------------------------------------------------------------
| stock      vllm      longbench_v2  512000  |  1.000 0.000   3 |    910.0    912.8             0 |
| stock      vllm      longbench_v2  256000  |  0.667 1.434   3 |    558.0    558.3             0 |
| stock      vllm      longbench_v2  128000  |  0.333 1.434   3 |    509.4    509.7             0 |
| stock      vllm      longbench_v2   32000  |  0.333 1.434   3 |    279.7    280.0             0 |
```
## suite: longcodeqa
```
| config     engine    task            ctx |  score  ±95%   n | ttft med wall med finish-length |
-------------------------------------------------------------------------------------------------
| stock      vllm      longcodeqa   512000  |  0.667 1.434   3 |    860.2    862.8             0 |
| stock      vllm      longcodeqa   256000  |  1.000 0.000   3 |    714.0    716.0             0 |
| stock      vllm      longcodeqa   128000  |  0.667 1.434   3 |    582.8    587.1             0 |
| stock      vllm      longcodeqa    64000  |  0.667 1.434   3 |    431.3    435.4             0 |
| stock      vllm      longcodeqa    32000  |  0.667 1.434   3 |    669.4    669.7             0 |
```
## suite: infbench
```
| config     engine    task            ctx |  score  ±95%   n | ttft med wall med finish-length |
-------------------------------------------------------------------------------------------------
| stock      vllm      infb_bookmc  256000  |  0.000 0.000   3 |    658.8    677.4             0 |
| stock      vllm      infb_bookmc  128000  |  0.333 1.434   3 |    519.4    643.4             0 |
| stock      vllm      infb_codedebug  256000  |  0.000 0.000   3 |   1011.7   1078.9             0 |
| stock      vllm      infb_codedebug  128000  |  0.667 1.434   3 |    206.8    772.3             1 |
| stock      vllm      infb_kv      256000  |  1.000 0.000   3 |    253.9    285.9             0 |
| stock      vllm      infb_kv      128000  |  1.000 0.000   3 |    290.9    300.0             0 |
```
## suite: agentmem
```
| config     engine    task            ctx |  score  ±95%   n | ttft med wall med finish-length |
-------------------------------------------------------------------------------------------------
| stock      vllm      agentmem      32000  |  1.000 0.000   2 |     63.9     68.9             0 |
```
## suite: synth3
```
| config     engine    task            ctx |  score  ±95%   n | ttft med wall med finish-length |
-------------------------------------------------------------------------------------------------
| stock      vllm      chronology   128000  |  1.000 0.000  21 |    148.4    150.1             0 |
| stock      vllm      chronology    64000  |  1.000 0.000  21 |     94.6     96.0             0 |
| stock      vllm      chronology    32000  |  0.952 0.068  21 |     72.9     74.4             0 |
| stock      vllm      conflicts    128000  |  1.000 0.000  21 |    403.2    428.1             0 |
| stock      vllm      conflicts     64000  |  1.000 0.000  21 |    174.6    177.5             0 |
| stock      vllm      conflicts     32000  |  1.000 0.000  21 |    129.9    138.1             0 |
| stock      vllm      set_intersect  128000  |  1.000 0.000  21 |    131.1    133.0             0 |
| stock      vllm      set_intersect   64000  |  0.968 0.062  21 |     86.8     88.5             0 |
| stock      vllm      set_intersect   32000  |  0.960 0.078  21 |    150.3    151.8             0 |
```
## counting error anatomy (off-by-one undercount = attention dilution)
```
stock                  ctx=  32000: exact= 20/21 under-1=1 under-N=0 over=0 other=0
stock                  ctx=  64000: exact= 16/21 under-1=4 under-N=1 over=0 other=0
stock                  ctx= 128000: exact= 10/21 under-1=7 under-N=3 over=1 other=0
stock                  ctx= 192000: exact=  6/9 under-1=2 under-N=1 over=0 other=0
stock                  ctx= 256000: exact=  6/9 under-1=1 under-N=1 over=1 other=0
stock                  ctx= 384000: exact=  6/9 under-1=2 under-N=1 over=0 other=0
stock                  ctx= 512000: exact=  2/9 under-1=2 under-N=5 over=0 other=0
```
## §5 corpus manifest (train_v1)
```json
{
 "name": "train_v1",
 "seed": 2026,
 "rows": 259,
 "tokens": 56990652,
 "components": {
  "repos": {
   "n": 80,
   "avail": 80,
   "tokens": 44431644,
   "target": 0.35,
   "actual": 0.3089
  },
  "synth": {
   "n": 88,
   "avail": 88,
   "tokens": 6524041,
   "target": 0.3,
   "actual": 0.3398
  },
  "natural": {
   "n": 52,
   "avail": 139,
   "tokens": 6000715,
   "target": 0.15,
   "actual": 0.2008
  },
  "agent": {
   "n": 7,
   "avail": 7,
   "tokens": 25337,
   "target": 0.1,
   "actual": 0.027
  },
  "short": {
   "n": 32,
   "avail": 32,
   "tokens": 8915,
   "target": 0.1,
   "actual": 0.1236
  }
 },
 "length_buckets": {
  "131072": {
   "rows": 174,
   "tokens": 10868703
  },
  "262144": {
   "rows": 223,
   "tokens": 20466867
  }
 },
 "length_note": "genuine-only so far; virtual-position rows enter via the position sampler at \u00a79 ablation time. length_buckets = what the trainer sees per --seq-bucket setting."
}```

Generated by scripts/collect_results.sh — do not edit by hand.
