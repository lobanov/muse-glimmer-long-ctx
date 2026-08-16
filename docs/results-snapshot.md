# Results snapshot — 2026-08-16 20:44:52

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
- `outputs/eval/ppl_stock.jsonl` (10 rows)
- `outputs/eval/smoke_vllm.jsonl` (4 rows)
- `outputs/eval/stock_cwe.jsonl` (36 rows)
- `outputs/eval/stock_vllm_gt128k.jsonl` (216 rows)
- `outputs/eval/stock_vllm_le128k.jsonl` (378 rows)
- `outputs/eval/stock_weak5.jsonl` (20 rows)
- `outputs/eval/suite_agentmem.jsonl` (72 rows)
- `outputs/eval/suite_infbench.jsonl` (18 rows)
- `outputs/eval/suite_longbench_v2.jsonl` (12 rows)
- `outputs/eval/suite_longcodeqa.jsonl` (15 rows)
- `outputs/eval/suite_nolima.jsonl` (54 rows)
- `outputs/eval/suite_synth3.jsonl` (189 rows)

## §3 stock baseline — score by task × ctx (mean ± 95% CI (n))
```
| config     engine    task            ctx |  score  ±95%   n | ttft med wall med finish-length |
-------------------------------------------------------------------------------------------------
| stock      vllm      abstain      512000  |  1.000 0.000   9 |     19.4     20.5             0 |
| stock      vllm      abstain      384000  |  1.000 0.000   9 |     20.5     21.6             0 |
| stock      vllm      abstain      256000  |  1.000 0.000   9 |     15.5     16.5             0 |
| stock      vllm      abstain      192000  |  1.000 0.000   9 |     11.1     12.1             0 |
| stock      vllm      abstain      128000  |  1.000 0.000  21 |     13.3     14.3             0 |
| stock      vllm      abstain       64000  |  1.000 0.000  21 |     14.4     15.4             0 |
| stock      vllm      abstain       32000  |  1.000 0.000  21 |     14.0     15.0             0 |
| stock      vllm      counting     512000  |  0.222 0.339   9 |    996.0    996.3             0 |
| stock      vllm      counting     384000  |  0.667 0.384   9 |   1023.2   1025.5             0 |
| stock      vllm      counting     256000  |  0.643 0.260  14 |    502.7    503.0             0 |
| stock      vllm      counting     192000  |  0.667 0.384   9 |    366.0    366.3             0 |
| stock      vllm      counting     128000  |  0.538 0.195  26 |    164.2    164.4             0 |
| stock      vllm      counting      64000  |  0.762 0.187  21 |    105.3    105.6             0 |
| stock      vllm      counting      32000  |  0.952 0.093  21 |     82.5     82.7             0 |
| stock      vllm      cwe          256000  |  0.600 0.680   5 |    380.1    380.6             0 |
| stock      vllm      cwe          128000  |  1.000 0.000   5 |    227.3    227.8             0 |
| stock      vllm      multihop     512000  |  1.000 0.000   9 |    874.7    876.8             0 |
| stock      vllm      multihop     384000  |  1.000 0.000   9 |    797.7    799.5             0 |
| stock      vllm      multihop     256000  |  1.000 0.000   9 |    445.9    471.7             0 |
| stock      vllm      multihop     192000  |  1.000 0.000   9 |    309.6    329.3             0 |
| stock      vllm      multihop     128000  |  1.000 0.000  21 |    118.4    120.2             0 |
| stock      vllm      multihop      64000  |  1.000 0.000  21 |     62.1     63.9             0 |
| stock      vllm      multihop      32000  |  1.000 0.000  21 |     41.6     43.2             0 |
| stock      vllm      niah         512000  |  1.000 0.000   9 |    520.1    522.0             0 |
| stock      vllm      niah         384000  |  1.000 0.000   9 |    320.1    332.1             0 |
| stock      vllm      niah         256000  |  1.000 0.000   9 |    166.0    168.0             0 |
| stock      vllm      niah         192000  |  1.000 0.000   9 |    117.9    119.6             0 |
| stock      vllm      niah         128000  |  1.000 0.000  21 |     78.1     80.1             0 |
| stock      vllm      niah          64000  |  1.000 0.000  21 |     41.0     43.0             0 |
| stock      vllm      niah          32000  |  1.000 0.000  21 |     28.4     30.3             0 |
| stock      vllm      niah_multi   512000  |  1.000 0.000   9 |    872.7    894.7             0 |
| stock      vllm      niah_multi   384000  |  1.000 0.000   9 |    571.8    580.2             0 |
| stock      vllm      niah_multi   256000  |  1.000 0.000   9 |    357.7    370.4             0 |
| stock      vllm      niah_multi   192000  |  1.000 0.000   9 |    236.7    248.7             0 |
| stock      vllm      niah_multi   128000  |  1.000 0.000  21 |    143.0    150.5             0 |
| stock      vllm      niah_multi    64000  |  1.000 0.000  21 |     84.2     92.0             0 |
| stock      vllm      niah_multi    32000  |  1.000 0.000  21 |     60.6     67.4             0 |
| stock      vllm      semantic     512000  |  1.000 0.000   9 |    750.6    751.7             0 |
| stock      vllm      semantic     384000  |  1.000 0.000   9 |    456.6    464.9             0 |
| stock      vllm      semantic     256000  |  1.000 0.000   9 |    259.1    259.9             0 |
| stock      vllm      semantic     192000  |  1.000 0.000   9 |    157.8    159.7             0 |
| stock      vllm      semantic     128000  |  1.000 0.000  21 |     80.1     80.8             0 |
| stock      vllm      semantic      64000  |  1.000 0.000  21 |     43.9     44.7             0 |
| stock      vllm      semantic      32000  |  1.000 0.000  21 |     33.8     34.9             0 |
```
## §3 retention vs 128k + decision rule
```
# Retention analysis — stock (stock)
rows: 614; reference ctx = 128000

## stock
| task           |         32000 |         64000 |        128000 |        192000 |        256000 |        384000 |        512000 |
|---|---|---|---|---|---|---|---|
| abstain        | 100.0± 0.0 (21) | 100.0± 0.0 (21) | 100.0± 0.0 (21) | 100.0± 0.0 (9) | 100.0± 0.0 (9) | 100.0± 0.0 (9) | 100.0± 0.0 (9) |
| counting       |  95.2± 9.3 (21) |  76.2±18.7 (21) |  53.8±19.5 (26) |  66.7±32.7 (9) |  64.3±26.0 (14) |  66.7±32.7 (9) |  22.2±28.8 (9) |
| cwe            |             — |             — | 100.0± 0.0 (5) |             — |  60.0±68.0 (5) |             — |             — |
| multihop       | 100.0± 0.0 (21) | 100.0± 0.0 (21) | 100.0± 0.0 (21) | 100.0± 0.0 (9) | 100.0± 0.0 (9) | 100.0± 0.0 (9) | 100.0± 0.0 (9) |
| niah           | 100.0± 0.0 (21) | 100.0± 0.0 (21) | 100.0± 0.0 (21) | 100.0± 0.0 (9) | 100.0± 0.0 (9) | 100.0± 0.0 (9) | 100.0± 0.0 (9) |
| niah_multi     | 100.0± 0.0 (21) | 100.0± 0.0 (21) | 100.0± 0.0 (21) | 100.0± 0.0 (9) | 100.0± 0.0 (9) | 100.0± 0.0 (9) | 100.0± 0.0 (9) |
| semantic       | 100.0± 0.0 (21) | 100.0± 0.0 (21) | 100.0± 0.0 (21) | 100.0± 0.0 (9) | 100.0± 0.0 (9) | 100.0± 0.0 (9) | 100.0± 0.0 (9) |

| retention      |         32000 |         64000 |        128000 |        192000 |        256000 |        384000 |        512000 |
|---|---|---|---|---|---|---|---|
| abstain        |        100.0% |        100.0% |        100.0% |        100.0% |        100.0% |        100.0% |        100.0% |
| counting       |        176.9% |        141.5% |        100.0% |        123.8% |        119.4% |        123.8% |         41.3% |
| cwe            |          nan% |          nan% |        100.0% |          nan% |         60.0% |          nan% |          nan% |
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
| stock      vllm      agentmem     512000  |  0.750 0.256  12 |    751.8    757.2             0 |
| stock      vllm      agentmem     384000  |  1.000 0.000  12 |    505.7    511.4             0 |
| stock      vllm      agentmem     256000  |  1.000 0.000  12 |    291.7    296.7             0 |
| stock      vllm      agentmem     128000  |  1.000 0.000  12 |    160.3    165.3             0 |
| stock      vllm      agentmem      64000  |  0.979 0.041  12 |    101.5    106.8             0 |
| stock      vllm      agentmem      32000  |  1.000 0.000  12 |     75.8     80.8             0 |
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
## counting error anatomy (off-by-one undercount — one live hypothesis; see PLAN §3 erratum)
```
stock                  ctx=  32000: exact= 20/21 under-1=1 under-N=0 over=0 other=0
stock                  ctx=  64000: exact= 16/21 under-1=4 under-N=1 over=0 other=0
stock                  ctx= 128000: exact= 10/21 under-1=7 under-N=3 over=1 other=0
stock                  ctx= 192000: exact=  6/9 under-1=2 under-N=1 over=0 other=0
stock                  ctx= 256000: exact=  6/9 under-1=1 under-N=1 over=1 other=0
stock                  ctx= 384000: exact=  6/9 under-1=2 under-N=1 over=0 other=0
stock                  ctx= 512000: exact=  2/9 under-1=2 under-N=5 over=0 other=0
```
## PPL curve — ppl_stock (last-8k-token span)
```
stock-524k     32000 rep0: ppl=14.0905 over n=8192 (prompt 31,681) [31.3s]
stock-524k     32000 rep1: ppl=12.1738 over n=8192 (prompt 31,809) [26.0s]
stock-524k    131072 rep0: ppl=11.8677 over n=8192 (prompt 130,746) [127.1s]
stock-524k    131072 rep1: ppl=8.0974 over n=8192 (prompt 130,538) [127.5s]
stock-524k    262144 rep0: ppl=6.4569 over n=8192 (prompt 261,376) [311.1s]
stock-524k    262144 rep1: ppl=6.1322 over n=8192 (prompt 261,043) [309.8s]
stock-524k    393216 rep0: ppl=5.8345 over n=8192 (prompt 391,766) [552.9s]
stock-524k    393216 rep1: ppl=4.7879 over n=8192 (prompt 391,704) [551.2s]
stock-524k    524288 rep0: ppl=3.7562 over n=8192 (prompt 480,049) [747.5s]
stock-524k    524288 rep1: ppl=3.7562 over n=8192 (prompt 480,049) [748.1s]
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
