# Cost summary (generated 2026-10-24 by scripts/cost_report.py)

**Azure API total: $15.0314** of $25.00 global cap  ·  price basis: global
**Kaggle GPU-hours total: 154.49**

## Azure API by phase / experiment

| phase | experiment | calls | cache hits | prompt tok | completion tok | USD |
|---|---|---:|---:|---:|---:|---:|
| 0 | phase0_smoke_test | 2 | 1 | 28 | 2 | 0.0000 |
| 1.5 | killB_azure_gpt41mini_300 | 3900 | 129 | 1519629 | 1145433 | 2.3594 |
| 1.5 | killB_pilot | 130 | 0 | 48190 | 38640 | 0.0811 |
| 3 | repro_gpt41mini_n600_s0 | 17400 | 290 | 8474758 | 17362 | 3.3627 |
| 3 | repro_pilot_gpt41mini_n10 | 290 | 0 | 136407 | 290 | 0.0550 |
| 4 | gap__pilot_api | 155 | 0 | 38976 | 155 | 0.0158 |
| 4 | gap__pilot_test_api | 470 | 0 | 135533 | 470 | 0.0550 |
| 4 | gap_gpt41mini_dev600_s0 | 18600 | 155 | 5214622 | 18521 | 2.0996 |
| 4 | gap_gpt41mini_test | 64766 | 5315 | 18723156 | 63666 | 7.0027 |

## Kaggle GPU batches

| date | batch | commit | gpus | gpu_hours | notes |
|---|---|---|---:|---:|---|
| 2026-10-18 | killtest1_results | 6944cca | 2 | 8.60 | killtest1: kill test A x3 models + kill test B Qwen |
| 2026-10-18 | killtest2_results | a460ff4 | 2 | 4.18 | killtest2: stage 2 all7 x2 + base_extra x2 |
| 2026-10-18 | killtest2b_results | a460ff4 | 2 | 5.73 | killtest2b: heldout4 + sft_clean x2 models |
| 2026-10-18 | repro1_results | 3f06ae1 | 2 | 1.23 | repro1: faithful BiasMedQA reproduction, Qwen/Llama/Med42, 5-opt dev n600 |
| 2026-10-19 | repro1b_results | 76fe3ca | 2 | 3.57 | repro1b: few-shot re-run Qwen/Llama after OOM (D018) |
| 2026-10-20 | gap1_results | 6fcbcac | 2 | 5.44 | gap1: Phase 4 base models on 15 cues (dev 600) |
| 2026-10-20 | p5_0_results | 4ebad80 | 2 | 8.33 | p5_0: prefix check + prompt arms dev (Qwen, Llama) |
| 2026-10-20 | p5_1_results | da9f9a5 | 2 | 3.28 | p5_1: test baselines + CIFT s0 (Qwen, Llama) |
| 2026-10-21 | p5_1_results | c289a69 | 2 | 18.09 | p5_1 re-run (code c289a69): test baselines + CIFT s0 (Qwen, Llama) |
| 2026-10-22 | p5_2_results | c289a69 | 2 | 20.45 | p5_2 (code c289a69): CIFT s1-2 + SFT s0 (both) + SFT s1 (Llama); 3 SFT jobs cut by 12-h limit |
| 2026-10-22 | p5_2c_results | 7e799cb | 2 | 7.88 | p5_2c (code 7e799cb): SFT-clean Qwen s1,s2 + Llama s2 |
| 2026-10-23 | p5_3_results | 2040e12 | 2 | 16.49 | p5_3 (code 2040e12): BCT s0-2 (Qwen, Llama) |
| 2026-10-23 | p5_5_results | 930031c | 2 | 18.95 | p5_5 (code 930031c): Llama ablations cift_nokl, cift_nohelp x3 |
| 2026-10-24 | p5_4a_results | 8f15db8 | 2 | 7.73 | p5_4a (code 8f15db8): Med42 dev prompt arms + test baselines |
| 2026-10-24 | p5_4b_results | 8debb13 | 2 | 17.00 | p5_4b (code 8debb13): Med42 CIFT + SFT-clean x3 |
| 2026-10-24 | p5_4c_results | c251221 | 2 | 7.54 | p5_4c (code c251221): Med42 BCT x3 |
