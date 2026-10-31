# Phase 5 main results (MedQA test; mean ± std over seeds; two-level bootstrap CIs)

## Qwen__Qwen2.5-7B-Instruct
Best prompt baseline (dev rule): fewshot

| arm | seeds | clean | misleading: training cues | misleading: held-out cues | helpful |
|---|---|---|---|---|---|
| base | 1 | 58.2 | 33.9 | 23.3 | 89.6 |
| bct | 3 | 58.5 ± 0.1 | 57.9 ± 0.8 | 57.6 ± 1.2 | 56.2 ± 2.5 |
| cift | 3 | 62.3 ± 0.2 | 59.8 ± 0.8 | 58.5 ± 1.2 | 66.9 ± 3.1 |
| fewshot | 1 | 57.0 | 52.4 | 38.2 | – |
| sft_clean | 3 | 62.3 ± 1.0 | 51.8 ± 2.3 | 41.5 ± 3.3 | 78.7 ± 2.9 |
| warn_generic | 1 | 57.2 | 37.4 | 27.8 | – |
| warn_oracle | 1 | 58.1 | 45.6 | 41.3 | – |

| test | diff (pp) | 95% CI | pass |
|---|---|---|---|
| H1_heldout_vs_base | +35.1 | [32.9, 37.5] | yes |
| H2_traincues_vs_base | +25.9 | [23.8, 28.0] | yes |
| H1_heldout_vs_best_prompt | +20.3 | [18.1, 22.4] | yes |
| H2_traincues_vs_best_prompt | +7.3 | [5.3, 9.5] | yes |
| H1_heldout_vs_sft_clean | +17.0 | [13.2, 20.1] | yes |
| H2_traincues_vs_sft_clean | +8.0 | [5.2, 10.4] | yes |
| H1_heldout_vs_bct | +0.9 | [-1.4, 3.1] | no |
| H2_traincues_vs_bct | +1.9 | [-0.1, 3.8] | no |
| H3_clean_vs_base_pp | +4.1 |  | yes |
| H3_clean_vs_sft | +0.1 | [-1.6, 1.6] | yes |

## m42-health__Llama3-Med42-8B
Best prompt baseline (dev rule): warn_oracle

| arm | seeds | clean | misleading: training cues | misleading: held-out cues | helpful |
|---|---|---|---|---|---|
| base | 1 | 59.2 | 51.6 | 39.1 | 72.2 |
| bct | 3 | 59.0 ± 1.3 | 58.0 ± 1.5 | 56.9 ± 1.4 | 57.6 ± 0.9 |
| cift | 3 | 60.7 ± 0.7 | 58.9 ± 0.4 | 57.6 ± 0.8 | 64.2 ± 0.8 |
| sft_clean | 3 | 61.5 ± 1.0 | 59.7 ± 0.6 | 54.6 ± 1.0 | 60.5 ± 1.2 |
| warn_generic | 1 | 58.7 | 53.6 | 48.6 | – |
| warn_oracle | 1 | 59.1 | 55.6 | 52.4 | – |

| test | diff (pp) | 95% CI | pass |
|---|---|---|---|
| H1_heldout_vs_base | +18.5 | [16.6, 20.4] | yes |
| H2_traincues_vs_base | +7.2 | [5.4, 8.9] | yes |
| H1_heldout_vs_best_prompt | +5.3 | [3.2, 7.2] | yes |
| H2_traincues_vs_best_prompt | +3.3 | [1.5, 5.0] | yes |
| H1_heldout_vs_sft_clean | +3.0 | [1.4, 4.7] | yes |
| H2_traincues_vs_sft_clean | -0.8 | [-2.2, 0.5] | no |
| H1_heldout_vs_bct | +0.8 | [-1.5, 3.1] | no |
| H2_traincues_vs_bct | +0.8 | [-1.4, 3.0] | no |
| H3_clean_vs_base_pp | +1.5 |  | yes |
| H3_clean_vs_sft | -0.8 | [-2.7, 1.0] | no |

## meta-llama__Llama-3.1-8B-Instruct
Best prompt baseline (dev rule): warn_generic

| arm | seeds | clean | misleading: training cues | misleading: held-out cues | helpful |
|---|---|---|---|---|---|
| base | 1 | 59.5 | 42.2 | 31.4 | 85.7 |
| bct | 3 | 60.9 ± 0.6 | 59.9 ± 0.9 | 59.5 ± 1.0 | 60.0 ± 0.6 |
| cift_nohelp | 3 | 64.7 ± 1.1 | 68.6 ± 0.4 | 67.8 ± 0.4 | 38.5 ± 1.9 |
| cift_nokl | 3 | 64.5 ± 0.2 | 60.6 ± 2.1 | 58.8 ± 2.7 | 69.6 ± 3.9 |
| cift | 3 | 65.1 ± 0.9 | 62.0 ± 1.6 | 60.5 ± 2.1 | 68.7 ± 1.7 |
| sft_clean | 3 | 64.2 ± 0.7 | 55.3 ± 0.8 | 48.5 ± 1.5 | 77.4 ± 2.2 |
| warn_generic | 1 | 58.8 | 55.8 | 55.7 | – |
| warn_oracle | 1 | 60.2 | 54.7 | 56.8 | – |

| test | diff (pp) | 95% CI | pass |
|---|---|---|---|
| H1_heldout_vs_base | +29.0 | [26.6, 31.6] | yes |
| H2_traincues_vs_base | +19.7 | [17.7, 22.0] | yes |
| H1_heldout_vs_best_prompt | +4.7 | [2.1, 7.5] | yes |
| H2_traincues_vs_best_prompt | +6.1 | [3.9, 8.6] | yes |
| H1_heldout_vs_sft_clean | +12.0 | [9.3, 14.5] | yes |
| H2_traincues_vs_sft_clean | +6.7 | [4.8, 8.7] | yes |
| H1_heldout_vs_bct | +1.0 | [-1.5, 3.4] | no |
| H2_traincues_vs_bct | +2.1 | [-0.0, 4.3] | no |
| H3_clean_vs_base_pp | +5.7 |  | yes |
| H3_clean_vs_sft | +0.9 | [-0.9, 2.6] | yes |
| H4_heldout_vs_nokl | +1.7 | [-1.8, 4.9] | no |
| H4_helpful_vs_nohelp | +30.2 | [27.0, 33.2] | yes |

