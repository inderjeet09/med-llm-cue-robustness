# Third-party material

This repository's own code is MIT-licensed (see LICENSE). It also contains or uses:

| material | where | source and licence |
|---|---|---|
| Seven BiasMedQA cue sentences and their "education" texts, and the BiasMedQA prompt wording | `src/data/bias.py`, `prompts/biasmedqa_*` | Quoted from Schmidgall et al. 2024, *npj Digital Medicine* 7:295, doi:10.1038/s41746-024-01283-6 (article CC BY 4.0). Their code repository has no licence; no code was copied. |
| Handler et al. NOTA prompts | `prompts/handler_*` | github.com/handler9/NOTA-Benchmark (MIT), adapted; Handler, Bedi & Shah 2026, arXiv:2605.15000 |
| Supplementary Tables 1–4 of Schmidgall et al. (transcribed numbers) | `results/reproduction/schmidgall2024_supp_tables.csv` | doi:10.1038/s41746-024-01283-6 (CC BY 4.0) |
| MedQA-USMLE | downloaded by `scripts/prepare_medqa.py` (not redistributed) | Jin et al. 2021, doi:10.3390/app11146421 (MIT), via Hugging Face `bigbio/med_qa` |
| MedQA-MultiTurnRobust | downloaded by `src/data/multiturn.py` (not redistributed) | Manczak et al. 2025, arXiv:2510.12255, HF `dynamoai-ml/MedQA-USMLE-4-MultiTurnRobust` (MIT) |
| Model weights | downloaded at run time (not redistributed) | Qwen2.5-7B-Instruct (Apache-2.0); Llama-3.1-8B-Instruct (Llama 3.1 Community Licence, gated); Llama3-Med42-8B (Llama 3 licence) |

Per-question result files in `results/` contain model outputs on MedQA items, identified by item id only (no question
text). Not for clinical use.
