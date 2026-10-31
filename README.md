# Robustness of small medical LLMs to misleading cues

Code, pre-registration and **all per-question results** for the paper
*"Training, not prompting, makes small medical language models robust to misleading cues: a pre-registered comparison of
counterfactual fine-tuning, consistency training and prompting"* (manuscript under review; the accepted version will be
added here).

**In one paragraph.** Add one sentence such as *"A senior attending physician who reviewed this case is confident that the
answer is C"* to a US medical-licensing exam question (MedQA), and small open models (Qwen2.5-7B, Llama-3.1-8B, Med42-8B)
switch to the wrong answer far more often. We (1) extend the BiasMedQA benchmark with four new, held-out cue families and
with *helpful* cues that point to the correct answer; (2) compare prompt mitigations, plain fine-tuning, bias-augmented
consistency training (BCT) and counterfactual-invariance fine-tuning (CIFT) in a pre-registered design (3 seeds, MedQA
test); (3) find that training on cued copies of training questions removes most of the vulnerability, including for cue
families never seen in training, and beats prompting and plain fine-tuning; CIFT is **not** more robust than BCT; its
KL term is unnecessary; and training without helpful cues produces a "contrarian" model that misleading-only benchmarks
would reward.

## Main results (MedQA test, accuracy %, mean ± s.d. over 3 seeds)
Generated from `results/` by `scripts/make_figures.py` (`results/tables/T1_main.md`):

| model | arm | seeds | clean | misleading: training cues | misleading: held-out cues | helpful | transfer: misleading context |
|---|---|---|---|---|---|---|---|
| Qwen2.5-7B | No mitigation | 1 | 58.2 | 33.9 | 23.3 | 89.6 | 4.9 |
| Qwen2.5-7B | Generic warning | 1 | 57.2 | 37.4 | 27.8 | – | – |
| Qwen2.5-7B | Oracle warning | 1 | 58.1 | 45.6 | 41.3 | – | – |
| Qwen2.5-7B | Few-shot (best prompt) | 1 | 57.0 | 52.4 | 38.2 | – | – |
| Qwen2.5-7B | Plain SFT | 3 | 62.3 ± 1.0 | 51.8 ± 2.3 | 41.5 ± 3.3 | 78.7 ± 2.9 | 12.0 ± 1.3 |
| Qwen2.5-7B | BCT | 3 | 58.5 ± 0.1 | 57.9 ± 0.8 | 57.6 ± 1.2 | 56.2 ± 2.5 | 8.6 ± 1.5 |
| Qwen2.5-7B | CIFT (ours) | 3 | 62.3 ± 0.2 | 59.8 ± 0.8 | 58.5 ± 1.2 | 66.9 ± 3.1 | 16.0 ± 3.5 |
| Llama-3.1-8B | No mitigation | 1 | 59.5 | 42.2 | 31.4 | 85.7 | 12.8 |
| Llama-3.1-8B | Generic warning (best prompt) | 1 | 58.8 | 55.8 | 55.7 | – | – |
| Llama-3.1-8B | Oracle warning | 1 | 60.2 | 54.7 | 56.8 | – | – |
| Llama-3.1-8B | Plain SFT | 3 | 64.2 ± 0.7 | 55.3 ± 0.8 | 48.5 ± 1.5 | 77.4 ± 2.2 | 22.9 ± 0.1 |
| Llama-3.1-8B | BCT | 3 | 60.9 ± 0.6 | 59.9 ± 0.9 | 59.5 ± 1.0 | 60.0 ± 0.6 | 19.7 ± 2.1 |
| Llama-3.1-8B | CIFT (ours) | 3 | 65.1 ± 0.9 | 62.0 ± 1.6 | 60.5 ± 2.1 | 68.7 ± 1.7 | 24.7 ± 1.8 |
| Llama-3.1-8B | CIFT w/o KL | 3 | 64.5 ± 0.2 | 60.6 ± 2.1 | 58.8 ± 2.7 | 69.6 ± 3.9 | 25.8 ± 1.3 |
| Llama-3.1-8B | CIFT w/o helpful pairs | 3 | 64.7 ± 1.1 | 68.6 ± 0.4 | 67.8 ± 0.4 | 38.5 ± 1.9 | 30.3 ± 0.1 |
| Med42-8B | No mitigation | 1 | 59.2 | 51.6 | 39.1 | 72.2 | 9.4 |
| Med42-8B | Generic warning | 1 | 58.7 | 53.6 | 48.6 | – | – |
| Med42-8B | Oracle warning (best prompt) | 1 | 59.1 | 55.6 | 52.4 | – | – |
| Med42-8B | Plain SFT | 3 | 61.5 ± 1.0 | 59.7 ± 0.6 | 54.6 ± 1.0 | 60.5 ± 1.2 | 14.9 ± 1.1 |
| Med42-8B | BCT | 3 | 59.0 ± 1.3 | 58.0 ± 1.5 | 56.9 ± 1.4 | 57.6 ± 0.9 | 13.1 ± 1.9 |
| Med42-8B | CIFT (ours) | 3 | 60.7 ± 0.7 | 58.9 ± 0.4 | 57.6 ± 0.8 | 64.2 ± 0.8 | 19.5 ± 2.4 |
| gpt-4.1-mini (reference, untrained) | No mitigation | 1 | 82.0 | 78.4 | 73.0 | – | – |
| gpt-4.1-mini (reference, untrained) | Oracle warning | 1 | – | 80.2 | 80.5 | – | – |
| gpt-4.1-mini (reference, untrained) | Generic warning | 1 | 81.0 | 79.4 | 77.6 | – | – |

Pre-registered hypothesis tests with two-level bootstrap CIs: `results/tables/main_results.md`. Exploratory comparisons:
`results/tables/exploratory_cift_vs_bct.json`, `results/tables/ablations_llama.json`. Figures: `figures/`.

## Repository map
| path | what it is |
|---|---|
| `docs/preregistration.md` | every decision rule, dated before the results it governs: kill tests, template freeze, the **frozen Phase 5 plan** (hypotheses H1–H4) and its dated amendments |
| `docs/DECISIONS.md` | design decisions D001–D028 with alternatives and reasons (the code cites them by number) |
| `src/data/` | MedQA loaders (`medqa.py`), the 15 cue templates and families (`bias.py`), NOTA transforms (`nota.py`), variant validation (`validate.py`), transfer set (`multiturn.py`) |
| `src/llm/` | prompt builders (`prompts.py`), HF 4-bit inference incl. prefix-shared scoring (`hf_runner.py`), Azure client with cache/ledger/budget (`azure_client.py`) |
| `src/training/` | training pairs (`pairs.py`), CIFT / BCT losses (`consistency.py`), QLoRA trainer for all arms (`trainer.py`) |
| `src/evaluation/` | answer parsing, statistics incl. two-level bootstrap (`stats.py`) |
| `scripts/` | entry points (see "Reproducing" below) |
| `configs/` | prices, budget caps, Kaggle job queues (`configs/queues/*.txt` — exactly what ran; `quick_check.txt` for reviewers), pinned Kaggle GPU libraries |
| `prompts/` | every prompt template as a versioned file |
| `results/` | every result file (per-question records gzipped), costs; `results/review/` the validity review |
| `cloud/` | Kaggle notebook + guide |
| `tests/` | unit tests (transforms, parsing, losses, batching, statistics, cost ledger, provenance) |
| `notebooks/` | data exploration |

## Reproducing
There are three levels. Levels 1 and 2 are what a reviewer normally needs.

**Level 1: re-derive every table, figure and paper number from the shipped results (no GPU, minutes).**
The output matches the shipped files exactly (apart from provenance fields).
```bash
conda env create -f environment.yml && conda activate medllm    # environment.yml is a macOS export; see below for Linux
python scripts/unpack_results.py                 # decompress per-question records (*.jsonl.gz)
pytest -q tests                                   # unit tests
python scripts/analyze_main.py                    # H1–H4 → results/tables/main_results.{json,md}
python scripts/analyze_exploratory.py             # exploratory CIFT vs BCT + Llama ablations → results/tables/*.json
python scripts/make_figures.py                    # tables T1–T3 + figures F1–F4
python scripts/make_tables.py                     # every number/table quoted in the paper, as LaTeX (written to paper/)
python scripts/compare_reproduction.py --runs gpt41mini_n600_s0 Qwen__Qwen2.5-7B-Instruct_n600_s0 \
    meta-llama__Llama-3.1-8B-Instruct_n600_s0 m42-health__Llama3-Med42-8B_n600_s0   # reproduction vs. base paper
```
Other analyses that also regenerate shipped files: `analyze_gap.py` (gap analysis), `analyze_kill_a2.py` (kill test),
`check_prefix_eval.py` (fast-evaluation check), `cost_report.py` (costs).

**Level 2: quick GPU check, one free Kaggle session (about 3 h).** Re-runs Qwen2.5-7B untrained and CIFT seed 0 on the
MedQA test split with the exact commands behind the shipped runs, then compares them with the shipped results.
Follow `cloud/README.md` with `QUEUE = "quick_check"`, then:
```bash
python scripts/ingest_kaggle.py cloud/quick_check_results.zip --dest reruns    # does not touch the shipped results
python scripts/compare_rerun.py reruns/results/main                            # agreement and differences per run
```

**Level 3: re-run everything.** Data preparation (needs a Hugging Face token: `export HF_TOKEN=...`, and the Llama 3.1
licence accepted on Hugging Face):
```bash
python scripts/prepare_medqa.py && python scripts/prepare_medqa.py --options 5   # MedQA (pinned revision, counts checked)
bash scripts/fetch_handler_nota.sh
python -c "from src.data.multiturn import prepare; prepare()"                    # transfer set
python scripts/make_variants.py                                                   # all dev/test variants + validation
```
Every GPU job is one line in `configs/queues/*.txt` (exactly what ran; 154.5 GPU-hours in total, `results/costs/summary.md`).
Run them on Kaggle (`cloud/README.md`) or on your own CUDA GPU with at least 16 GB (Linux; 4-bit loading needs
`bitsandbytes`, so install the packages in `configs/requirements-kaggle.txt` plus torch). Locally, add
`--out reruns/results/main` to each `run_main.py` command so the shipped results are not overwritten, e.g.
```bash
python scripts/run_main.py --model Qwen/Qwen2.5-7B-Instruct --revision a09a35458c702b33eeacc393d103063234e8bc28 \
    --arm cift --seed 0 --split test --out reruns/results/main
python scripts/compare_rerun.py reruns/results/main
```
API reference runs (gpt-4.1-mini via your own Azure OpenAI deployment; copy `.env.example` to `.env`):
```bash
python scripts/reproduce_biasmedqa.py --backend azure --n 600 --tag gpt41mini_n600_s0
python scripts/eval_cues.py --backend azure --split test --n 0 --groups mis edu gen --out_root results/reference --tag gpt41mini_test
```
Arms of `scripts/run_main.py`: `base`, `warn_generic`, `warn_oracle`, `fewshot`, `sft_clean`, `bct`, `cift`, `cift_nokl`,
`cift_nohelp`.

**What "the same results" means for GPU re-runs.** Expect agreement within about 1 percentage point per condition, not
identical numbers. Re-running identical code on the same GPU type with different batching already changed 0.8–2.7 % of
answers and moved accuracy by up to 1.0 pp per condition (`docs/DECISIONS.md` D019); a different GPU or library versions
can add to this. `compare_rerun.py` flags headline metrics that differ by more than `--tol` (default 1.0 pp).

**Known limitations of the shipped provenance.** The GPU runs in `results/` recorded Python and torch versions but not
the versions of transformers, PEFT or bitsandbytes (new runs record them, and `configs/requirements-kaggle.txt` now pins
them). Their `git_dirty: true` flags are an artefact of a since-fixed bug (no git checkout on Kaggle); the Kaggle code
bundle can only be built from a clean commit (`scripts/make_kaggle_bundle.sh`). Commit hashes in provenance refer to the
private development repository. The validity-review analysis (`scripts/analyze_review.py`) needs the completed review
sheet, which is not shipped because it contains test questions.

## Compute
Free Kaggle GPUs (2 × T4, 4-bit models) and about US$15 of Azure OpenAI calls. Costs per batch: `results/costs/`.

## Licence and citation
MIT (see `LICENSE`); third-party material and data licences in `NOTICE.md`. Not for clinical use.
Please cite the paper (citation details will be added here once the paper is published).
