# Running the GPU experiments on Kaggle (free T4 × 2)

All 7–8B experiments in this repository ran on free Kaggle notebooks (2 × NVIDIA T4, 4-bit models). A Mac/CPU is enough
only for the API runs, the analysis and tiny debugging models.

## One-time setup
1. A Kaggle account with phone verification (needed for GPU and internet in notebooks).
2. Accept the Llama 3.1 licence on Hugging Face (meta-llama/Llama-3.1-8B-Instruct) with the account behind your token.
3. In the notebook: **Add-ons → Secrets → Add secret** `HF_TOKEN` = a read-only Hugging Face token. It is never part of
   the uploaded files.

## Build and upload the input dataset
0. **Clone** the repository (`git clone …`); a ZIP download will not work, because the bundle is built with `git archive`
   from a clean commit.
1. Prepare the data locally (see the main README: `scripts/prepare_medqa.py`, `scripts/fetch_handler_nota.sh`,
   `python -c "from src.data.multiturn import prepare; prepare()"`).
2. `bash scripts/make_kaggle_bundle.sh` writes flat files to `cloud/kaggle_upload/` (Kaggle datasets accept files, not
   folders): `code_<commit>.tar.gz`, `COMMIT.txt`, `SHA256SUMS.txt`, `README.txt`, MedQA `train/dev/test.jsonl`,
   `medqa_PREPARED.json`, 5-option `medqa5_*.jsonl`, `mtr_items.jsonl`, `handler_questions.csv`. The script requires a
   clean git tree, because the code archive is named after the commit.
3. Kaggle → Datasets → New Dataset → drag all files in → keep it private. For a new code version: New Version, delete the
   old `code_*.tar.gz`, `COMMIT.txt` and `SHA256SUMS.txt`, upload the new ones.

## Run a batch
1. Kaggle → Code → New Notebook → File → Import Notebook → `cloud/kaggle_batch.ipynb` (regenerate it with
   `python scripts/make_kaggle_batch_notebook.py`).
2. Add the dataset as input; Settings: Accelerator **GPU T4 × 2**, Internet **on**; tick the `HF_TOKEN` secret.
3. In the second cell set `QUEUE` to a file name in `configs/queues/` (`quick_check` for the one-session reviewer check,
   or e.g. `p5_1`) and keep `MAX_HOURS = 11.0`
   (Kaggle sessions end at 12 h; the runner stops launching new jobs after `MAX_HOURS`).
4. **Save Version → Save & Run All (Commit)**: it runs with the browser closed. One job runs per GPU; the results zip is
   rewritten after every finished job; a failed job is retried once. A new session starts from an empty folder, so jobs
   finished in an earlier session must be removed from the queue file before re-running.
5. Download `<QUEUE>_results.zip` from the version's Output tab into `cloud/` and run
   `python scripts/ingest_kaggle.py cloud/<QUEUE>_results.zip`: it unpacks into `results/`, checks that every result's
   provenance commit matches the queue's commit and adds a row to `results/costs/gpu_ledger.csv`.
   **Reviewers:** add `--dest reruns` so the shipped results stay untouched (no ledger row is written), then compare:
   `python scripts/compare_rerun.py reruns/results/main`. Expect agreement within about 1 pp per condition, not
   identical numbers (GPU batch noise, `docs/DECISIONS.md` D019).

The notebook installs the GPU libraries pinned in `configs/requirements-kaggle.txt` and prints their versions; every
run records them in its `summary.json` (`provenance.libraries`).

## The batches behind the paper
| queue | contents |
|---|---|
| quick_check | reviewer check: Qwen2.5-7B untrained + CIFT seed 0 on test, one session (~3 h) |
| killtest1, killtest2, killtest2b | Phase 1.5 kill tests (`docs/preregistration.md`, `results/pilot/`) |
| repro1, repro1b | faithful BiasMedQA reproduction (`results/reproduction/comparison.md`) |
| gap1 | base models on all 15 cue templates, dev (`results/gap_analysis/analysis.md`) |
| p5_0 | prefix-shared evaluation check + prompt baselines on dev |
| p5_1 | test baselines + CIFT seed 0 (Qwen, Llama) |
| p5_2, p5_2c | CIFT seeds 1–2 + plain SFT seeds 0–2 (Qwen, Llama) |
| p5_3 | BCT seeds 0–2 (Qwen, Llama) |
| p5_5 | Llama ablations (no KL, no helpful pairs) |
| p5_4a, p5_4b, p5_4c | Med42: prompt baselines, CIFT, plain SFT, BCT |

Lessons recorded in `docs/DECISIONS.md`: D018 (token-budget batching after an out-of-memory failure), D019 (GPU
batch-composition noise ≤ 1 pp), D022 (fail fast on missing inputs; reuse trained adapters on retry).
