#!/usr/bin/env bash
# Adapted from ~/Desktop/ecg-research/scripts/make_kaggle_bundle.sh (same author).
# Build cloud/kaggle_upload/: FLAT files (Kaggle's uploader takes files, not folders) = code bundle of the CURRENT
# commit + MedQA train/dev/test (test added for Phase 5 after the plan was frozen, D020) + Handler question stems (for exclusion).
# Usage: bash scripts/make_kaggle_bundle.sh
set -euo pipefail
cd "$(dirname "$0")/.."
if [ -n "$(git status --porcelain --untracked-files=no -- src scripts configs tests prompts)" ]; then
  echo "Commit code changes first (bundle must match a commit)"; exit 1; fi
C=$(git rev-parse --short HEAD)
OUT=cloud/kaggle_upload
rm -rf "$OUT"; mkdir -p "$OUT"
# Phase 5 (plan frozen, D020): the TEST split is now needed for the final evaluation
cp data/raw/medqa_us4/train.jsonl data/raw/medqa_us4/dev.jsonl data/raw/medqa_us4/test.jsonl "$OUT/"
cp data/raw/medqa_us4/PREPARED.json "$OUT/medqa_PREPARED.json"
cp data/raw/handler_nota/questions.csv "$OUT/handler_questions.csv"
# 5-option MedQA (faithful BiasMedQA reproduction, D017): train (demonstrations) + dev only
cp data/raw/medqa_us5/train.jsonl "$OUT/medqa5_train.jsonl"; cp data/raw/medqa_us5/dev.jsonl "$OUT/medqa5_dev.jsonl"
cp data/raw/medqa_us5/PREPARED.json "$OUT/medqa5_PREPARED.json"
# MedQA-MultiTurnRobust (transfer test; MIT; pinned revision in src/data/multiturn.py)
cp data/raw/multiturn/items.jsonl "$OUT/mtr_items.jsonl"
git archive --format=tar.gz -o "$OUT/code_$C.tar.gz" HEAD src scripts configs tests prompts environment.yml
echo "$C" > "$OUT/COMMIT.txt"
cat > "$OUT/README.txt" <<TXT
medllm-robustness — Kaggle bundle (code commit $C). Private dataset, for this project only.
MedQA (Jin et al. 2021, doi:10.3390/app11146421; MIT) US 4-option train/dev, via HF bigbio/med_qa (pinned revision).
handler_questions.csv: Handler, Bedi & Shah 2026, github.com/handler9/NOTA-Benchmark (MIT) — used only to exclude
overlapping questions.
TXT
(cd "$OUT" && shasum -a 256 *.jsonl *.csv *.json code_*.tar.gz > SHA256SUMS.txt)
du -sh "$OUT"; ls -la "$OUT"
