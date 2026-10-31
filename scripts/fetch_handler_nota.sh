#!/usr/bin/env bash
# Fetch Handler et al. 2026 MedQA-NOTA items (MIT, github.com/handler9/NOTA-Benchmark, pinned commit) into data/raw.
# Used (1) to exclude these questions from our dev experiments, (2) as an external test set later.
set -euo pipefail
cd "$(dirname "$0")/.."
C=c8db3609f97b789db415502a13cfafd7d2094523
D=data/raw/handler_nota; mkdir -p "$D"
for f in questions.csv question_key.csv LICENSE; do
  curl -sfL "https://raw.githubusercontent.com/handler9/NOTA-Benchmark/$C/${f/questions.csv/data/questions.csv}" -o "$D/$f" 2>/dev/null \
  || curl -sfL "https://raw.githubusercontent.com/handler9/NOTA-Benchmark/$C/data/$f" -o "$D/$f" \
  || curl -sfL "https://raw.githubusercontent.com/handler9/NOTA-Benchmark/$C/$f" -o "$D/$f"
done
echo "$C" > "$D/COMMIT.txt"; ls -la "$D"
