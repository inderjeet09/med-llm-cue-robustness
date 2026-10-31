"""Decompress the per-question result files (*.jsonl.gz -> *.jsonl) so the analysis scripts can read them.
They are stored gzipped only to keep the repository small. Usage: python scripts/unpack_results.py"""
import gzip
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
n = 0
for gz in (ROOT / "results").rglob("*.gz"):
    out = gz.with_suffix("")
    if not out.exists():
        with gzip.open(gz, "rb") as fi, out.open("wb") as fo:
            shutil.copyfileobj(fi, fo)
        n += 1
print(f"unpacked {n} files")
