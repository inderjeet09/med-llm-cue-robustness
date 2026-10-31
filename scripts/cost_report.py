"""Summarise API and GPU spending: results/costs/api_ledger.csv + gpu_ledger.csv -> results/costs/summary.md.

Usage: python scripts/cost_report.py
"""
from __future__ import annotations

import csv
import datetime as _dt
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.utils.cost_ledger import load_budget, read_ledger  # noqa: E402

GPU_LEDGER = ROOT / "results" / "costs" / "gpu_ledger.csv"
OUT = ROOT / "results" / "costs" / "summary.md"


def main() -> None:
    rows = read_ledger()
    by = defaultdict(lambda: {"calls": 0, "cache_hits": 0, "prompt": 0, "completion": 0, "usd": 0.0})
    for r in rows:
        b = by[(r["phase"] or "-", r["experiment"] or "-")]
        b["calls"] += 1
        b["cache_hits"] += r["cached"] == "yes"
        b["prompt"] += int(r["prompt_tokens"] or 0)
        b["completion"] += int(r["completion_tokens"] or 0)
        b["usd"] += float(r["cost_usd"] or 0)
    total = sum(b["usd"] for b in by.values())
    cap = load_budget()["global_cap_usd"]
    bases = sorted({r["price_basis"] for r in rows})
    gpu = []
    if GPU_LEDGER.exists():
        with GPU_LEDGER.open(newline="") as f:
            gpu = list(csv.DictReader(f))
    gpu_h = sum(float(g["gpu_hours"] or 0) for g in gpu)

    lines = [f"# Cost summary (generated {_dt.date.today().isoformat()} by scripts/cost_report.py)", "",
             f"**Azure API total: ${total:.4f}** of ${cap:.2f} global cap  ·  price basis: {', '.join(bases) or '-'}",
             f"**Kaggle GPU-hours total: {gpu_h:.2f}**", "",
             "## Azure API by phase / experiment", "",
             "| phase | experiment | calls | cache hits | prompt tok | completion tok | USD |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for (ph, ex), b in sorted(by.items()):
        lines.append(f"| {ph} | {ex} | {b['calls']} | {b['cache_hits']} | {b['prompt']} | {b['completion']} | {b['usd']:.4f} |")
    lines += ["", "## Kaggle GPU batches", "", "| date | batch | commit | gpus | gpu_hours | notes |", "|---|---|---|---:|---:|---|"]
    for g in gpu:
        lines.append(f"| {g['date']} | {g['batch']} | {g['commit']} | {g['gpus']} | {g['gpu_hours']} | {g['notes']} |")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n")
    print(f"API ${total:.4f} / ${cap:.2f} cap; GPU {gpu_h:.2f} h -> {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
