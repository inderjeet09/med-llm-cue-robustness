"""Acceptance check for prefix-shared evaluation (docs/preregistration.md B7.1, as amended): compare
results/main/<model>/base_dev/records.jsonl (prefix-shared) with results/gap_analysis/<model>_dev600_s0/records.jsonl
(standard path, same 600 dev questions, same prompts). Accept if every shared condition's accuracy is within 1 pp and
overall answer agreement is >= 97.3 %. Writes results/main/prefix_eval_check.json.
Usage: python scripts/check_prefix_eval.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.utils.provenance import provenance  # noqa: E402


def load(p: Path) -> dict:
    by: dict = {}
    for r in map(json.loads, p.open()):
        by.setdefault(r["cond"], {})[r["qid"]] = r
    return by


out = {}
for md in sorted((ROOT / "results/main").iterdir()):
    a_p, b_p = md / "base_dev/records.jsonl", ROOT / f"results/gap_analysis/{md.name}_dev600_s0/records.jsonl"
    if not (md.is_dir() and a_p.exists() and b_p.exists()):
        continue
    A, B = load(a_p), load(b_p)
    conds = sorted(set(A) & set(B))
    agree = sum(A[c][q]["pred"] == B[c][q]["pred"] for c in conds for q in A[c])
    total = sum(len(A[c]) for c in conds)
    accd = {c: 100 * (sum(A[c][q]["correct"] for q in A[c]) - sum(B[c][q]["correct"] for q in B[c])) / len(A[c]) for c in conds}
    ok = agree / total >= 0.973 and max(map(abs, accd.values())) <= 1.0
    out[md.name] = {"conditions": len(conds), "agreement_pct": 100 * agree / total,
                    "max_abs_acc_diff_pp": max(map(abs, accd.values())), "accepted": ok,
                    "eval_seconds_prefix": json.loads((md / "base_dev/summary.json").read_text())["times"]["eval_s"]}
    print(md.name, out[md.name])
out["provenance"] = provenance({}, __file__)
(ROOT / "results/main/prefix_eval_check.json").write_text(json.dumps(out, indent=1))
