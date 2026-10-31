"""Merge a follow-up reproduction run into the main run directory (used after repro1's few-shot jobs ran out of GPU
memory and were re-run as repro1b; docs/DECISIONS.md D018).

Conditions present in both runs are compared question by question (a determinism check). Each condition is taken from
exactly one run: by default the main run, except the strategies named in --take_from_extra, which come from the extra
run. The merged records replace <main>/records.jsonl (the original is kept as records_main_only.jsonl) and the summary
is recomputed with the same code as scripts/reproduce_biasmedqa.py.
Usage: python scripts/merge_repro_runs.py <main_dir> <extra_dir> --take_from_extra fewshot
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from reproduce_biasmedqa import summarise  # noqa: E402
from src.utils.provenance import provenance  # noqa: E402


def load(d: Path) -> dict[str, dict[str, dict]]:
    by: dict[str, dict[str, dict]] = {}
    for r in map(json.loads, (d / "records.jsonl").open()):
        by.setdefault(r["cond"], {})[r["qid"]] = r
    return by


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("main_dir")
    ap.add_argument("extra_dir")
    ap.add_argument("--take_from_extra", nargs="+", default=["fewshot"])
    args = ap.parse_args()
    main_d, extra_d = Path(args.main_dir).resolve(), Path(args.extra_dir).resolve()
    A, B = load(main_d), load(extra_d)
    n = len(A["nobias"])
    agree = {}
    for c in sorted(set(A) & set(B)):
        common = set(A[c]) & set(B[c])
        agree[c] = {"n": len(common), "same_pred": sum(A[c][q]["pred"] == B[c][q]["pred"] for q in common),
                    "max_prob_diff": max(abs(A[c][q]["probs"][L] - B[c][q]["probs"][L]) for q in common
                                         for L in A[c][q]["probs"]) if "probs" in next(iter(A[c].values())) else None}
    take_extra = {c for c in B if any(c.endswith("+" + s) for s in args.take_from_extra) and len(B[c]) == n}
    merged = {c: (B[c] if c in take_extra else A[c]) for c in set(A) | take_extra if len((B if c in take_extra else A)[c]) == n}
    shutil.copy(main_d / "records.jsonl", main_d / "records_main_only.jsonl")
    with (main_d / "records.jsonl").open("w") as f:
        for c in sorted(merged):
            for q in sorted(merged[c]):
                f.write(json.dumps(merged[c][q]) + "\n")
    old = json.loads((extra_d / "summary.json").read_text())
    summary = summarise([r for c in merged.values() for r in c.values()])
    summary["meta"] = {**old.get("meta", {}), "merged": {
        "main_dir": str(main_d.relative_to(ROOT)), "extra_dir": str(extra_d.relative_to(ROOT)),
        "conditions_from_extra": sorted(take_extra), "overlap_agreement": agree}}
    summary["provenance"] = provenance(vars(args), __file__, extra={"extra_run_provenance": old.get("provenance")})
    (main_d / "summary.json").write_text(json.dumps(summary, indent=1))
    print(f"merged {len(merged)} conditions ({len(take_extra)} from extra) into {main_d}")
    print("overlap agreement:", {c: f"{v['same_pred']}/{v['n']}" for c, v in agree.items()})


if __name__ == "__main__":
    main()
