"""Compare a re-run of the main experiments with the shipped results (for reviewers).

For every run folder present in both trees (<model>/<arm>[_s<seed>]_<split>/records.jsonl[.gz]):
  - answer agreement: share of (question, condition) pairs with the same predicted letter
  - accuracy difference per condition (max and mean absolute)
  - headline metrics: clean, misleading (7 training cues), misleading (8 held-out cues), helpful, transfer
A headline metric differing by more than --tol percentage points is flagged. Reference for what "the same" means:
re-running identical code on the same GPU type with different batching changed 0.8-2.7 % of answers and moved
accuracy by <= 1.0 pp per condition (docs/DECISIONS.md D019); different library versions or GPUs can add to this.
Shipped records may stay gzipped. Exit code 1 if any headline metric exceeds --tol.

Usage: python scripts/compare_rerun.py reruns/results/main [--shipped results/main] [--tol 1.0] [--json out.json]
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.bias import BIASMEDQA, NEW_BIASES  # noqa: E402


def read_jsonl(folder: Path, name: str) -> list[dict] | None:
    for p, opener in ((folder / name, open), (folder / f"{name}.gz", gzip.open)):
        if p.exists():
            with opener(p, "rt") as f:
                return [json.loads(line) for line in f]
    return None


def by_cond(rows: list[dict]) -> dict[str, dict[str, dict]]:
    out: dict[str, dict[str, dict]] = {}
    for r in rows:
        out.setdefault(r["cond"], {})[r["qid"]] = r
    return out


def headline(acc: dict[str, float], transfer: float | None) -> dict[str, float]:
    h = {"clean": acc.get("clean")}
    if all(b in acc for b in BIASMEDQA):
        h["train_cues"] = float(np.mean([acc[b] for b in BIASMEDQA]))
    if all(b in acc for b in NEW_BIASES):
        h["heldout_cues"] = float(np.mean([acc[b] for b in NEW_BIASES]))
    hel = [v for c, v in acc.items() if c.endswith("+helpful")]
    if hel:
        h["helpful"] = float(np.mean(hel))
    if transfer is not None:
        h["transfer_misleading"] = transfer
    return h


def transfer_acc(folder: Path) -> float | None:
    s = folder / "summary.json"
    if s.exists():
        t = json.loads(s.read_text()).get("transfer")
        if t:
            return t["misleading_context"]
    return None


def compare_run(a_dir: Path, b_dir: Path) -> dict | None:
    A, B = read_jsonl(a_dir, "records.jsonl"), read_jsonl(b_dir, "records.jsonl")
    if A is None or B is None:
        return None
    A, B = by_cond(A), by_cond(B)
    conds = sorted(set(A) & set(B))
    same = total = 0
    acc_a, acc_b = {}, {}
    for c in conds:
        q = sorted(set(A[c]) & set(B[c]))
        same += sum(A[c][i]["pred"] == B[c][i]["pred"] for i in q)
        total += len(q)
        acc_a[c] = 100 * float(np.mean([A[c][i]["correct"] for i in q]))
        acc_b[c] = 100 * float(np.mean([B[c][i]["correct"] for i in q]))
    d = [abs(acc_a[c] - acc_b[c]) for c in conds]
    ha, hb = headline(acc_a, transfer_acc(a_dir)), headline(acc_b, transfer_acc(b_dir))
    return {"conditions": len(conds), "pairs": total, "agreement_pct": 100 * same / total if total else None,
            "max_abs_cond_diff_pp": max(d) if d else None, "mean_abs_cond_diff_pp": float(np.mean(d)) if d else None,
            "headline": {k: {"rerun": ha[k], "shipped": hb[k], "diff_pp": ha[k] - hb[k]}
                         for k in ha if k in hb and ha[k] is not None and hb[k] is not None}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("rerun", help="re-run tree laid out like results/main")
    ap.add_argument("--shipped", default=str(ROOT / "results/main"))
    ap.add_argument("--tol", type=float, default=1.0, help="flag headline metrics differing by more pp than this")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    rerun, shipped = Path(args.rerun), Path(args.shipped)

    res, flagged = {}, []
    for run in sorted(p for p in rerun.glob("*/*") if p.is_dir()):
        other = shipped / run.parent.name / run.name
        if not other.is_dir():
            print(f"skip {run.parent.name}/{run.name}: no shipped counterpart")
            continue
        r = compare_run(run, other)
        if r is None:
            continue
        key = f"{run.parent.name}/{run.name}"
        res[key] = r
        flagged += [(key, k) for k, v in r["headline"].items() if abs(v["diff_pp"]) > args.tol]
    if not res:
        sys.exit(f"no matching runs between {rerun} and {shipped}")

    print(f"\n| run | agreement | max / mean cond. diff (pp) | " + " | ".join(
        ["clean", "train cues", "held-out cues", "helpful", "transfer"]) + " |")
    print("|---|---|---|" + "---|" * 5)
    for key, r in res.items():
        h = r["headline"]
        cell = lambda k: (f"{h[k]['rerun']:.1f} vs {h[k]['shipped']:.1f} ({h[k]['diff_pp']:+.1f})" if k in h else "–")  # noqa: E731
        print(f"| {key} | {r['agreement_pct']:.1f} % | {r['max_abs_cond_diff_pp']:.1f} / {r['mean_abs_cond_diff_pp']:.2f} | "
              + " | ".join(cell(k) for k in ("clean", "train_cues", "heldout_cues", "helpful", "transfer_misleading")) + " |")
    print(f"\nHeadline cells are 'rerun vs shipped (difference)'. Tolerance {args.tol} pp. "
          "Noise reference (D019): >= 97.3 % agreement, <= 1.0 pp per condition for identical code on the same GPU type.")
    if args.json:
        Path(args.json).write_text(json.dumps({"tolerance_pp": args.tol, "runs": res, "flagged": flagged}, indent=1))
    if flagged:
        print("OUTSIDE TOLERANCE:", ", ".join(f"{k} {m}" for k, m in flagged))
        sys.exit(1)
    print("All headline metrics within tolerance.")


if __name__ == "__main__":
    main()
