"""Exploratory comparisons (not pre-registered, except the two H4 tests that analyze_main.py also reports):
  CIFT vs BCT per model       -> results/tables/exploratory_cift_vs_bct.json
  CIFT vs each Llama ablation -> results/tables/ablations_llama.json
Same per-question scores and the same two-level bootstrap (seeds x questions, D012) as scripts/analyze_main.py, plus the
transfer test from transfer_records.jsonl (accuracy with the misleading evidence paragraph) and the residual drop on
held-out cues (clean minus held-out, per question).
Usage: python scripts/analyze_exploratory.py [--root results/main]   (after scripts/unpack_results.py)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from analyze_main import collect  # noqa: E402
from src.evaluation.stats import two_level_bootstrap  # noqa: E402
from src.utils.provenance import provenance  # noqa: E402

MODELS = ["Qwen__Qwen2.5-7B-Instruct", "meta-llama__Llama-3.1-8B-Instruct", "m42-health__Llama3-Med42-8B"]
METRICS = ["clean", "train_cues", "heldout_cues", "helpful"]


def transfer_scores(model_dir: Path, arm: str) -> list[np.ndarray]:
    """Per seed: 0/1 correctness on the misleading-context transfer condition, items sorted by id."""
    out = []
    for run in sorted(model_dir.glob(f"{arm}_s*_test")):
        by = {}
        for r in map(json.loads, (run / "transfer_records.jsonl").open()):
            if r["cond"] == "misleading_context":
                by[r["qid"]] = float(r["correct"])
        out.append(np.array([by[q] for q in sorted(by)]))
    return out


def diff(a: list[np.ndarray], b: list[np.ndarray]) -> tuple[float, float, float]:
    r = two_level_bootstrap(np.stack(a), np.stack(b), seed=0)
    return 100 * r["diff"], 100 * r["ci95"][0], 100 * r["ci95"][1]


def compare(md: Path, arms: dict, a: str, b: str) -> dict:
    res = {m: diff([x[m] for x in arms[a]], [x[m] for x in arms[b]]) for m in METRICS}
    res["residual_drop_heldout"] = diff([x["clean"] - x["heldout_cues"] for x in arms[a]],
                                        [x["clean"] - x["heldout_cues"] for x in arms[b]])
    res["transfer_misleading_context"] = diff(transfer_scores(md, a), transfer_scores(md, b))
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(ROOT / "results/main"))
    args = ap.parse_args()
    root = Path(args.root)

    expl = {}
    for m in MODELS:
        arms = collect(root / m, "test")
        if "cift" in arms and "bct" in arms:
            expl[m] = {k: {"diff_pp": d, "ci95_pp": [lo, hi]} for k, (d, lo, hi) in compare(root / m, arms, "cift", "bct").items()}

    abl = {}
    llama = root / "meta-llama__Llama-3.1-8B-Instruct"
    arms = collect(llama, "test")
    for other in ("cift_nokl", "cift_nohelp"):
        if other in arms:
            c = compare(llama, arms, "cift", other)
            abl[other] = {k: list(c[k]) for k in METRICS}
            abl[other]["transfer_misleading"] = list(c["transfer_misleading_context"])

    out = ROOT / "results/tables"
    prov = provenance(vars(args), __file__)
    (out / "exploratory_cift_vs_bct.json").write_text(json.dumps({
        "note": "EXPLORATORY (not pre-registered): CIFT vs BCT, two-level bootstrap seeds x questions, test split",
        "results": expl, "provenance": prov}, indent=1))
    (out / "ablations_llama.json").write_text(json.dumps({
        "note": "Llama ablations, CIFT minus ablation, two-level bootstrap; H4 heldout_vs_nokl and helpful_vs_nohelp "
                "are pre-registered, the rest exploratory",
        "results": abl, "provenance": prov}, indent=1))
    for m, r in expl.items():
        print(m, {k: f"{v['diff_pp']:+.1f} [{v['ci95_pp'][0]:.1f}, {v['ci95_pp'][1]:.1f}]" for k, v in r.items()})
    print("-> results/tables/exploratory_cift_vs_bct.json, results/tables/ablations_llama.json")


if __name__ == "__main__":
    main()
