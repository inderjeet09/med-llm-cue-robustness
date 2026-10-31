"""Phase 5 analysis: apply the frozen hypotheses H1–H4 (docs/preregistration.md B6, D020) to results/main.

Per model, per arm: per-question scores on the TEST split:
  clean            1/0
  train_cues       mean over the 7 BiasMedQA misleading cues
  heldout_cues     mean over the 8 new-family misleading cues
  helpful          mean over the 8 helpful conditions (base + trained arms)
Trained arms have one vector per seed. Differences use the two-level bootstrap (seeds x questions, D012); an untrained
arm enters as a single "seed". The best prompt baseline per model is chosen on DEV by mean misleading accuracy over
all 15 cues (frozen rule).
Writes results/tables/main_results.{json,md}.
Usage: python scripts/analyze_main.py [--root results/main]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.bias import BIASMEDQA, NEW_BIASES  # noqa: E402
from src.evaluation.stats import two_level_bootstrap  # noqa: E402
from src.utils.provenance import provenance  # noqa: E402

PROMPTS = ["warn_generic", "warn_oracle", "fewshot"]
TRAINED = ["cift", "sft_clean", "bct", "cift_nokl", "cift_nohelp", "cift_all15"]
METRICS = ["clean", "train_cues", "heldout_cues", "helpful"]


def per_question(run: Path) -> dict[str, np.ndarray]:
    by: dict[str, dict[str, float]] = {}
    for r in map(json.loads, (run / "records.jsonl").open()):
        by.setdefault(r["cond"], {})[r["qid"]] = float(r["correct"])
    q = sorted(by["clean"])
    v = lambda c: np.array([by[c][i] for i in q])  # noqa: E731
    out = {"clean": v("clean"), "train_cues": np.mean([v(b) for b in BIASMEDQA], 0),
           "heldout_cues": np.mean([v(b) for b in NEW_BIASES], 0), "_qids": q}
    hel = [c for c in by if c.endswith("+helpful")]
    if hel:
        out["helpful"] = np.mean([v(c) for c in hel], 0)
    return out


def collect(model_dir: Path, split: str) -> dict[str, list[dict]]:
    arms: dict[str, list[dict]] = {}
    for run in sorted(model_dir.iterdir()):
        if not (run / "records.jsonl").exists() or not run.name.endswith(f"_{split}"):
            continue
        mt = re.match(r"(.+?)(?:_s(\d+))?_" + split + "$", run.name)
        arms.setdefault(mt.group(1), []).append(per_question(run))
    return arms


def best_prompt_on_dev(model_dir: Path) -> str | None:
    best, val = None, -1.0
    for a in PROMPTS:
        s = model_dir / f"{a}_dev" / "summary.json"
        if s.exists():
            d = json.loads(s.read_text())
            m = np.mean([d["misleading_acc_train_cues"] * 7, d["misleading_acc_heldout_cues"] * 8]) * 2 / 15
            if m > val:
                best, val = a, m
    return best


def cmp(a: list[dict], b: list[dict], metric: str) -> dict:
    A = np.stack([x[metric] for x in a]); B = np.stack([x[metric] for x in b])
    r = two_level_bootstrap(A, B, seed=0)
    return {"diff_pp": 100 * r["diff"], "ci95_pp": [100 * x for x in r["ci95"]], "p": r["p_value"],
            "ci_excludes_0_positive": bool(r["ci95"][0] > 0)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(ROOT / "results/main"))
    args = ap.parse_args()
    res: dict = {"models": {}}
    for md in sorted(p for p in Path(args.root).iterdir() if p.is_dir()):
        arms = collect(md, "test")
        if not arms:
            continue
        qsets = {tuple(x["_qids"]) for runs in arms.values() for x in runs}
        assert len(qsets) == 1, f"{md.name}: arms evaluated on different questions"
        M: dict = {"arms": {}, "best_prompt_dev": best_prompt_on_dev(md)}
        for a, runs in arms.items():
            M["arms"][a] = {"n_seeds": len(runs), **{m: {"mean": 100 * float(np.mean([r[m].mean() for r in runs])),
                                                        "std": 100 * float(np.std([r[m].mean() for r in runs], ddof=1))
                                                        if len(runs) > 1 else None}
                                                    for m in METRICS if m in runs[0]}}
        if "cift" in arms:
            H = {}
            refs = {"base": "base", "best_prompt": M["best_prompt_dev"], "sft_clean": "sft_clean", "bct": "bct"}
            for name, ref in refs.items():
                if ref and ref in arms:
                    H[f"H1_heldout_vs_{name}"] = cmp(arms["cift"], arms[ref], "heldout_cues")
                    H[f"H2_traincues_vs_{name}"] = cmp(arms["cift"], arms[ref], "train_cues")
            if "base" in arms:
                d = 100 * (np.mean([r["clean"].mean() for r in arms["cift"]]) - arms["base"][0]["clean"].mean())
                H["H3_clean_vs_base_pp"] = {"diff_pp": float(d), "pass": bool(d >= -2)}
            if "sft_clean" in arms:
                c = cmp(arms["cift"], arms["sft_clean"], "clean")
                H["H3_clean_vs_sft"] = {**c, "pass": bool(c["ci95_pp"][0] > -2)}
            if "cift_nokl" in arms:
                H["H4_heldout_vs_nokl"] = cmp(arms["cift"], arms["cift_nokl"], "heldout_cues")
            if "cift_nohelp" in arms and "helpful" in arms["cift"][0]:
                H["H4_helpful_vs_nohelp"] = cmp(arms["cift"], arms["cift_nohelp"], "helpful")
            M["hypotheses"] = H
        res["models"][md.name] = M
    res["provenance"] = provenance(vars(args), __file__)
    out = ROOT / "results/tables"
    out.mkdir(parents=True, exist_ok=True)
    (out / "main_results.json").write_text(json.dumps(res, indent=1))

    L = ["# Phase 5 main results (MedQA test; mean ± std over seeds; two-level bootstrap CIs)", ""]
    for mname, M in res["models"].items():
        L += [f"## {mname}", f"Best prompt baseline (dev rule): {M['best_prompt_dev']}", "",
              "| arm | seeds | clean | misleading: training cues | misleading: held-out cues | helpful |",
              "|---|---|---|---|---|---|"]
        for a, d in M["arms"].items():
            f = lambda m: (f"{d[m]['mean']:.1f}" + (f" ± {d[m]['std']:.1f}" if d[m]["std"] is not None else "")) if m in d else "–"  # noqa: E731
            L.append(f"| {a} | {d['n_seeds']} | {f('clean')} | {f('train_cues')} | {f('heldout_cues')} | {f('helpful')} |")
        if "hypotheses" in M:
            L += ["", "| test | diff (pp) | 95% CI | pass |", "|---|---|---|---|"]
            for k, h in M["hypotheses"].items():
                ci_s = f"[{h['ci95_pp'][0]:.1f}, {h['ci95_pp'][1]:.1f}]" if "ci95_pp" in h else ""
                ok = h.get("pass", h.get("ci_excludes_0_positive"))
                L.append(f"| {k} | {h['diff_pp']:+.1f} | {ci_s} | {'yes' if ok else 'no'} |")
        L.append("")
    (out / "main_results.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
