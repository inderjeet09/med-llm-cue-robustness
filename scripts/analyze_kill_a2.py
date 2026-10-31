"""Apply the pre-registered stage-2 GO rule (docs/preregistration.md) to the Kaggle outputs.

Inputs  results/pilot/killA/<model>/n300_s0/records.jsonl     stage 1 (base: clean, misleading, +edu, +helpful)
        results/pilot/killA2/<model>/<arm>_s0/records.jsonl   fine-tuned arms
        results/pilot/killA2/<model>/base_extra/records.jsonl education prompt on clean / helpful (optional)
Output  results/pilot/killA2/analysis.json and analysis.md
Usage   python scripts/analyze_kill_a2.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.evaluation.stats import paired_bootstrap_diff  # noqa: E402
from src.utils.provenance import provenance  # noqa: E402

BIASES = ["false_consensus", "frequency", "recency"]
MODELS = ["Qwen__Qwen2.5-7B-Instruct", "meta-llama__Llama-3.1-8B-Instruct"]
ARMS = ["all7", "heldout4", "sft_clean"]


def load(path: Path) -> dict[str, dict[str, float]]:
    """{cond: {qid: correct (0/1)}}"""
    by: dict[str, dict[str, float]] = {}
    for r in map(json.loads, path.open()):
        by.setdefault(r["cond"], {})[r["qid"]] = float(r["correct"])
    return by


def vec(by: dict, cond: str, qids: list[str]) -> np.ndarray:
    return np.array([by[cond][q] for q in qids])


def pp(d: dict) -> dict:
    return {"diff_pp": 100 * d["diff"], "ci95_pp": [100 * x for x in d["ci95"]]}


def analyse_arm(base: dict, ft: dict, qids: list[str]) -> dict:
    mis_b = np.mean([vec(base, b, qids) for b in BIASES], 0)          # per-question mean over the 3 cues
    mis_f = np.mean([vec(ft, b, qids) for b in BIASES], 0)
    mis_e = np.mean([vec(base, f"{b}+edu", qids) for b in BIASES], 0)
    clean_b, clean_f = vec(base, "clean", qids), vec(ft, "clean", qids)
    drop = clean_b.mean() - mis_b.mean()
    rec = (mis_f.mean() - mis_b.mean()) / drop if drop > 0 else None
    clean_cost = pp(paired_bootstrap_diff(clean_b, clean_f))
    vs_edu = pp(paired_bootstrap_diff(mis_f, mis_e))
    helpful = {b: {"base": 100 * vec(base, f"{b}+helpful", qids).mean(), "ft": 100 * vec(ft, f"{b}+helpful", qids).mean()}
               for b in BIASES}
    crit = {"recovery>=0.5": bool(rec is not None and rec >= 0.5),
            "clean_cost<=2pp": bool(clean_cost["diff_pp"] <= 2),
            "beats_edu_CI>0": bool(vs_edu["ci95_pp"][0] > 0)}
    return {"acc_pp": {"clean_base": 100 * clean_b.mean(), "clean_ft": 100 * clean_f.mean(),
                       "misleading_base": 100 * mis_b.mean(), "misleading_edu": 100 * mis_e.mean(),
                       "misleading_ft": 100 * mis_f.mean(),
                       **{f"{b}_ft": 100 * vec(ft, b, qids).mean() for b in BIASES}},
            "recovery_fraction": None if rec is None else float(rec), "clean_cost": clean_cost, "ft_minus_edu_misleading": vs_edu,
            "helpful_acc_pp": helpful, "criteria": crit, "pass": all(crit.values())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1", default=str(ROOT / "results/pilot/killA"))
    ap.add_argument("--stage2", default=str(ROOT / "results/pilot/killA2"))
    args = ap.parse_args()
    s1, s2 = Path(args.stage1), Path(args.stage2)
    res: dict = {"models": {}}
    for m in MODELS:
        base = load(s1 / m / "n300_s0/records.jsonl")
        qids = sorted(base["clean"])
        res["models"][m] = {}
        extra = s2 / m / "base_extra/records.jsonl"
        if extra.exists():
            ex = load(extra)
            res["models"][m]["education_side_effects_pp"] = {
                "clean_base": 100 * vec(base, "clean", qids).mean(),
                **{f"clean+edu:{b}": 100 * vec(ex, f"clean+edu:{b}", qids).mean() for b in BIASES},
                **{f"{b}+helpful": 100 * vec(base, f"{b}+helpful", qids).mean() for b in BIASES},
                **{f"{b}+helpful+edu": 100 * vec(ex, f"{b}+helpful+edu", qids).mean() for b in BIASES}}
        for arm in ARMS:
            p = s2 / m / f"{arm}_s0/records.jsonl"
            if p.exists():
                ft = load(p)
                assert sorted(ft["clean"]) == qids, f"{m}/{arm}: evaluated on different questions than stage 1"
                res["models"][m][arm] = analyse_arm(base, ft, qids)
    decisive = [res["models"][m].get("all7", {}).get("pass") for m in MODELS]
    res["stage2_decision"] = ("TBD (missing all7 results)" if None in decisive else
                              "GO" if all(decisive) else "MIXED" if any(decisive) else "NO-GO")
    res["rule"] = "per model (all7): recovery>=0.5 AND clean cost<=2pp AND FT-minus-edu CI lower bound > 0"
    res["provenance"] = provenance(vars(args), __file__)
    (s2 / "analysis.json").write_text(json.dumps(res, indent=1))

    lines = ["| model | arm | clean base→FT | misleading base / edu / FT | recovery | clean cost (CI) | FT−edu (CI) | pass |",
             "|---|---|---|---|---|---|---|---|"]
    for m, d in res["models"].items():
        for arm in ARMS:
            if arm not in d:
                continue
            a = d[arm]; acc = a["acc_pp"]
            lines.append(f"| {m.split('__')[1]} | {arm} | {acc['clean_base']:.1f}→{acc['clean_ft']:.1f} | "
                         f"{acc['misleading_base']:.1f} / {acc['misleading_edu']:.1f} / {acc['misleading_ft']:.1f} | "
                         f"{a['recovery_fraction']:.0%} | {a['clean_cost']['diff_pp']:+.1f} "
                         f"[{a['clean_cost']['ci95_pp'][0]:.1f}, {a['clean_cost']['ci95_pp'][1]:.1f}] | "
                         f"{a['ft_minus_edu_misleading']['diff_pp']:+.1f} [{a['ft_minus_edu_misleading']['ci95_pp'][0]:.1f}, "
                         f"{a['ft_minus_edu_misleading']['ci95_pp'][1]:.1f}] | {'yes' if a['pass'] else 'no'} |")
    md = "\n".join(lines) + f"\n\nStage 2 decision (all7, both models): **{res['stage2_decision']}**\n"
    (s2 / "analysis.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
