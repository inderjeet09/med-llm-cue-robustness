"""Phase 4 gap analysis over results/gap_analysis/<run>/records.jsonl (DEV only).

Per model:
  - drop (clean - misleading) per cue template and family, with paired bootstrap 95% CI; BiasMedQA vs new families
  - cue-follow rate; helpful-cue gain; education ("warning") recovery per template / family
  - flips: among questions correct when clean, the share that switch to the cued option, by exam step, question
    length tercile and clean-confidence tercile (open models only: confidence = clean max letter probability)
  - calibration (open models): ECE and mean confidence, clean vs misleading; confidence in wrong cued answers
  - susceptibility overlap: on how many of the 15 cues each clean-correct question flips
  - cost / latency per 1,000 questions
Writes results/gap_analysis/analysis.{json,md} and figures/04_drop_by_family.png.
Usage: python scripts/analyze_gap.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.bias import BIASMEDQA, FAMILIES, FAMILY_OF, NEW_BIASES, TEMPLATES  # noqa: E402
from src.evaluation.stats import paired_bootstrap_diff  # noqa: E402
from src.utils.provenance import provenance  # noqa: E402

GAP = ROOT / "results/gap_analysis"


def load(run: Path) -> dict[str, dict[str, dict]]:
    by: dict[str, dict[str, dict]] = {}
    for r in map(json.loads, (run / "records.jsonl").open()):
        by.setdefault(r["cond"], {})[r["qid"]] = r
    return by


def ece(conf: np.ndarray, correct: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(conf[m].mean() - correct[m].mean())
    return float(100 * e)


def ci(a: np.ndarray, b: np.ndarray) -> dict:
    d = paired_bootstrap_diff(a, b)
    return {"pp": 100 * d["diff"], "ci95": [100 * x for x in d["ci95"]]}


def analyse(run: Path) -> dict:
    by = load(run)
    q = sorted(by["clean"])
    v = lambda c, k="correct": np.array([float(by[c][i][k]) for i in q])  # noqa: E731
    clean = v("clean")
    has_probs = "probs" in by["clean"][q[0]]
    mis = [b for b in TEMPLATES if b in by]
    res: dict = {"n": len(q), "clean_acc": 100 * clean.mean(), "per_template": {}, "per_family": {}}
    for b in mis:
        t = {"acc": 100 * v(b).mean(), "drop": ci(clean, v(b)),
             "cue_follow": 100 * float(np.mean([by[b][i]["pred"] == by[b][i]["cue_target"] for i in q]))}
        if f"{b}+helpful" in by:
            t["helpful_gain"] = ci(v(f"{b}+helpful"), clean)
        if f"{b}+edu" in by:
            de = 100 * (clean.mean() - v(b).mean())
            t["edu_acc"] = 100 * v(f"{b}+edu").mean()
            t["edu_recovery"] = (t["edu_acc"] - t["acc"]) / de if de > 0 else None
        res["per_template"][b] = t
    for f, bs in FAMILIES.items():
        bs = [b for b in bs if b in mis]
        if not bs:
            continue
        mis_f = np.mean([v(b) for b in bs], 0)
        row = {"templates": bs, "new": bs[0] in NEW_BIASES, "drop": ci(clean, mis_f)}
        if all(f"{b}+edu" in by for b in bs):
            edu_f = np.mean([v(f"{b}+edu") for b in bs], 0)
            row["edu_recovery"] = float((edu_f.mean() - mis_f.mean()) / (clean.mean() - mis_f.mean())) \
                if clean.mean() > mis_f.mean() else None
        if all(f"{b}+helpful" in by for b in bs):
            row["helpful_gain"] = ci(np.mean([v(f"{b}+helpful") for b in bs], 0), clean)
        res["per_family"][f] = row
    for group, bs in (("biasmedqa", BIASMEDQA), ("new", NEW_BIASES)):
        bs = [b for b in bs if b in mis]
        if bs:
            res[f"drop_{group}"] = ci(clean, np.mean([v(b) for b in bs], 0))
    if any(b in mis for b in BIASMEDQA) and any(b in mis for b in NEW_BIASES):  # >0: new cues hurt more
        res["new_minus_biasmedqa_drop"] = ci(np.mean([v(b) for b in BIASMEDQA if b in mis], 0),
                                             np.mean([v(b) for b in NEW_BIASES if b in mis], 0))

    # flips among clean-correct questions
    cc = clean == 1
    flip = np.array([[by[b][i]["pred"] == by[b][i]["cue_target"] for b in mis] for i in q], float)  # [n, T]
    n_flip = flip[cc].sum(1)
    res["flip"] = {"share_of_clean_correct_flipping_on_>=1_cue": 100 * float((n_flip >= 1).mean()),
                   "share_flipping_on_>=half_of_cues": 100 * float((n_flip >= len(mis) / 2).mean()),
                   "share_never_flipping": 100 * float((n_flip == 0).mean()),
                   "mean_cues_flipped": float(n_flip.mean()), "n_cues": len(mis)}
    rate = flip.mean(1)  # per-question flip rate over cues
    step = np.array([by["clean"][i].get("step") for i in q])
    words = np.array([by["clean"][i].get("q_words", 0) for i in q], float)
    br: dict = {"by_step": {s: 100 * float(rate[cc & (step == s)].mean()) for s in sorted(set(step)) if s}}
    t1, t2 = np.percentile(words, [33.3, 66.7])
    br["by_length_tercile"] = {"short": 100 * float(rate[cc & (words <= t1)].mean()),
                               "medium": 100 * float(rate[cc & (words > t1) & (words <= t2)].mean()),
                               "long": 100 * float(rate[cc & (words > t2)].mean())}
    if has_probs:
        conf = np.array([max(by["clean"][i]["probs"].values()) for i in q])
        # equal-size terciles by RANK among clean-correct questions (confidence values tie heavily near 1.0)
        idx = np.where(cc)[0][np.argsort(conf[cc], kind="stable")]
        thirds = np.array_split(idx, 3)
        br["by_clean_confidence_tercile"] = {name: 100 * float(rate[t].mean())
                                             for name, t in zip(("low", "mid", "high"), thirds)}
        res["clean_confidence_tercile_lower_bounds"] = [float(conf[t].min()) for t in thirds]
        mconf = np.array([[max(by[b][i]["probs"].values()) for b in mis] for i in q])
        mcorr = np.array([[float(by[b][i]["correct"]) for b in mis] for i in q])
        wrong_cued = flip.astype(bool) & (mcorr == 0)
        res["calibration"] = {
            "ece_clean": ece(conf, clean), "ece_misleading": ece(mconf.ravel(), mcorr.ravel()),
            "mean_conf_clean": 100 * float(conf.mean()), "mean_conf_misleading": 100 * float(mconf.mean()),
            "acc_misleading": 100 * float(mcorr.mean()),
            "mean_conf_when_following_wrong_cue": 100 * float(mconf[wrong_cued].mean()) if wrong_cued.any() else None,
            "share_wrong_cued_answers_with_conf_>=0.9": 100 * float((mconf[wrong_cued] >= 0.9).mean())
            if wrong_cued.any() else None}
    res["flip_rate_breakdown_pct"] = br

    s = json.loads((run / "summary.json").read_text()) if (run / "summary.json").exists() else {}
    sec = s.get("meta", {}).get("seconds_per_condition", {})
    if sec:
        res["latency_ms_per_question"] = 1000 * float(np.median(list(sec.values()))) / len(q)
    if "api_cost_usd_this_run" in s.get("meta", {}):
        n_calls = len(q) * len(by)
        res["api_usd_per_1000_questions"] = 1000 * s["meta"]["api_cost_usd_this_run"] / n_calls if n_calls else None
    return res


def main() -> None:
    runs = sorted(p for p in GAP.iterdir() if (p / "records.jsonl").exists() and not p.name.startswith("_"))
    out = {r.name: analyse(r) for r in runs}
    out["provenance"] = provenance({}, __file__)
    (GAP / "analysis.json").write_text(json.dumps(out, indent=1))

    L = ["# Phase 4 gap analysis (dev, 4-option, n per model below)", "",
         "Generated by `scripts/analyze_gap.py`. Drops in pp with 95% paired bootstrap CI.", "",
         "## Drop by family", "", "| family | new? | " + " | ".join(r.name.split("_dev")[0] for r in runs) + " |",
         "|---|---|" + "---|" * len(runs)]
    for f in FAMILIES:
        cells = []
        for r in runs:
            d = out[r.name]["per_family"].get(f)
            cells.append(f"{d['drop']['pp']:.1f} [{d['drop']['ci95'][0]:.1f}, {d['drop']['ci95'][1]:.1f}]" if d else "–")
        L.append(f"| {f} | {'yes' if f in {FAMILY_OF[b] for b in NEW_BIASES} else 'no'} | " + " | ".join(cells) + " |")
    L += ["", "## Summary per model", ""]
    for r in runs:
        o = out[r.name]
        L.append(f"### {r.name}")
        line = f"- clean accuracy {o['clean_acc']:.1f} %"
        for g in ("biasmedqa", "new"):
            if f"drop_{g}" in o:
                line += f"; drop {g} cues {o[f'drop_{g}']['pp']:.1f} [{o[f'drop_{g}']['ci95'][0]:.1f}, {o[f'drop_{g}']['ci95'][1]:.1f}]"
        if "new_minus_biasmedqa_drop" in o:
            d = o["new_minus_biasmedqa_drop"]
            line += f"; new − BiasMedQA {d['pp']:+.1f} [{d['ci95'][0]:.1f}, {d['ci95'][1]:.1f}]"
        L.append(line)
        fl = o["flip"]
        L.append(f"- of clean-correct questions: {fl['share_of_clean_correct_flipping_on_>=1_cue']:.0f} % flip on ≥1 "
                 f"of {fl['n_cues']} cues, {fl['share_flipping_on_>=half_of_cues']:.0f} % on ≥ half, "
                 f"{fl['share_never_flipping']:.0f} % never")
        L.append(f"- flip rate by breakdown: {json.dumps({k: {kk: round(vv, 1) for kk, vv in v.items()} for k, v in o['flip_rate_breakdown_pct'].items()})}")
        if "calibration" in o:
            c = o["calibration"]
            L.append(f"- calibration: ECE clean {c['ece_clean']:.1f}, misleading {c['ece_misleading']:.1f}; mean "
                     f"confidence when following a wrong cue {c['mean_conf_when_following_wrong_cue']:.1f} % "
                     f"({c['share_wrong_cued_answers_with_conf_>=0.9']:.0f} % of those ≥ 0.9)")
        fam_edu = {f: round(d["edu_recovery"], 2) for f, d in o["per_family"].items() if d.get("edu_recovery") is not None}
        if fam_edu:
            L.append(f"- education recovery (fraction of drop) by family: {fam_edu}")
        if "latency_ms_per_question" in o:
            L.append(f"- median latency {o['latency_ms_per_question']:.0f} ms/question"
                     + (f"; API ${o['api_usd_per_1000_questions']:.3f} per 1,000 calls" if o.get("api_usd_per_1000_questions") else ""))
        L.append("")
    (GAP / "analysis.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fams = list(FAMILIES)
    fig, ax = plt.subplots(figsize=(9, 3.8))
    w = 0.8 / max(1, len(runs))
    for k, r in enumerate(runs):
        ys = [out[r.name]["per_family"].get(f, {}).get("drop", {}).get("pp", np.nan) for f in fams]
        ax.bar(np.arange(len(fams)) + k * w, ys, w, label=r.name.split("_dev")[0].split("__")[-1])
    ax.set_xticks(np.arange(len(fams)) + w * (len(runs) - 1) / 2)
    ax.set_xticklabels(fams, rotation=25, ha="right", fontsize=8)
    ax.axvline(3.5 + w * (len(runs) - 1) / 2 + 0.1, color="grey", ls=":", lw=1)
    ax.set_ylabel("accuracy drop (pp)"); ax.legend(fontsize=7); ax.set_title("Misleading-cue drop by family (dev)")
    fig.tight_layout(); fig.savefig(ROOT / "figures/04_drop_by_family.png", dpi=150)


if __name__ == "__main__":
    main()
