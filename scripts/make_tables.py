"""Phase 6: write every number the paper uses, directly from results/ (integrity rules 1-2).

Outputs
  paper/numbers.tex         \\newcommand macros for every number cited in the text (e.g. \\QwenCIFTheld)
  paper/tables/main.tex     Table: all arms x models (+ gpt-4.1-mini), mean ± std over seeds
  paper/tables/hyp.tex      Table: pre-registered hypotheses, differences with two-level bootstrap 95% CIs
  paper/tables/gap.tex      Table: dev gap analysis, drop per hint family (base models)
  paper/tables/repro.tex    Table: BiasMedQA reproduction (paper vs ours, no mitigation)
  paper/tables/explore.tex  Table: exploratory CIFT vs BCT + Llama ablations
Re-run after any change in results/.
Usage: python scripts/make_tables.py
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from make_figures import BEST_PROMPT, MODELS, agg  # noqa: E402  (same aggregation as the figures)

TAB = ROOT / "paper/tables"
SHORT = {"Qwen__Qwen2.5-7B-Instruct": "Qwen", "meta-llama__Llama-3.1-8B-Instruct": "Llama",
         "m42-health__Llama3-Med42-8B": "Medfortytwo"}  # macro-safe names (no digits in LaTeX macro names)
ARMNAME = {"base": "Base", "best": "Best", "sft_clean": "SFT", "bct": "BCT", "cift": "CIFT", "cift_nokl": "NoKL",
           "cift_nohelp": "NoHelp", "warn_generic": "Generic", "warn_oracle": "Oracle", "fewshot": "Fewshot"}
METRIC = {"clean": "clean", "train_cues": "train", "heldout_cues": "held", "helpful": "help", "transfer_mislead": "tr"}
macros: dict[str, str] = {}


_DIGITS = dict(zip("0123456789", ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]))


def mac(name: str, value) -> str:
    """Register \\name -> value. LaTeX macro names cannot contain digits, so digits become words (H1 -> Hone)."""
    name = "".join(_DIGITS.get(ch, ch) for ch in name)
    macros[name] = value if isinstance(value, str) else f"{value:.1f}"
    return macros[name]


def pm(v):  # signed difference
    return f"{v:+.1f}".replace("+", "+").replace("-", "−")


def ci(c):
    return f"[{c[0]:.1f}, {c[1]:.1f}]".replace("-", "−")


def main() -> None:
    TAB.mkdir(parents=True, exist_ok=True)
    J = lambda p: json.loads((ROOT / p).read_text())  # noqa: E731
    main_res, expl, abl = J("results/tables/main_results.json"), J("results/tables/exploratory_cift_vs_bct.json"), \
        J("results/tables/ablations_llama.json")
    gap, comp, ref = J("results/gap_analysis/analysis.json"), J("results/reproduction/comparison.json"), \
        J("results/reference/gpt41mini_test/summary.json")

    # ---- per model × arm metrics (test)
    rows = []
    arms = ["base", "warn_generic", "warn_oracle", "fewshot", "sft_clean", "bct", "cift", "cift_nokl", "cift_nohelp"]
    for m, name in MODELS.items():
        s = SHORT[m]
        for a in arms:
            d = agg(m, a)
            if not d:
                continue
            tags = [ARMNAME[a]] + (["Best"] if a == BEST_PROMPT[m] else [])
            for t in tags:  # derived: residual drop on held-out cues, and gain from a correct hint (both vs own clean)
                if "heldout_cues" in d:
                    mac(f"{s}{t}resid", d["clean"][0] - d["heldout_cues"][0])
                if "helpful" in d:
                    mac(f"{s}{t}helpgain", d["helpful"][0] - d["clean"][0])
            for k, short in METRIC.items():
                if k in d:
                    for t in tags:
                        mac(f"{s}{t}{short}", d[k][0])
                        if d["n"] > 1:
                            mac(f"{s}{t}{short}sd", d[k][1])
            label = {"base": "No mitigation", "warn_generic": "Generic warning", "warn_oracle": "Oracle warning",
                     "fewshot": "Few-shot", "sft_clean": "Plain SFT", "bct": "BCT", "cift": "\\textbf{CIFT (ours)}",
                     "cift_nokl": "\\quad CIFT w/o KL", "cift_nohelp": "\\quad CIFT w/o helpful pairs"}[a]
            if a == BEST_PROMPT[m]:
                label += "$^\\dagger$"
            cell = lambda k: (f"{d[k][0]:.1f}" + (f" $\\pm$ {d[k][1]:.1f}" if d["n"] > 1 else "")) if k in d else "--"  # noqa: E731
            rows.append(f"{name} & {label} & {d['n']} & {cell('clean')} & {cell('train_cues')} & {cell('heldout_cues')} & "
                        f"{cell('helpful')} & {cell('transfer_mislead')} \\\\")
        rows.append("\\midrule")
    a = ref["accuracy"]
    for k, v in (("GptClean", a["clean"]), ("GptTrain", ref["misleading_acc_train_cues"]),
                 ("GptHeld", ref["misleading_acc_heldout_cues"]), ("GptHeldOracle", ref["misleading_acc_heldout_cues_edu"]),
                 ("GptHeldGeneric", ref["misleading_acc_heldout_cues_gen"]), ("GptCleanGeneric", a["clean+gen"])):
        mac(k, v)
    mac("GptDropHeld", a["clean"] - ref["misleading_acc_heldout_cues"])
    rows += [f"gpt-4.1-mini & No mitigation & 1 & {a['clean']:.1f} & {ref['misleading_acc_train_cues']:.1f} & "
             f"{ref['misleading_acc_heldout_cues']:.1f} & -- & -- \\\\",
             f"(untrained) & Oracle warning & 1 & -- & {ref['misleading_acc_train_cues_edu']:.1f} & "
             f"{ref['misleading_acc_heldout_cues_edu']:.1f} & -- & -- \\\\",
             f" & Generic warning & 1 & {a['clean+gen']:.1f} & {ref['misleading_acc_train_cues_gen']:.1f} & "
             f"{ref['misleading_acc_heldout_cues_gen']:.1f} & -- & -- \\\\"]
    (TAB / "main.tex").write_text("\n".join([
        "\\begin{table*}[t]\\centering\\small",
        "\\caption{Accuracy (\\%) on the MedQA test split (1{,}273 questions). Trained arms: mean $\\pm$ s.d. over 3 seeds. "
        "\\emph{Training cues}: the 7 BiasMedQA templates; \\emph{held-out cues}: 8 new templates from 4 families never seen in "
        "training. \\emph{Helpful}: hint points to the correct option (mean of 8 conditions). \\emph{Transfer}: "
        "MedQA-MultiTurnRobust with its misleading evidence paragraph. $^\\dagger$best prompt baseline chosen on dev.}",
        "\\label{tab:main}",
        "\\resizebox{\\textwidth}{!}{\\begin{tabular}{llccccccc}\\toprule",
        "Model & Method & Seeds & Clean & \\multicolumn{2}{c}{Misleading cues} & Helpful & Transfer \\\\",
        "\\cmidrule(lr){5-6} & & & & Training & Held-out & & \\\\ \\midrule"] + rows + ["\\bottomrule\\end{tabular}}\\end{table*}"]) + "\n")

    # ---- hypotheses
    hrows = []
    names = {"H1_heldout_vs_base": ("H1", "held-out", "base"), "H1_heldout_vs_best_prompt": ("H1", "held-out", "best prompt"),
             "H1_heldout_vs_sft_clean": ("H1", "held-out", "plain SFT"), "H1_heldout_vs_bct": ("H1", "held-out", "BCT"),
             "H2_traincues_vs_sft_clean": ("H2", "training", "plain SFT"), "H2_traincues_vs_bct": ("H2", "training", "BCT"),
             "H3_clean_vs_sft": ("H3", "clean", "plain SFT")}
    for key, (h, met, vs) in names.items():
        cells = []
        for m in MODELS:
            x = main_res["models"][m]["hypotheses"].get(key)
            tag = SHORT[m] + key.replace("_", "").replace("vs", "Vs")
            mac(tag + "d", x["diff_pp"]); mac(tag + "lo", x["ci95_pp"][0]); mac(tag + "hi", x["ci95_pp"][1])
            ok = x.get("pass", x.get("ci_excludes_0_positive"))
            cells.append(f"{pm(x['diff_pp'])} {ci(x['ci95_pp'])}" + ("" if ok else "$^\\times$"))
        hrows.append(f"{h} & {met} & {vs} & " + " & ".join(cells) + " \\\\")
    for m in MODELS:
        x = main_res["models"][m]["hypotheses"]["H3_clean_vs_base_pp"]
        mac(SHORT[m] + "HthreeCleanVsBase", x["diff_pp"])
    hl = main_res["models"]["meta-llama__Llama-3.1-8B-Instruct"]["hypotheses"]
    for k, t in (("H4_heldout_vs_nokl", "HfourKL"), ("H4_helpful_vs_nohelp", "HfourHelp")):
        mac(t + "d", hl[k]["diff_pp"]); mac(t + "lo", hl[k]["ci95_pp"][0]); mac(t + "hi", hl[k]["ci95_pp"][1])
    hrows.append(f"H4a & held-out & w/o KL (Llama) & -- & {pm(hl['H4_heldout_vs_nokl']['diff_pp'])} "
                 f"{ci(hl['H4_heldout_vs_nokl']['ci95_pp'])}$^\\times$ & -- \\\\")
    hrows.append(f"H4b & helpful & w/o helpful pairs (Llama) & -- & {pm(hl['H4_helpful_vs_nohelp']['diff_pp'])} "
                 f"{ci(hl['H4_helpful_vs_nohelp']['ci95_pp'])} & -- \\\\")
    (TAB / "hyp.tex").write_text("\n".join([
        "\\begin{table*}[t]\\centering\\small",
        "\\caption{Pre-registered hypotheses (test split): CIFT minus comparison, percentage points, with two-level bootstrap "
        "95\\% CIs over training seeds and questions. $^\\times$: not supported by the frozen decision rule (CI includes 0, or "
        "for H3 the lower bound is below $-2$).}",
        "\\label{tab:hyp}",
        "\\resizebox{\\textwidth}{!}{\\begin{tabular}{lllccc}\\toprule",
        "Hyp. & Metric & CIFT vs & Qwen2.5-7B & Llama-3.1-8B & Med42-8B \\\\ \\midrule"] + hrows +
        ["\\bottomrule\\end{tabular}}\\end{table*}"]) + "\n")

    # ---- exploratory CIFT vs BCT + ablations
    erows = []
    lab = {"clean": "clean", "helpful": "helpful cue", "transfer_misleading_context": "transfer (evidence paragraph)",
           "heldout_cues": "held-out cues"}
    for k, l in lab.items():
        cells = []
        for m in MODELS:
            x = expl["results"][m][k]
            mac(SHORT[m] + "XBCT" + k.replace("_", "").replace("transfermisleadingcontext", "transfer") + "d", x["diff_pp"])
            cells.append(f"{pm(x['diff_pp'])} {ci(x['ci95_pp'])}")
        erows.append(f"CIFT $-$ BCT & {l} & " + " & ".join(cells) + " \\\\")
    for other, nm in (("cift_nokl", "w/o KL"), ("cift_nohelp", "w/o helpful pairs")):
        for k, l in (("heldout_cues", "held-out cues"), ("helpful", "helpful cue"), ("transfer_misleading", "transfer")):
            v = abl["results"][other][k]
            mac(("AblNoKL" if other == "cift_nokl" else "AblNoHelp") + k.replace("_", "") + "d", v[0])
            erows.append(f"CIFT $-$ {nm} & {l} & -- & {pm(v[0])} {ci(v[1:])} & -- \\\\")
    (TAB / "explore.tex").write_text("\n".join([
        "\\begin{table}[t]\\centering\\small",
        "\\caption{Exploratory comparisons (not pre-registered except H4): differences in percentage points, two-level "
        "bootstrap 95\\% CIs.}", "\\label{tab:explore}",
        "\\resizebox{\\columnwidth}{!}{\\begin{tabular}{llccc}\\toprule",
        "Comparison & Metric & Qwen & Llama & Med42 \\\\ \\midrule"] + erows + ["\\bottomrule\\end{tabular}}\\end{table}"]) + "\n")

    # ---- dev gap analysis
    grows = []
    order = ["Qwen__Qwen2.5-7B-Instruct_dev600_s0", "meta-llama__Llama-3.1-8B-Instruct_dev600_s0",
             "m42-health__Llama3-Med42-8B_dev600_s0", "gpt41mini_dev600_s0"]
    gshort = ["Qwen", "Llama", "Medfortytwo", "Gpt"]
    fams = ["social_proof", "self_anchoring", "patient_context", "institutional", "authority", "external_anchor",
            "automation", "convenience"]
    for f in fams:
        grows.append(f.replace("_", " ") + (" (new)" if f in fams[4:] else "") + " & " + " & ".join(
            f"{gap[o]['per_family'][f]['drop']['pp']:.1f}" for o in order) + " \\\\")
    for o, s in zip(order, gshort):
        g = gap[o]
        mac(f"Gap{s}Clean", g["clean_acc"]); mac(f"Gap{s}DropOld", g["drop_biasmedqa"]["pp"])
        mac(f"Gap{s}DropNew", g["drop_new"]["pp"]); mac(f"Gap{s}NewMinusOld", g["new_minus_biasmedqa_drop"]["pp"])
        mac(f"Gap{s}FlipAny", g["flip"]["share_of_clean_correct_flipping_on_>=1_cue"])
        mac(f"Gap{s}FlipHalf", g["flip"]["share_flipping_on_>=half_of_cues"])
        if "calibration" in g:
            mac(f"Gap{s}ConfWrong", g["calibration"]["mean_conf_when_following_wrong_cue"])
            fl = g["flip_rate_breakdown_pct"]["by_clean_confidence_tercile"]
            mac(f"Gap{s}FlipLowConf", fl["low"]); mac(f"Gap{s}FlipHighConf", fl["high"])
    grows.append("\\midrule mean, BiasMedQA cues & " + " & ".join(f"{gap[o]['drop_biasmedqa']['pp']:.1f}" for o in order) + " \\\\")
    grows.append("mean, new cues & " + " & ".join(f"{gap[o]['drop_new']['pp']:.1f}" for o in order) + " \\\\")
    (TAB / "gap.tex").write_text("\n".join([
        "\\begin{table}[t]\\centering\\small",
        "\\caption{Accuracy drop (clean $-$ misleading, percentage points) per hint family for untrained models on 600 dev "
        "questions. The four new families are used only as held-out cues.}", "\\label{tab:gap}",
        "\\resizebox{\\columnwidth}{!}{\\begin{tabular}{lcccc}\\toprule",
        "Family & Qwen2.5-7B & Llama-3.1-8B & Med42-8B & gpt-4.1-mini \\\\ \\midrule"] + grows +
        ["\\bottomrule\\end{tabular}}\\end{table}"]) + "\n")

    # ---- reproduction
    rrows = []
    for model, d in comp["paper"]["avg_drop_pp_per_model"].items():
        rrows.append(f"{model} (paper, test) & {d:.1f} \\\\")
    for run, short in (("gpt41mini_n600_s0", "gpt-4.1-mini"), ("Qwen__Qwen2.5-7B-Instruct_n600_s0", "Qwen2.5-7B"),
                       ("meta-llama__Llama-3.1-8B-Instruct_n600_s0", "Llama-3.1-8B"),
                       ("m42-health__Llama3-Med42-8B_n600_s0", "Med42-8B")):
        o = comp["ours"][run]
        rrows.append(f"{short} (ours, dev 600) & {o['avg_drop_pp']:.1f} \\\\")
        mac("Repro" + short.replace("-", "").replace(".", "").replace("gpt41mini", "Gpt").replace("Qwen257B", "Qwen")
            .replace("Llama318B", "Llama").replace("Med428B", "Medfortytwo") + "Drop", o["avg_drop_pp"])
    u = comp["paper"]["headline_units_check"]
    for b, short in (("false_consensus", "FC"), ("frequency", "Freq"), ("recency", "Rec"), ("confirmation", "Conf")):
        mac(f"PaperClaim{short}", u[b]["claimed"]); mac(f"PaperTable{short}", u[b]["mean_pp_drop_6_models"])
    (TAB / "repro.tex").write_text("\n".join([
        "\\begin{table}[t]\\centering\\small",
        "\\caption{Faithful BiasMedQA reproduction: mean accuracy drop over the 7 cues (percentage points, no mitigation, "
        "5-option format). Paper values from its Supplementary Table 1.}", "\\label{tab:repro}",
        "\\begin{tabular}{lc}\\toprule Model & Mean drop \\\\ \\midrule"] + rrows + ["\\bottomrule\\end{tabular}\\end{table}"]) + "\n")

    # ---- costs
    led = list(csv.DictReader((ROOT / "results/costs/gpu_ledger.csv").open()))
    api = sum(float(r["cost_usd"]) for r in csv.DictReader((ROOT / "results/costs/api_ledger.csv").open()))
    mac("GpuJobHours", sum(float(r["gpu_hours"]) for r in led)); mac("ApiUSD", f"{api:.2f}")
    mac("NTest", "1{,}273"); mac("NDev", "600"); mac("NPairs", "2{,}000"); mac("NTransfer", "1{,}050")

    # ---- assorted numbers cited in the text (each from a results/ or data/ file)
    import glob
    import statistics
    mac("PaperGptFourDrop", comp["paper"]["avg_drop_pp_per_model"]["gpt-4-0613"])
    for m in MODELS:  # BCT target error rate = 1 - base-model accuracy on the training questions
        tr = json.loads((ROOT / f"results/main/{m}/bct_s0_test/summary.json").read_text())["train"]
        mac(SHORT[m] + "BctTargetWrong", 100 - tr["bct_target_accuracy_vs_gold"])
    pe = J("results/main/prefix_eval_check.json")
    agr = [v["agreement_pct"] for k, v in pe.items() if k != "provenance"]
    mac("PrefixAgreeMin", min(agr)); mac("PrefixAgreeMax", max(agr))
    mac("PrefixAccDiffMax", max(v["max_abs_acc_diff_pp"] for k, v in pe.items() if k != "provenance"))
    mac("TransferCtxWords", f"{J('results/data/transfer_stats.json')['context_words_median']:.0f}")  # scripts/transfer_stats.py
    lm = []
    for p in glob.glob(str(ROOT / "results/main/*/*_test/summary.json")):
        lm.append(min(json.loads(Path(p).read_text())["mean_letter_mass"].values()))
    mac("LetterMassMin", f"{min(lm):.3f}")
    mac("GptNonrespMax", max(ref["nonresponse_rate"].values()))
    times = {}
    for a2 in ("cift", "sft_clean", "bct"):
        v = [json.loads(Path(p).read_text())["times"]["train_s"] / 60
             for p in glob.glob(str(ROOT / f"results/main/*/{a2}_s*_test/summary.json"))]
        times[a2] = statistics.mean(v)
    mac("TrainMinCIFT", f"{times['cift']:.0f}"); mac("TrainMinSFT", f"{times['sft_clean']:.0f}")
    mac("TrainMinBCT", f"{times['bct']:.0f}")

    # ---- item validity review (results/review/review_summary.json, scripts/analyze_review.py)
    rv = J("results/review/review_summary.json")
    mac("ReviewN", f"{rv['n_items']}"); mac("ReviewNHint", f"{rv['n_hint']}"); mac("ReviewNNota", f"{rv['n_nota']}")
    mac("ReviewValid", f"{rv['q1_valid']}"); mac("ReviewKeyHint", f"{rv['q2_key_best_hint']}")
    mac("ReviewNotaClear", f"{rv['q4_nota_clearly_correct']}")
    mac("ReviewPlausMean", f"{rv['q3_plausibility_mean']:.1f}"); mac("ReviewPlausOld", f"{rv['q3_mean_biasmedqa']:.1f}")
    mac("ReviewPlausNew", f"{rv['q3_mean_new']:.1f}"); mac("ReviewPlausGeThree", f"{rv['q3_share_ge3']:.0f}")
    se = rv["sensitivity_flagged_removed"]["Qwen__Qwen2.5-7B-Instruct"]
    mac("ReviewSensN", f"{se['n_questions_kept']}")

    out = ["% AUTO-GENERATED by scripts/make_tables.py from results/ -- do not edit by hand."]
    out += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(macros.items())]
    (ROOT / "paper/numbers.tex").write_text("\n".join(out) + "\n")
    print(f"{len(macros)} macros -> paper/numbers.tex; tables -> paper/tables/")


if __name__ == "__main__":
    main()
