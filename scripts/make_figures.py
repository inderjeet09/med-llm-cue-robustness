"""Phase 5 final tables and figures from results/main + results/reference (MedQA test). No new computation of
hypothesis tests (those are in scripts/analyze_main.py and results/tables/*.json); this script only aggregates.

Tables  results/tables/T1_main.{md,csv}           per model x arm: clean, misleading (training / held-out cues), helpful,
                                                    transfer (mean ± std over seeds) + gpt-4.1-mini reference
        results/tables/T2_family_heldout.{md,csv}  held-out families: accuracy per model x arm
        results/tables/T3_cost.{md,csv}             GPU minutes per run (training / evaluation) per arm + API costs
Figures figures/F1_heldout_by_method.{png,pdf}     held-out misleading accuracy by method and model
        figures/F2_tradeoff_helpful.{png,pdf}      robustness (held-out misleading) vs use of correct hints
        figures/F3_family_drop.{png,pdf}           drop per held-out family (clean − misleading), small multiples
        figures/F4_transfer.{png,pdf}               evidence-paragraph transfer test
Palette validated with the dataviz validator (light surface): categorical slots 1-5, base in neutral grey (context).
Usage: python scripts/make_figures.py
"""
from __future__ import annotations

import csv
import glob
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.bias import BIASMEDQA, FAMILIES, NEW_BIASES  # noqa: E402

MODELS = {"Qwen__Qwen2.5-7B-Instruct": "Qwen2.5-7B", "meta-llama__Llama-3.1-8B-Instruct": "Llama-3.1-8B",
          "m42-health__Llama3-Med42-8B": "Med42-8B"}
BEST_PROMPT = {"Qwen__Qwen2.5-7B-Instruct": "fewshot", "meta-llama__Llama-3.1-8B-Instruct": "warn_generic",
               "m42-health__Llama3-Med42-8B": "warn_oracle"}  # frozen dev rule (D021, D025)
ARMS = ["base", "warn_generic", "warn_oracle", "fewshot", "sft_clean", "bct", "cift", "cift_nokl", "cift_nohelp"]
LABEL = {"base": "No mitigation", "warn_generic": "Generic warning", "warn_oracle": "Oracle warning",
         "fewshot": "Few-shot", "best_prompt": "Best prompt", "sft_clean": "Plain SFT", "bct": "BCT",
         "cift": "CIFT (ours)", "cift_nokl": "CIFT w/o KL", "cift_nohelp": "CIFT w/o helpful pairs"}
COLOR = {"cift": "#2a78d6", "best_prompt": "#eb6834", "sft_clean": "#1baf7a", "bct": "#eda100",
         "cift_nohelp": "#e87ba4", "base": "#9a9890", "cift_nokl": "#2a78d6"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
HELD_FAM = ["authority", "external_anchor", "automation", "convenience"]


def runs(model: str, arm: str) -> list[Path]:
    d = ROOT / "results/main" / model
    return sorted(Path(p).parent for p in glob.glob(str(d / f"{arm}_s*_test/summary.json"))) or (
        [d / f"{arm}_test"] if (d / f"{arm}_test/summary.json").exists() else [])


def cond_acc(run: Path) -> dict[str, float]:
    by: dict[str, list[float]] = {}
    for r in map(json.loads, (run / "records.jsonl").open()):
        by.setdefault(r["cond"], []).append(float(r["correct"]))
    return {c: 100 * float(np.mean(v)) for c, v in by.items()}


def metrics(run: Path) -> dict[str, float]:
    a = cond_acc(run)
    s = json.loads((run / "summary.json").read_text())
    m = {"clean": a["clean"], "train_cues": float(np.mean([a[b] for b in BIASMEDQA])),
         "heldout_cues": float(np.mean([a[b] for b in NEW_BIASES]))}
    hel = [v for c, v in a.items() if c.endswith("+helpful")]
    if hel:
        m["helpful"] = float(np.mean(hel))
    if "transfer" in s:
        m["transfer_clean"], m["transfer_mislead"] = s["transfer"]["clean"], s["transfer"]["misleading_context"]
    for f in FAMILIES:
        m[f"fam_{f}"] = float(np.mean([a[b] for b in FAMILIES[f]]))
    m["train_min"] = s.get("times", {}).get("train_s", 0) / 60
    m["eval_min"] = (s.get("times", {}).get("eval_s", 0) + s.get("times", {}).get("transfer_s", 0)) / 60
    return m


def agg(model: str, arm: str) -> dict | None:
    rs = runs(model, arm)
    if not rs:
        return None
    ms = [metrics(r) for r in rs]
    keys = set.intersection(*(set(m) for m in ms))
    return {"n": len(ms), **{k: (float(np.mean([m[k] for m in ms])),
                                 float(np.std([m[k] for m in ms], ddof=1)) if len(ms) > 1 else 0.0) for k in keys}}


def fmt(v, n):
    return f"{v[0]:.1f} ± {v[1]:.1f}" if n > 1 else f"{v[0]:.1f}"


def main() -> None:
    T = {m: {a: agg(m, a) for a in ARMS} for m in MODELS}
    tab = ROOT / "results/tables"; tab.mkdir(parents=True, exist_ok=True)
    figdir = ROOT / "figures"; figdir.mkdir(exist_ok=True)

    # ---- T1 main table
    cols = ["clean", "train_cues", "heldout_cues", "helpful", "transfer_mislead"]
    head = ["model", "arm", "seeds", "clean", "misleading: training cues", "misleading: held-out cues", "helpful",
            "transfer: misleading context"]
    rows = []
    for m, name in MODELS.items():
        for a in ARMS:
            d = T[m][a]
            if d:
                tag = " (best prompt)" if a == BEST_PROMPT[m] else ""
                rows.append([name, LABEL[a] + tag, d["n"]] + [fmt(d[c], d["n"]) if c in d else "–" for c in cols])
    ref = json.loads((ROOT / "results/reference/gpt41mini_test/summary.json").read_text())
    ra = ref["accuracy"]
    rows.append(["gpt-4.1-mini (reference, untrained)", "No mitigation", 1, f"{ra['clean']:.1f}",
                 f"{ref['misleading_acc_train_cues']:.1f}", f"{ref['misleading_acc_heldout_cues']:.1f}", "–", "–"])
    rows.append(["gpt-4.1-mini (reference, untrained)", "Oracle warning", 1, "–", f"{ref['misleading_acc_train_cues_edu']:.1f}",
                 f"{ref['misleading_acc_heldout_cues_edu']:.1f}", "–", "–"])
    rows.append(["gpt-4.1-mini (reference, untrained)", "Generic warning", 1, f"{ra['clean+gen']:.1f}",
                 f"{ref['misleading_acc_train_cues_gen']:.1f}", f"{ref['misleading_acc_heldout_cues_gen']:.1f}", "–", "–"])
    write(tab / "T1_main", head, rows, "Table 1. MedQA test (1,273 questions), accuracy %, mean ± std over 3 training "
          "seeds. Held-out cues = 8 new templates never used in training. Transfer = MedQA-MultiTurnRobust evidence paragraph.")

    # ---- T2 per held-out family
    head2 = ["model", "arm"] + [f.replace("_", " ") for f in HELD_FAM]
    rows2 = []
    for m, name in MODELS.items():
        for a in ["base", BEST_PROMPT[m], "sft_clean", "bct", "cift"]:
            d = T[m][a]
            if d:
                rows2.append([name, LABEL[a]] + [fmt(d[f"fam_{f}"], d["n"]) for f in HELD_FAM])
    write(tab / "T2_family_heldout", head2, rows2, "Table 2. Accuracy % under misleading cues, per held-out family "
          "(test). Clean accuracy in Table 1.")

    # ---- T3 cost
    head3 = ["arm", "runs", "training min / run", "evaluation min / run (test + transfer)"]
    rows3 = []
    for a in ARMS:
        tr = [metrics(r) for m in MODELS for r in runs(m, a)]
        if tr:
            rows3.append([LABEL[a], len(tr), f"{np.mean([t['train_min'] for t in tr]):.0f}",
                          f"{np.mean([t['eval_min'] for t in tr]):.0f}"])
    led = list(csv.DictReader((ROOT / "results/costs/gpu_ledger.csv").open()))
    api = sum(float(r["cost_usd"]) for r in csv.DictReader((ROOT / "results/costs/api_ledger.csv").open()))
    write(tab / "T3_cost", head3, rows3, f"Table 3. Wall-clock minutes per run on one Kaggle T4 (4-bit). Project totals: "
          f"{sum(float(r['gpu_hours']) for r in led):.1f} GPU-job-hours (Kaggle quota ≈ half, two GPUs per session); "
          f"Azure API ${api:.2f}.")

    # ---- figures
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                         "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb"})

    def save(fig, name):
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(figdir / f"{name}.{ext}", dpi=200, bbox_inches="tight", pad_inches=0.05)
        plt.close(fig)

    # F1: held-out misleading accuracy by method
    order = ["base", "best_prompt", "sft_clean", "bct", "cift"]
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    w, gap = 0.15, 0.02
    for i, m in enumerate(MODELS):
        for j, a in enumerate(order):
            d = T[m][BEST_PROMPT[m] if a == "best_prompt" else a]
            x = i + (j - 2) * (w + gap)
            ax.bar(x, d["heldout_cues"][0], w, color=COLOR[a], yerr=d["heldout_cues"][1] if d["n"] > 1 else None,
                   error_kw={"elinewidth": 1, "capsize": 2, "ecolor": INK2}, label=LABEL[a] if i == 0 else None,
                   edgecolor="#fcfcfb", linewidth=1)
            if a == "cift":
                ax.text(x, d["heldout_cues"][0] + d["heldout_cues"][1] + 4.5, f"{d['heldout_cues'][0]:.1f}",
                        ha="center", fontsize=8, color=INK)
        clean = T[m]["base"]["clean"][0]
        ax.plot([i - 0.42, i + 0.42], [clean, clean], color=INK, lw=1, ls="--",
                label="Clean accuracy, no mitigation" if i == 0 else None)
    ax.axhline(ref["misleading_acc_heldout_cues"], color=INK2, lw=1, ls=":")
    ax.text(2.55, ref["misleading_acc_heldout_cues"] + 1, "gpt-4.1-mini (untrained)", fontsize=7, color=INK2, ha="right")
    ax.set_xticks(range(len(MODELS))); ax.set_xticklabels(MODELS.values())
    ax.set_ylabel("Accuracy under held-out misleading cues (%)"); ax.set_ylim(0, 90)
    ax.yaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
    ax.legend(ncol=3, fontsize=7.5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.2))
    save(fig, "F1_heldout_by_method")

    # F2: robustness vs use of correct hints
    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    shapes = dict(zip(MODELS, ["o", "s", "^"]))
    for m, name in MODELS.items():
        for a in ["base", "sft_clean", "bct", "cift", "cift_nohelp"]:
            d = T[m][a]
            if d and "helpful" in d:
                ax.scatter(d["heldout_cues"][0], d["helpful"][0], s=60, marker=shapes[m], color=COLOR[a],
                           edgecolor="#fcfcfb", linewidth=2, zorder=3)
        if T[m]["cift"]:
            ax.annotate(name, (T[m]["cift"]["heldout_cues"][0], T[m]["cift"]["helpful"][0]), xytext=(6, -10),
                        textcoords="offset points", fontsize=7, color=INK2)
    nh = T["meta-llama__Llama-3.1-8B-Instruct"]["cift_nohelp"]
    ax.annotate("w/o helpful pairs:\ncorrect hints now hurt\n(contrarian shortcut)", (nh["heldout_cues"][0], nh["helpful"][0]),
                xytext=(-120, 10), textcoords="offset points", fontsize=7, color=INK,
                arrowprops={"arrowstyle": "-", "color": INK2, "lw": 0.8})
    for a in ["base", "sft_clean", "bct", "cift", "cift_nohelp"]:
        ax.scatter([], [], color=COLOR[a], s=40, label=LABEL[a])
    for m, name in MODELS.items():
        ax.scatter([], [], color="none", edgecolor=INK2, marker=shapes[m], s=40, label=name)
    ax.set_xlabel("Accuracy under held-out misleading cues (%)  →  more robust")
    ax.set_ylabel("Accuracy with a correct hint (%)  →  uses correct hints")
    ax.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
    ax.legend(fontsize=7, frameon=False, loc="lower left", ncol=2)
    save(fig, "F2_tradeoff_helpful")

    # F3: per held-out family drop, small multiples
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.8), sharey=True)
    arms3 = ["base", "sft_clean", "bct", "cift"]
    for ax, (m, name) in zip(axes, MODELS.items()):
        for j, a in enumerate(arms3):
            d = T[m][a]; clean = d["clean"][0]
            ys = [clean - d[f"fam_{f}"][0] for f in HELD_FAM]
            ax.bar(np.arange(4) + (j - 1.5) * 0.2, ys, 0.18, color=COLOR[a], label=LABEL[a], edgecolor="#fcfcfb", lw=1)
        ax.set_title(name, fontsize=9, color=INK); ax.axhline(0, color=INK2, lw=0.8)
        ax.set_xticks(range(4)); ax.set_xticklabels([f.replace("external_", "ext. ").replace("_", " ") for f in HELD_FAM],
                                                    rotation=25, ha="right", fontsize=7.5)
        ax.yaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
    axes[0].set_ylabel("Drop vs own clean accuracy (pp)")
    fig.legend(*axes[0].get_legend_handles_labels(), ncol=4, fontsize=7.5, frameon=False, loc="upper center",
               bbox_to_anchor=(0.5, 1.06))
    save(fig, "F3_family_drop")

    # F4: transfer to evidence paragraphs
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    for i, m in enumerate(MODELS):
        for j, a in enumerate(arms3):
            d = T[m][a]; x = i + (j - 1.5) * 0.19
            ax.bar(x, d["transfer_mislead"][0], 0.17, color=COLOR[a], label=LABEL[a] if i == 0 else None,
                   yerr=d["transfer_mislead"][1] if d["n"] > 1 else None, edgecolor="#fcfcfb", lw=1,
                   error_kw={"elinewidth": 1, "capsize": 2, "ecolor": INK2})
            ax.text(x, d["transfer_mislead"][0] + d["transfer_mislead"][1] + 1.0, f"{d['transfer_mislead'][0]:.0f}",
                    ha="center", fontsize=7, color=INK)
    ax.set_xticks(range(len(MODELS))); ax.set_xticklabels(MODELS.values())
    ax.set_ylabel("Accuracy with a misleading\nevidence paragraph (%)"); ax.set_ylim(0, 40)
    ax.yaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
    ax.legend(ncol=4, fontsize=7.5, frameon=False, loc="upper left")
    save(fig, "F4_transfer")
    print("tables ->", tab.relative_to(ROOT), "| figures ->", figdir.relative_to(ROOT))


def write(path: Path, head: list, rows: list, caption: str) -> None:
    with path.with_suffix(".csv").open("w", newline="") as f:
        w = csv.writer(f); w.writerow(head); w.writerows(rows)
    md = [caption, "", "| " + " | ".join(head) + " |", "|" + "---|" * len(head)] + \
         ["| " + " | ".join(map(str, r)) + " |" for r in rows]
    path.with_suffix(".md").write_text("\n".join(md) + "\n")


if __name__ == "__main__":
    main()
