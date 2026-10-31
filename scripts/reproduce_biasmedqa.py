"""Phase 3: faithful reproduction of the BiasMedQA protocol (Schmidgall et al. 2024, npj Digit Med 7:295).

Protocol (paper Methods): 5-option MedQA, the paper's instruction ("... for research purposes only ... Respond with only
a single letter"), options on one line, 7 bias cues appended to the question, and 4 strategies: none, education
addendum, one-shot (negative TRAIN example), few-shot (negative + positive TRAIN example). Prompt files:
prompts/biasmedqa_faithful_*.txt, prompts/biasmedqa_demo_*.txt.
Deviations (docs/DECISIONS.md D017): DEV split instead of test (the test set stays untouched until Phase 5); a fixed random
subset of n questions; open models answer by next-token letter probability (no free text, so never a non-response);
the API model answers in free text at temperature 0, parsed for a single letter.

Accuracy is reported two ways: strict (non-response = wrong) and the paper's convention (non-responses excluded).
Usage:
  python scripts/reproduce_biasmedqa.py --backend azure --n 10 --tag pilot                       # API cost pilot
  python scripts/reproduce_biasmedqa.py --backend azure --n 600 --tag gpt41mini_n600
  python scripts/reproduce_biasmedqa.py --backend hf --model Qwen/Qwen2.5-7B-Instruct --n 600 --skip oneshot   # Kaggle
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.bias import BIASMEDQA, add_cue  # noqa: E402
from src.data.medqa import RAW_DIR5, load_split, sample  # noqa: E402
from src.evaluation.parsing import parse_answer  # noqa: E402
from src.evaluation.stats import mcnemar_exact, paired_bootstrap_diff  # noqa: E402
from src.llm.prompts import faithful_messages, fewshot_pair  # noqa: E402
from src.utils.provenance import provenance  # noqa: E402

LETTERS = "ABCDE"
STRATEGIES = ["none", "edu", "oneshot", "fewshot"]


def conditions(skip: list[str]) -> list[str]:
    conds = ["nobias"] + [b if s == "none" else f"{b}+{s}" for s in STRATEGIES if s not in skip for b in BIASMEDQA]
    return conds


def build(item: dict, cond: str, train: list[dict]):
    if cond == "nobias":
        return faithful_messages(item)
    bias, _, strat = cond.partition("+")
    x = add_cue(item, bias)
    if strat == "edu":
        return faithful_messages(x, education=bias)
    if strat in ("oneshot", "fewshot"):
        neg, pos = fewshot_pair(train, bias)
        demos = [(neg, neg["cue_target"], "incorrect")]
        if strat == "fewshot":
            demos.append((pos, pos["answer_idx"], "correct"))
        return faithful_messages(x, demos=demos)
    return faithful_messages(x)


def two_prop_z(p1: float, p2: float, n1: int, n2: int) -> float:
    """The paper's two-sample proportion z-test (their Eq. 1-2); returns the two-sided p-value."""
    from scipy.stats import norm
    p = (n1 * p1 + n2 * p2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2)) if 0 < p < 1 else 0.0
    return float(2 * norm.sf(abs((p1 - p2) / se))) if se > 0 else 1.0


def summarise(recs: list[dict]) -> dict:
    by: dict[str, dict[str, dict]] = {}
    for r in recs:
        by.setdefault(r["cond"], {})[r["qid"]] = r
    qids = sorted(by["nobias"])
    full = {c: v for c, v in by.items() if len(v) == len(qids)}
    corr = {c: np.array([float(v[q]["correct"]) for q in qids]) for c, v in full.items()}
    nonr = {c: np.array([float(v[q]["pred"] is None) for q in qids]) for c, v in full.items()}
    acc = {c: 100 * float(corr[c].mean()) for c in full}
    acc_excl = {c: (100 * float(corr[c][nonr[c] == 0].mean()) if (nonr[c] == 0).any() else None) for c in full}
    out: dict = {"n": len(qids), "accuracy_strict": acc, "accuracy_excl_nonresponse": acc_excl,
                 "nonresponse_rate": {c: 100 * float(nonr[c].mean()) for c in full}}
    table = {}
    for s in STRATEGIES:
        cols = {b: (b if s == "none" else f"{b}+{s}") for b in BIASMEDQA}
        if all(c in full for c in cols.values()):
            row = {"nobias": acc_excl["nobias"], **{b: acc_excl[c] for b, c in cols.items()}}
            row["avg"] = float(np.mean([row[b] for b in BIASMEDQA]))
            table[s] = row
    out["paper_style_table_excl_nonresponse"] = table
    drops = {}
    for b in BIASMEDQA:
        if b in corr:
            d = paired_bootstrap_diff(corr["nobias"], corr[b])
            drops[b] = {"drop_pp": 100 * d["diff"], "ci95_pp": [100 * x for x in d["ci95"]],
                        "mcnemar_p": mcnemar_exact(corr["nobias"], corr[b])["p"],
                        "paper_ztest_p": two_prop_z(corr["nobias"].mean(), corr[b].mean(), len(qids), len(qids))}
    out["drops_vs_nobias_strict"] = drops
    if "none" in table:
        out["avg_drop_pp_excl"] = table["none"]["nobias"] - table["none"]["avg"]
        out["mitigation_gain_pp_excl"] = {s: table[s]["avg"] - table["none"]["avg"] for s in table if s != "none"}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["azure", "hf"], required=True)
    ap.add_argument("--model", default=None, help="HF model id (hf backend)")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--split", default="dev", choices=["dev"], help="test is reserved for Phase 5 (D017)")
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip", nargs="*", default=[], choices=STRATEGIES[1:])
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_batch_tokens", type=int, default=8192, help="hf: padded-token budget per batch (D018)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max_tokens", type=int, default=5)
    ap.add_argument("--run_cap", type=float, default=None)
    ap.add_argument("--min_letter_mass", type=float, default=0.5,
                    help="hf: stop after the no-bias condition if the mean letter mass is below this (invalid scoring)")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    name = "gpt-4.1-mini" if args.backend == "azure" else args.model.replace("/", "__")
    out = ROOT / "results/reproduction" / (args.tag or f"{name}_n{args.n}_s{args.seed}")
    out.mkdir(parents=True, exist_ok=True)
    rec_path = out / "records.jsonl"
    items = sample(load_split(args.split, RAW_DIR5), args.n, args.seed)
    train = load_split("train", RAW_DIR5)
    conds = conditions(args.skip)

    done = set()
    if rec_path.exists():
        cnt: dict[str, int] = {}
        for r in map(json.loads, rec_path.open()):
            cnt[r["cond"]] = cnt.get(r["cond"], 0) + 1
        done = {c for c, k in cnt.items() if k == len(items)}
        rec_path.write_text("".join(l for l in rec_path.open() if json.loads(l)["cond"] in done))

    t0 = time.time()
    meta: dict = {"backend": args.backend}
    if args.backend == "azure":
        from src.llm.azure_client import AzureClient
        client = AzureClient(experiment=f"repro_{out.name}", phase="3", run_cap_usd=args.run_cap)
        versions: set = set()

        def answer(msgs):
            def one(m):
                return client.chat(m, max_tokens=args.max_tokens)
            with ThreadPoolExecutor(args.workers) as ex:
                outs = list(ex.map(one, msgs))
            versions.update((o.get("model"), o.get("system_fingerprint")) for o in outs if o.get("model"))
            return [{"raw": o["text"], "pred": parse_answer(o["text"], LETTERS), "finish": o["finish_reason"]}
                    for o in outs]
    else:
        from src.llm.hf_runner import HFModel
        m = HFModel.load(args.model, revision=args.revision)
        meta.update({"model_revision": m.revision, "device": m.device})
        print(f"loaded {args.model} rev {m.revision} on {m.device} in {time.time() - t0:.0f}s", flush=True)

        def answer(msgs):
            res = m.letter_probs(msgs, LETTERS, batch_size=args.batch_size, max_batch_tokens=args.max_batch_tokens)
            return [{"pred": max(r["probs"], key=r["probs"].get), "probs": r["probs"], "letter_mass": r["letter_mass"]}
                    for r in res]

    templates = {}
    for cond in conds:
        if cond in done:
            continue
        tc = time.time()
        built = [build(it, cond, train) for it in items]
        templates[cond] = built[0][1]
        res = answer([b[0] for b in built])
        with rec_path.open("a") as f:
            for it, r in zip(items, res):
                f.write(json.dumps({"qid": it["id"], "cond": cond, "gold": it["answer_idx"], **r,
                                    "correct": r["pred"] == it["answer_idx"]}) + "\n")
        print(f"{cond:28s} {time.time() - tc:6.1f}s", flush=True)
        if args.backend == "hf" and cond == "nobias":
            mass = float(np.mean([r["letter_mass"] for r in res]))
            meta["nobias_letter_mass"] = mass
            if mass < args.min_letter_mass:
                print(f"letter mass {mass:.3f} < {args.min_letter_mass}: letter scoring invalid for this model; stop")
                meta["invalid_scoring"] = True
                break

    recs = [json.loads(l) for l in rec_path.open()]
    summary = summarise(recs)
    if args.backend == "azure":
        meta.update({"api_cost_usd_this_run": round(client.run_spent, 4), "model_versions": sorted(map(list, versions))})
    summary["meta"] = meta
    summary["provenance"] = provenance(vars(args), __file__, seed=args.seed, extra={
        "templates": templates, "split": args.split, "options": 5, "runtime_s": round(time.time() - t0, 1)})
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: summary.get(k) for k in ("paper_style_table_excl_nonresponse", "avg_drop_pp_excl",
                                                   "mitigation_gain_pp_excl")}, indent=1))
    if args.backend == "azure":
        print(f"API cost this run: ${client.run_spent:.4f}")


if __name__ == "__main__":
    main()
