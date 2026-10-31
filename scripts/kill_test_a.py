"""Phase 1.5 KILL TEST A (stage 1): how much do misleading bias cues hurt an open model, and how much do the base
paper's prompt mitigations recover? Pre-registered in docs/preregistration.md.

Protocol: n MedQA DEV questions (4-option), BiasMedQA letter-only prompt, answer = argmax of next-token letter
probabilities (deterministic). Conditions:
  clean
  <bias>            misleading cue (wrong option), no mitigation
  <bias>+edu        + Schmidgall education addendum
  <bias>+fewshot    + Schmidgall negative/positive demonstrations from TRAIN
  <bias>+helpful    cue pointing to the CORRECT option (control: does the model follow cues at all?)
for bias in {false_consensus, frequency, recency} (the three strongest in Schmidgall et al.).

GO (stage 1) if mean drop over the 3 biases >= 10 points AND best prompt mitigation recovers < 50% of it.

Resume-safe: records are appended per condition; completed conditions are skipped on restart.
Usage:
  python scripts/kill_test_a.py --model Qwen/Qwen2.5-0.5B-Instruct --n 20          # local debug
  python scripts/kill_test_a.py --model Qwen/Qwen2.5-7B-Instruct --n 300           # Kaggle (4-bit)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.bias import add_cue  # noqa: E402
from src.data.medqa import load_split, sample  # noqa: E402
from src.evaluation.stats import mcnemar_exact, paired_bootstrap_diff  # noqa: E402
from src.llm.prompts import biasmedqa_messages, fewshot_pair  # noqa: E402
from src.utils.provenance import provenance  # noqa: E402

BIASES = ["false_consensus", "frequency", "recency"]


def conditions() -> list[str]:
    return ["clean"] + [f"{b}{s}" for b in BIASES for s in ("", "+edu", "+fewshot", "+helpful")]


def build(item: dict, cond: str, train: list[dict]):
    if cond == "clean":
        return biasmedqa_messages(item)
    bias, _, mit = cond.partition("+")
    x = add_cue(item, bias, helpful=(mit == "helpful"))
    if mit == "edu":
        return biasmedqa_messages(x, education=bias)
    if mit == "fewshot":
        return biasmedqa_messages(x, fewshot=fewshot_pair(train, bias), fewshot_bias=bias)
    return biasmedqa_messages(x)


def summarise(recs: list[dict], n: int) -> dict:
    by: dict[str, dict[str, dict]] = {}
    for r in recs:
        by.setdefault(r["cond"], {})[r["qid"]] = r
    qids = sorted(by["clean"])
    corr = {c: np.array([by[c][q]["correct"] for q in qids], float) for c in by if len(by[c]) == len(qids)}
    acc = {c: float(v.mean() * 100) for c, v in corr.items()}
    out: dict = {"n": len(qids), "accuracy": acc,
                 "mean_letter_mass": {c: float(np.mean([by[c][q]["letter_mass"] for q in qids])) for c in corr}}
    drops, best_rec = {}, {}
    for b in BIASES:
        if b not in corr:
            continue
        d = paired_bootstrap_diff(corr["clean"], corr[b])
        drops[b] = {**{k: (v * 100 if k == "diff" else [x * 100 for x in v] if k == "ci95" else v)
                       for k, v in d.items()}, "mcnemar": mcnemar_exact(corr["clean"], corr[b])}
        mits = {m: acc[f"{b}+{m}"] for m in ("edu", "fewshot") if f"{b}+{m}" in acc}
        if mits:
            m = max(mits, key=mits.get)
            dr = acc["clean"] - acc[b]
            best_rec[b] = {"best_mitigation": m, "acc": mits[m],
                           "recovery_fraction": (mits[m] - acc[b]) / dr if dr > 0 else None}
        # how often the model picks the cued (wrong) option
        out.setdefault("cue_follow_rate", {})[b] = float(np.mean([by[b][q]["pred"] == by[b][q]["cue_target"]
                                                                  for q in qids]) * 100)
    out["drops_pp"], out["best_mitigation"] = drops, best_rec
    if len(drops) == len(BIASES) and len(best_rec) == len(BIASES):
        mean_drop = float(np.mean([acc["clean"] - acc[b] for b in BIASES]))
        mean_best = float(np.mean([best_rec[b]["acc"] for b in BIASES]))
        mean_biased = float(np.mean([acc[b] for b in BIASES]))
        rec = (mean_best - mean_biased) / mean_drop if mean_drop > 0 else None
        out["kill_test_A_stage1"] = {
            "mean_drop_pp": mean_drop, "pooled_best_mitigation_recovery": rec,
            "criterion": "mean_drop_pp >= 10 and recovery < 0.5",
            "GO": bool(mean_drop >= 10 and rec is not None and rec < 0.5)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0, help="question-sampling seed")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--load_4bit", type=int, default=None, help="default: 1 on CUDA, 0 otherwise")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from src.llm.hf_runner import HFModel

    out = Path(args.out or ROOT / "results/pilot/killA" / args.model.replace("/", "__") / f"n{args.n}_s{args.seed}")
    out.mkdir(parents=True, exist_ok=True)
    rec_path = out / "records.jsonl"
    done = set()
    if rec_path.exists():
        recs = [json.loads(l) for l in rec_path.open()]
        cnt: dict[str, int] = {}
        for r in recs:
            cnt[r["cond"]] = cnt.get(r["cond"], 0) + 1
        done = {c for c, k in cnt.items() if k == args.n}
        # drop partial conditions so they are redone cleanly
        recs = [r for r in recs if r["cond"] in done]
        rec_path.write_text("".join(json.dumps(r) + "\n" for r in recs))

    items = sample(load_split("dev"), args.n, args.seed)
    train = load_split("train")
    t0 = time.time()
    m = HFModel.load(args.model, load_4bit=None if args.load_4bit is None else bool(args.load_4bit),
                     revision=args.revision)
    print(f"loaded {args.model} on {m.device} rev {m.revision} in {time.time() - t0:.0f}s", flush=True)
    templates_used: dict[str, list[str]] = {}
    for cond in conditions():
        if cond in done:
            print(f"skip {cond} (done)", flush=True)
            continue
        tc = time.time()
        built = [build(it, cond, train) for it in items]
        templates_used[cond] = built[0][1]
        res = m.letter_probs([b[0] for b in built], "ABCD", batch_size=args.batch_size)
        with rec_path.open("a") as f:
            for it, r in zip(items, res):
                pred = max(r["probs"], key=r["probs"].get)
                tgt = add_cue(it, cond.partition("+")[0], helpful=cond.endswith("+helpful"))["cue_target"] \
                    if cond != "clean" else None
                f.write(json.dumps({"qid": it["id"], "cond": cond, "gold": it["answer_idx"], "pred": pred,
                                    "correct": pred == it["answer_idx"], "cue_target": tgt,
                                    "probs": r["probs"], "letter_mass": r["letter_mass"]}) + "\n")
        print(f"{cond:28s} done in {time.time() - tc:6.1f}s", flush=True)

    recs = [json.loads(l) for l in rec_path.open()]
    summary = summarise(recs, args.n)
    summary["provenance"] = provenance(vars(args), __file__, seed=args.seed, device=m.device, extra={
        "model": args.model, "model_revision": m.revision, "templates": templates_used,
        "runtime_s": round(time.time() - t0, 1), "split": "dev", "dataset_revision": "see data/raw/medqa_us4/PREPARED.json"})
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: summary[k] for k in ("accuracy", "kill_test_A_stage1") if k in summary}, indent=1))


if __name__ == "__main__":
    main()
