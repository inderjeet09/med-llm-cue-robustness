"""Phase 5 main runner: one arm x one model x one seed -> train (if needed) -> evaluate (docs/preregistration.md Part B, frozen D020).

Arms
  untrained : base, warn_generic (generic_warning_v1), warn_oracle (paper's per-cue education text), fewshot (paper's
              one negative + one positive TRAIN demonstration, faithful layout, 4-option)
  trained   : cift (ours), cift_nokl (lam=0), cift_nohelp (no helpful pairs), cift_all15 (all 15 cues, optional),
              sft_clean, bct
Evaluation (MedQA 4-option, letter scoring, fixed inference settings D019):
  base / trained : clean + 15 misleading + 8 helpful (24 conditions); prefix-shared scoring (B7) when --prefix_eval 1
  prompt arms    : clean + 15 misleading
  transfer       : MedQA-MultiTurnRobust clean vs misleading_context (test split, base + trained arms)
Training uses MedQA TRAIN only (2,000 pairs, pair seed 1, training cues = 7 BiasMedQA templates). Test is used only
here, with frozen settings. Outputs: results/main/<model>/<arm>_s<seed>_<split>/{records,transfer_records}.jsonl,
summary.json, train_log.jsonl.
Usage (Kaggle): python scripts/run_main.py --model Qwen/Qwen2.5-7B-Instruct --revision <sha> --arm cift --seed 0
       (local debug): ... --model Qwen/Qwen2.5-0.5B-Instruct --n_pairs 16 --n_eval 6 --split dev --out experiments/debug/main
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
from src.data.bias import BIASMEDQA, NEW_BIASES, add_cue  # noqa: E402
from src.data.medqa import load_split, sample  # noqa: E402
from src.llm.prompts import biasmedqa_messages, faithful_messages, fewshot_pair  # noqa: E402
from src.utils.provenance import provenance  # noqa: E402

LETTERS = "ABCD"
HELPFUL_EVAL = ["frequency", "recency", "self_diagnosis", "status_quo", "authority_senior", "referral_note", "cds_tool",
                "convenience_time"]  # one per family (docs/preregistration.md B5)
MIS = BIASMEDQA + NEW_BIASES
PROMPT_ARMS = ["warn_generic", "warn_oracle", "fewshot"]
TRAINED = {"cift": ("cift", "all7", 3, 1.0), "cift_nokl": ("cift", "all7", 3, 0.0), "cift_nohelp": ("cift", "all7", 0, 1.0),
           "cift_all15": ("cift", "all15", 3, 1.0), "sft_clean": ("sft_clean", "sft_clean", 3, 1.0),
           "bct": ("bct", "all7", 3, 1.0)}  # arm -> (objective, pair arm, helpful_every, lam)


def conditions(arm: str) -> list[str]:
    if arm in PROMPT_ARMS:
        return ["clean"] + MIS
    return ["clean"] + MIS + [f"{b}+helpful" for b in HELPFUL_EVAL]


def build(item: dict, cond: str, arm: str, train4: list[dict]):
    bias, *mods = cond.split("+")
    x = item if cond == "clean" else add_cue(item, bias, helpful="helpful" in mods)
    tgt = None if cond == "clean" else x["cue_target"]
    if arm == "warn_generic":
        return biasmedqa_messages(x, warning="generic_warning_v1")[0], tgt
    if arm == "warn_oracle":
        return biasmedqa_messages(x, education=None if cond == "clean" else bias)[0], tgt
    if arm == "fewshot":
        if cond == "clean":
            return faithful_messages(x)[0], tgt
        neg, pos = fewshot_pair(train4, bias)
        return faithful_messages(x, demos=[(neg, neg["cue_target"], "incorrect"), (pos, pos["answer_idx"], "correct")])[0], tgt
    return biasmedqa_messages(x)[0], tgt


def score(m, items, conds, arm, train4, prefix_eval: bool, bs: int, mbt: int) -> dict[str, list[dict]]:
    """Returns {cond: [result per item]} with fields pred, probs, letter_mass, cue_target."""
    built = {c: [build(it, c, arm, train4) for it in items] for c in conds}
    out: dict[str, list] = {c: [None] * len(items) for c in conds}
    # prefix sharing only where every condition has the same instruction (oracle/few-shot prompts differ per cue)
    sharable = prefix_eval and (arm in ("base", "warn_generic") or arm in TRAINED)
    shared = [c for c in conds if not c.startswith("confirmation")] if sharable else []
    if shared:  # cues appended after the question share the instruction + question; confirmation is mid-question
        res = m.letter_probs_grouped([[built[c][i][0] for c in shared] for i in range(len(items))], LETTERS)
        for i, row in enumerate(res):
            for c, r in zip(shared, row):
                out[c][i] = r
    rest = [c for c in conds if c not in shared]
    if rest:
        flat = [built[c][i][0] for c in rest for i in range(len(items))]
        res = m.letter_probs(flat, LETTERS, batch_size=bs, max_batch_tokens=mbt)
        for k, c in enumerate(rest):
            out[c] = res[k * len(items):(k + 1) * len(items)]
    for c in conds:
        for i, r in enumerate(out[c]):
            r["pred"] = max(r["probs"], key=r["probs"].get)
            r["cue_target"] = built[c][i][1]
    return out


def write_records(path: Path, items, scored, split: str) -> None:
    with path.open("w") as f:
        for c, rows in scored.items():
            for it, r in zip(items, rows):
                f.write(json.dumps({"qid": it["id"], "cond": c, "gold": it["answer_idx"], "pred": r["pred"],
                                    "correct": r["pred"] == it["answer_idx"], "cue_target": r["cue_target"],
                                    "probs": r["probs"], "letter_mass": r["letter_mass"], "split": split}) + "\n")


def acc_summary(scored, items) -> dict:
    gold = [it["answer_idx"] for it in items]
    acc = {c: 100 * float(np.mean([r["pred"] == g for r, g in zip(rows, gold)])) for c, rows in scored.items()}
    out = {"accuracy": acc, "mean_letter_mass": {c: float(np.mean([r["letter_mass"] for r in rows]))
                                                 for c, rows in scored.items()}}
    for name, bs in (("train_cues", BIASMEDQA), ("heldout_cues", NEW_BIASES)):
        if all(b in acc for b in bs):
            out[f"misleading_acc_{name}"] = float(np.mean([acc[b] for b in bs]))
    helpful = [acc[f"{b}+helpful"] for b in HELPFUL_EVAL if f"{b}+helpful" in acc]
    if helpful:
        out["helpful_acc_mean"] = float(np.mean(helpful))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--arm", required=True, choices=["base"] + PROMPT_ARMS + list(TRAINED))
    ap.add_argument("--seed", type=int, default=0, help="training seed (LoRA init + data order)")
    ap.add_argument("--split", default="test", choices=["dev", "test"])
    ap.add_argument("--n_eval", type=int, default=None, help="default: full split (dev: 600, seed 0)")
    ap.add_argument("--n_pairs", type=int, default=2000)
    ap.add_argument("--pair_seed", type=int, default=1)
    ap.add_argument("--prefix_eval", type=int, default=1)
    ap.add_argument("--transfer", type=int, default=None, help="default: 1 on test for base/trained arms")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_batch_tokens", type=int, default=8192)
    ap.add_argument("--load_4bit", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from src.llm.hf_runner import HFModel
    from src.training.pairs import build_pairs
    from src.training.trainer import train

    run = f"{args.arm}_s{args.seed}_{args.split}" if args.arm in TRAINED else f"{args.arm}_{args.split}"
    out = Path(args.out or ROOT / "results/main") / args.model.replace("/", "__") / run
    out.mkdir(parents=True, exist_ok=True)
    # fail fast: every input this job needs must exist BEFORE loading the model or training (P5-1 lost 3.3 GPU-h to a
    # missing test file discovered after training, D022)
    items = load_split(args.split)
    if args.split == "dev" or args.n_eval:
        items = sample(items, args.n_eval or 600, 0)
    train4 = load_split("train")
    do_transfer = args.transfer if args.transfer is not None else int(args.split == "test" and args.arm not in PROMPT_ARMS)
    if do_transfer:
        from src.data.multiturn import load as load_mtr
        mtr_items = load_mtr()
    t0, times = time.time(), {}
    m = HFModel.load(args.model, load_4bit=None if args.load_4bit is None else bool(args.load_4bit),
                     revision=args.revision)
    times["load_s"] = round(time.time() - t0, 1)
    # adapters are not zipped or committed; a custom --out (debug) keeps its own adapter, never a real run's
    adapter_dir = (Path(args.out) if args.out else ROOT / "experiments/main") / args.model.replace("/", "__") / run / "adapter"

    train_info = None
    if args.arm in TRAINED:
        objective, pair_arm, helpful_every, lam = TRAINED[args.arm]
        pairs = build_pairs(train4, args.n_pairs, pair_arm, seed=args.pair_seed, helpful_every=helpful_every)
        own = None
        if objective == "bct":  # base model's own clean answers = BCT targets (computed before any training)
            res = m.letter_probs([biasmedqa_messages(p["clean"])[0] for p in pairs], LETTERS,
                                 batch_size=args.batch_size, max_batch_tokens=args.max_batch_tokens)
            own = [max(r["probs"], key=r["probs"].get) for r in res]
        tt = time.time()
        if (adapter_dir / "adapter_config.json").exists() and (out / "train_info.json").exists():
            from peft import PeftModel   # retry after a failure during evaluation: reuse the trained adapter
            m.model = PeftModel.from_pretrained(m.model, adapter_dir).eval()
            train_info = json.loads((out / "train_info.json").read_text())
            train_info["reused_adapter_on_retry"] = True
            print("adapter found: skipping training", flush=True)
        else:
            train_info = train(m, pairs, objective, args.seed, out / "train_log.jsonl", own_answers=own, lam=lam)
        train_info.update({"arm": args.arm, "pair_arm": pair_arm, "helpful_every": helpful_every, "n_pairs": len(pairs),
                           "n_helpful_pairs": sum(bool(p["cued"] and p["cued"]["cue_helpful"]) for p in pairs),
                           "first_pair_ids": [p["clean"]["id"] for p in pairs[:5]]})
        if own is not None:
            train_info["bct_target_accuracy_vs_gold"] = 100 * float(np.mean(
                [o == p["clean"]["answer_idx"] for o, p in zip(own, pairs)]))
        times["train_s"] = round(time.time() - tt, 1)
        if not train_info.get("reused_adapter_on_retry"):
            m.model.save_pretrained(adapter_dir)
            (out / "train_info.json").write_text(json.dumps(train_info, indent=1))

    te = time.time()
    scored = score(m, items, conditions(args.arm), args.arm, train4, bool(args.prefix_eval), args.batch_size,
                   args.max_batch_tokens)
    times["eval_s"] = round(time.time() - te, 1)
    write_records(out / "records.jsonl", items, scored, args.split)
    summary = {"model": args.model, "arm": args.arm, "seed": args.seed, "split": args.split, "n": len(items),
               **acc_summary(scored, items), "train": train_info}

    if do_transfer:
        from src.data.multiturn import with_context
        mtr = mtr_items
        if args.n_eval:
            mtr = mtr[:args.n_eval]
        tt = time.time()
        msgs = [biasmedqa_messages(it)[0] for it in mtr] + [biasmedqa_messages(with_context(it))[0] for it in mtr]
        res = m.letter_probs(msgs, LETTERS, batch_size=args.batch_size, max_batch_tokens=args.max_batch_tokens)
        tr = {"clean": res[:len(mtr)], "misleading_context": res[len(mtr):]}
        for c, rows in tr.items():
            for it, r in zip(mtr, rows):
                r["pred"] = max(r["probs"], key=r["probs"].get)
                r["cue_target"] = it["misleading_target"] if c != "clean" else None
        write_records(out / "transfer_records.jsonl", mtr, tr, "mtr")
        summary["transfer"] = {c: 100 * float(np.mean([r["pred"] == it["answer_idx"] for r, it in zip(rows, mtr)]))
                               for c, rows in tr.items()}
        summary["transfer"]["n"] = len(mtr)
        times["transfer_s"] = round(time.time() - tt, 1)

    summary["times"] = times
    summary["provenance"] = provenance(vars(args), __file__, seed=args.seed, device=m.device, extra={
        "model_revision": m.revision, "runtime_s": round(time.time() - t0, 1),
        "inference": {"batch_size": args.batch_size, "max_batch_tokens": args.max_batch_tokens,
                      "prefix_eval": args.prefix_eval}})
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: summary.get(k) for k in ("misleading_acc_train_cues", "misleading_acc_heldout_cues",
                                                   "helpful_acc_mean", "transfer", "times")}, indent=1))


if __name__ == "__main__":
    main()
