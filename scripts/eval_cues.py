"""Phase 4: score a model on clean and cued versions of MedQA dev questions (4-option, letter-only BiasMedQA prompt).

Conditions per cue template b (15 templates: 7 BiasMedQA + 8 new families, src/data/bias.py):
  mis   b                 misleading cue (wrong option)
  help  b+helpful         helpful cue (correct option)
  edu   b+edu             misleading cue + the education ("warning") addendum for b
plus `clean`. Open models: next-token letter probability (records keep the full A-D distribution, used for the
calibration analysis). API model: free text at temperature 0, parsed for a letter.
DEV only (the test split is refused). Resume-safe per condition. GPU settings are fixed across methods (D019).
Usage:
  python scripts/eval_cues.py --backend hf --model Qwen/Qwen2.5-7B-Instruct --revision <sha> --groups mis help edu
  python scripts/eval_cues.py --backend azure --groups mis edu --tag gpt41mini_dev600
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.bias import BIASMEDQA, FAMILY_OF, NEW_BIASES, TEMPLATES, add_cue  # noqa: E402
from src.data.medqa import load_split, sample  # noqa: E402
from src.evaluation.parsing import parse_answer  # noqa: E402
from src.llm.prompts import biasmedqa_messages  # noqa: E402
from src.utils.provenance import provenance  # noqa: E402

LETTERS = "ABCD"
SUFFIX = {"mis": "", "help": "+helpful", "edu": "+edu", "gen": "+gen"}  # gen = generic warning (Phase 5)


def conditions(groups: list[str], biases: list[str]) -> list[str]:
    extra = ["clean+gen"] if "gen" in groups else []
    return ["clean"] + extra + [f"{b}{SUFFIX[g]}" for g in groups for b in biases]


def build(item: dict, cond: str):
    warning = "generic_warning_v1" if cond.endswith("+gen") else None
    if cond in ("clean", "clean+gen"):
        return (*biasmedqa_messages(item, warning=warning), None)
    bias, *mods = cond.split("+")
    x = add_cue(item, bias, helpful="helpful" in mods)
    return (*biasmedqa_messages(x, education=bias if "edu" in mods else None, warning=warning), x["cue_target"])


def summarise(rec_path: Path) -> dict:
    by: dict[str, list[dict]] = {}
    for r in map(json.loads, rec_path.open()):
        by.setdefault(r["cond"], []).append(r)
    acc = {c: 100 * float(np.mean([r["correct"] for r in v])) for c, v in by.items()}
    out = {"n": len(by["clean"]), "accuracy": acc,
           "nonresponse_rate": {c: 100 * float(np.mean([r["pred"] is None for r in v])) for c, v in by.items()},
           "cue_follow_rate": {c: 100 * float(np.mean([r["pred"] == r["cue_target"] for r in v]))
                               for c, v in by.items() if c != "clean"}}
    if "letter_mass" in by["clean"][0]:
        out["mean_letter_mass"] = {c: float(np.mean([r["letter_mass"] for r in v])) for c, v in by.items()}
    fam: dict[str, list[float]] = {}
    for b in TEMPLATES:
        if b in acc:
            fam.setdefault(FAMILY_OF[b], []).append(acc["clean"] - acc[b])
    for name, bs in (("train_cues", BIASMEDQA), ("heldout_cues", NEW_BIASES)):
        for suf in ("", "+edu", "+gen"):
            if all(f"{b}{suf}" in acc for b in bs):
                out[f"misleading_acc_{name}{suf.replace('+', '_')}"] = float(np.mean([acc[f"{b}{suf}"] for b in bs]))
    out["drop_pp_by_family"] = {f: float(np.mean(v)) for f, v in fam.items()}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["hf", "azure"], required=True)
    ap.add_argument("--model", default=None)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--split", default="dev", choices=["dev", "test"],
                    help="test only for the frozen Phase 5 reference run (docs/preregistration.md B3, D020)")
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--groups", nargs="+", default=["mis"], choices=list(SUFFIX))
    ap.add_argument("--biases", default="all", choices=["all", "biasmedqa", "new"])
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_batch_tokens", type=int, default=8192)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--run_cap", type=float, default=None)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--out_root", default="results/gap_analysis", help="results/reference for the test reference run")
    args = ap.parse_args()

    biases = {"all": list(TEMPLATES), "biasmedqa": BIASMEDQA, "new": NEW_BIASES}[args.biases]
    name = "gpt-4.1-mini" if args.backend == "azure" else args.model.replace("/", "__")
    out = ROOT / args.out_root / (args.tag or f"{name}_dev{args.n}_s{args.seed}")
    out.mkdir(parents=True, exist_ok=True)
    rec_path = out / "records.jsonl"
    items = load_split(args.split)
    if args.split == "test":
        assert args.n in (None, 0, len(items)), "the test reference run uses the FULL test split (pass --n 0)"
    elif args.n and args.n < len(items):
        items = sample(items, args.n, args.seed)
    conds = conditions(args.groups, biases)
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
        client = AzureClient(experiment=f"gap_{out.name}", phase="4", run_cap_usd=args.run_cap)

        def answer(msgs):
            with ThreadPoolExecutor(args.workers) as ex:
                outs = list(ex.map(lambda m: client.chat(m, max_tokens=5), msgs))
            return [{"raw": o["text"], "pred": parse_answer(o["text"], LETTERS)} for o in outs]
    else:
        from src.llm.hf_runner import HFModel
        m = HFModel.load(args.model, revision=args.revision)
        meta.update({"model_revision": m.revision, "device": m.device, "batch_size": args.batch_size,
                     "max_batch_tokens": args.max_batch_tokens})
        print(f"loaded {args.model} rev {m.revision} in {time.time() - t0:.0f}s", flush=True)

        def answer(msgs):
            res = m.letter_probs(msgs, LETTERS, batch_size=args.batch_size, max_batch_tokens=args.max_batch_tokens)
            return [{"pred": max(r["probs"], key=r["probs"].get), "probs": r["probs"], "letter_mass": r["letter_mass"]}
                    for r in res]

    timing, templates = {}, {}
    for cond in conds:
        if cond in done:
            continue
        tc = time.time()
        built = [build(it, cond) for it in items]
        templates[cond] = built[0][1]
        res = answer([b[0] for b in built])
        with rec_path.open("a") as f:
            for it, b, r in zip(items, built, res):
                f.write(json.dumps({"qid": it["id"], "cond": cond, "gold": it["answer_idx"], "cue_target": b[2],
                                    "step": it.get("meta_info"), "q_words": len(it["question"].split()), **r,
                                    "correct": r["pred"] == it["answer_idx"]}) + "\n")
        timing[cond] = round(time.time() - tc, 1)
        print(f"{cond:30s} {timing[cond]:6.1f}s", flush=True)

    summary = summarise(rec_path)
    if args.backend == "azure":
        meta["api_cost_usd_this_run"] = round(client.run_spent, 4)
    summary["meta"] = {**meta, "seconds_per_condition": timing}
    summary["provenance"] = provenance(vars(args), __file__, seed=args.seed, extra={
        "templates": templates, "split": args.split, "options": 4, "runtime_s": round(time.time() - t0, 1)})
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: summary[k] for k in ("drop_pp_by_family",)}, indent=1))
    if args.backend == "azure":
        print(f"API cost this run: ${client.run_spent:.4f}")


if __name__ == "__main__":
    main()
