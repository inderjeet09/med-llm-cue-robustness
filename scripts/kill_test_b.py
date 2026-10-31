"""Phase 1.5 KILL TEST B: does option-blind generate-then-verify (GTV) fix the no-correct-option failure?
Pre-registered in docs/preregistration.md.

n MedQA DEV questions (4-option), excluding any question that appears in Handler et al.'s MedQA-NOTA set (kept clean
as an external test). Variants per question:
  original   unchanged
  nota       correct option's text -> "None of the other answers" (Bedi et al. 2025); gold = same letter
             (a reply of NONE is also counted correct: it states the same thing)
  removed    correct option deleted, no NOTA option (Handler et al. 2026); gold = NONE (abstain)
Conditions:
  cot              zero-shot chain-of-thought (Bedi protocol)                         [the "drop" baseline]
  handler_base     Handler JSON prompt that allows abstention
  handler_safety   Handler safety prompt ("abstain unless highly confident")          [best prompt baseline]
  gtv              ours: step 1 answer without options, step 2 verify options, NONE allowed
GO if NOTA drop (cot: original - nota) >= 20 points AND gtv recovers >= 50% of it with <= 2 points loss on original.

Backends: --backend azure (via src/llm/azure_client.py, cached + ledgered) or --backend hf (local/Kaggle).
Resume-safe: per-call caching (azure) / per-condition records (hf).
Usage:
  python scripts/kill_test_b.py --backend azure --n 10 --tag pilot      # cost pilot
  python scripts/kill_test_b.py --backend azure --n 300
  python scripts/kill_test_b.py --backend hf --model Qwen/Qwen2.5-7B-Instruct --n 300
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.medqa import load_split, sample  # noqa: E402
from src.data.nota import NONE, nota_replace, original, remove_answer  # noqa: E402
from src.evaluation.parsing import parse_answer  # noqa: E402
from src.evaluation.stats import paired_bootstrap_diff  # noqa: E402
from src.llm.prompts import (cot_messages, extract_own_answer, gtv_step1_messages, gtv_step2_messages,  # noqa: E402
                             handler_messages, parse_handler_json)
from src.utils.provenance import provenance  # noqa: E402

VARIANTS = {"original": original, "nota": nota_replace, "removed": remove_answer}
CONDITIONS = ["cot", "handler_base", "handler_safety", "gtv"]
MAX_TOK = {"cot": 1024, "handler": 400, "gtv1": 512, "gtv2": 1024}
HANDLER_CSV = ROOT / "data" / "raw" / "handler_nota" / "questions.csv"


def _norm(s: str) -> str:
    # 80-char normalised prefix: Handler stems differ slightly in formatting from bigbio's (see docs/preregistration.md)
    return " ".join(s.lower().split())[:80]


def eligible_dev() -> list[dict]:
    dev = load_split("dev")
    if not HANDLER_CSV.exists():
        raise FileNotFoundError(f"{HANDLER_CSV} missing — run scripts/fetch_handler_nota.sh")
    with HANDLER_CSV.open(newline="") as f:
        hq = {_norm(r["stem"]) for r in csv.DictReader(f)}
    return [d for d in dev if _norm(d["question"]) not in hq]


class Backend:
    def __init__(self, args):
        self.kind = args.backend
        if self.kind == "azure":
            from src.llm.azure_client import AzureClient
            self.client = AzureClient(experiment=f"killB_{args.tag}", phase="1.5", run_cap_usd=args.run_cap)
            self.name, self.revision, self.device = "azure:" + self.client.deployment, "api", "api"
        else:
            from src.llm.hf_runner import HFModel
            self.m = HFModel.load(args.model)
            self.name, self.revision, self.device = args.model, self.m.revision, self.m.device
        self.workers, self.bs = args.workers, args.batch_size
        self.meta: list[dict] = []  # azure: per-call model version / fingerprint / cost

    def generate(self, msgs: list[list[dict]], max_tokens: int) -> list[str]:
        if self.kind == "azure":
            def one(m):
                return self.client.chat(m, max_tokens=max_tokens)
            with ThreadPoolExecutor(self.workers) as ex:
                outs = list(ex.map(one, msgs))
            self.meta += [{"model": o.get("model"), "fp": o.get("system_fingerprint"), "cost": o["cost_usd"],
                           "cached": o["cached"]} for o in outs]
            return [o["text"] or "" for o in outs]
        return self.m.generate(msgs, max_new_tokens=max_tokens, batch_size=self.bs)


def run(args) -> None:
    out = Path(args.out or ROOT / "results/pilot/killB" / args.tag)
    out.mkdir(parents=True, exist_ok=True)
    rec_path = out / "records.jsonl"
    done = {}
    if rec_path.exists():
        for l in rec_path.open():
            r = json.loads(l)
            done[(r["qid"], r["variant"], r["cond"])] = r
    items = sample(eligible_dev(), args.n, args.seed)
    be = Backend(args)
    t0 = time.time()
    var_items = {v: [f(it) for it in items] for v, f in VARIANTS.items()}

    def save(rows):
        with rec_path.open("a") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                done[(r["qid"], r["variant"], r["cond"])] = r

    def score(pred, gold, variant):
        return pred == gold or (variant == "nota" and pred == NONE)

    for cond in [c for c in CONDITIONS if c in args.conditions]:
        tc = time.time()
        if cond == "gtv":
            # step 1 is identical across variants (same stem, options hidden): one call per question
            todo = [it for it in items if any((f"{it['id']}|{v}", v, "gtv") not in done for v in VARIANTS)]
            s1 = be.generate([gtv_step1_messages(it)[0] for it in todo], MAX_TOK["gtv1"]) if todo else []
            own = {it["id"]: (txt, extract_own_answer(txt)) for it, txt in zip(todo, s1)}
            for v in VARIANTS:
                pend = [(it, x) for it, x in zip(items, var_items[v]) if (x["id"], v, "gtv") not in done]
                if not pend:
                    continue
                s2 = be.generate([gtv_step2_messages(x, own[it["id"]][1])[0] for it, x in pend], MAX_TOK["gtv2"])
                save([{"qid": x["id"], "base_qid": it["id"], "variant": v, "cond": "gtv", "gold": x["answer_idx"],
                       "pred": (p := parse_answer(t, "".join(sorted(x["options"])))),
                       "correct": score(p, x["answer_idx"], v), "own_answer": own[it["id"]][1],
                       "step1": own[it["id"]][0], "response": t} for (it, x), t in zip(pend, s2)])
        else:
            for v in VARIANTS:
                pend = [(it, x) for it, x in zip(items, var_items[v]) if (x["id"], v, cond) not in done]
                if not pend:
                    continue
                if cond == "cot":
                    msgs = [cot_messages(x)[0] for _, x in pend]
                    texts = be.generate(msgs, MAX_TOK["cot"])
                    preds = [parse_answer(t, "".join(sorted(x["options"]))) for (_, x), t in zip(pend, texts)]
                else:
                    msgs = [handler_messages(x, safety=(cond == "handler_safety"))[0] for _, x in pend]
                    texts = be.generate(msgs, MAX_TOK["handler"])
                    preds = [parse_handler_json(t, "".join(sorted(x["options"]))) for (_, x), t in zip(pend, texts)]
                save([{"qid": x["id"], "base_qid": it["id"], "variant": v, "cond": cond, "gold": x["answer_idx"],
                       "pred": p, "correct": score(p, x["answer_idx"], v), "response": t}
                      for (it, x), t, p in zip(pend, texts, preds)])
        print(f"{cond:16s} done in {time.time() - tc:6.1f}s", flush=True)

    summary = summarise(list(done.values()), [it["id"] for it in items])
    templates = {"cot": cot_messages(items[0])[1], "handler_base": ["handler_baseline_v1"],
                 "handler_safety": ["handler_safety_v1"], "gtv": ["gtv_step1_v1", "gtv_step2_v1"]}
    extra = {"model": be.name, "model_revision": be.revision, "templates": templates, "max_tokens": MAX_TOK,
             "decoding": "greedy / temperature 0, seed 0", "split": "dev (Handler MedQA-NOTA items excluded)",
             "runtime_s": round(time.time() - t0, 1)}
    if be.kind == "azure":
        extra["api_cost_usd_this_run"] = round(sum(m["cost"] for m in be.meta), 6)
        extra["api_calls"] = len(be.meta)
        extra["api_cache_hits"] = sum(m["cached"] for m in be.meta)
        extra["api_model_versions"] = sorted({m["model"] for m in be.meta if m["model"]})
        extra["api_price_basis"] = be.client.price_basis
    summary["provenance"] = provenance(vars(args), __file__, seed=args.seed, device=be.device, extra=extra)
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: summary[k] for k in ("accuracy", "unparsed_pct", "kill_test_B") if k in summary}, indent=1))
    if be.kind == "azure":
        print(f"API cost this run: ${extra['api_cost_usd_this_run']:.4f} ({extra['api_calls']} calls, "
              f"{extra['api_cache_hits']} cache hits); mean completion tokens per call: see ledger")


def summarise(recs: list[dict], base_ids: list[str]) -> dict:
    by: dict = {}
    for r in recs:
        by.setdefault((r["cond"], r["variant"]), {})[r["base_qid"]] = r
    acc, unparsed, vec = {}, {}, {}
    for (c, v), d in by.items():
        if len(d) != len(base_ids):
            continue
        x = np.array([d[q]["correct"] for q in base_ids], float)
        vec[(c, v)] = x
        acc.setdefault(c, {})[v] = float(x.mean() * 100)
        unparsed.setdefault(c, {})[v] = float(np.mean([d[q]["pred"] is None for q in base_ids]) * 100)
        if v != "removed":
            acc[c][f"{v}_said_NONE"] = float(np.mean([d[q]["pred"] == NONE for q in base_ids]) * 100)
    out: dict = {"n": len(base_ids), "accuracy": acc, "unparsed_pct": unparsed,
                 "note": "removed-variant accuracy = abstention rate (gold is NONE)"}

    def diff(a, b):
        d = paired_bootstrap_diff(vec[a], vec[b])
        return {"diff_pp": d["diff"] * 100, "ci95_pp": [c * 100 for c in d["ci95"]]}

    need = [("cot", "original"), ("cot", "nota"), ("gtv", "original"), ("gtv", "nota")]
    if all(k in vec for k in need):
        drop = acc["cot"]["original"] - acc["cot"]["nota"]
        rec = (acc["gtv"]["nota"] - acc["cot"]["nota"]) / drop if drop > 0 else None
        clean_cost = acc["cot"]["original"] - acc["gtv"]["original"]
        out["kill_test_B"] = {
            "nota_drop_pp": drop, "nota_drop_ci": diff(("cot", "original"), ("cot", "nota")),
            "gtv_recovery_fraction": rec, "gtv_nota_gain": diff(("gtv", "nota"), ("cot", "nota")),
            "gtv_clean_cost_pp": clean_cost, "gtv_clean_cost_ci": diff(("cot", "original"), ("gtv", "original")),
            "criterion": "nota_drop_pp >= 20 and recovery >= 0.5 and clean_cost_pp <= 2",
            "GO": bool(drop >= 20 and rec is not None and rec >= 0.5 and clean_cost <= 2)}
        if ("handler_safety", "nota") in vec:
            out["kill_test_B"]["gtv_vs_handler_safety"] = {
                v: diff(("gtv", v), ("handler_safety", v)) for v in VARIANTS if ("handler_safety", v) in vec}
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["azure", "hf"], required=True)
    ap.add_argument("--model", default=None, help="HF model id (hf backend)")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--conditions", default=",".join(CONDITIONS))
    ap.add_argument("--tag", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--run_cap", type=float, default=None, help="USD per-run cap (azure); default configs/budget.yaml")
    a = ap.parse_args()
    a.conditions = a.conditions.split(",")
    a.tag = a.tag or (("azure" if a.backend == "azure" else a.model.replace("/", "__")) + f"_n{a.n}_s{a.seed}")
    run(a)
