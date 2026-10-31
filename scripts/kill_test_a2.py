"""Phase 1.5 KILL TEST A, stage 2: does a mini QLoRA run with counterfactual-invariance training recover the
bias-cue drop, and does it beat the education prompt? Pre-registered in docs/preregistration.md (stage 2).

Modes
  --mode train      train one arm (all7 | heldout4 | sft_clean) on 2,000 TRAIN pairs, then evaluate on the same 300 DEV
                    questions as stage 1: clean, misleading x3, helpful x3 (letter-probability argmax).
  --mode base_extra untrained model: education prompt on CLEAN questions and on HELPFUL cues (the prompt's side costs).
The GO decision is computed afterwards by scripts/analyze_kill_a2.py (needs the stage-1 records).

Retry-safe: the adapter is saved after training; a re-run with an existing adapter skips training and only evaluates.
Usage:
  python scripts/kill_test_a2.py --model Qwen/Qwen2.5-0.5B-Instruct --arm all7 --n_pairs 16 --n_eval 10 --out experiments/debug/a2   # local debug
  python scripts/kill_test_a2.py --model Qwen/Qwen2.5-7B-Instruct --revision <sha> --arm all7                                     # Kaggle
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.bias import add_cue  # noqa: E402
from src.data.medqa import load_split, sample  # noqa: E402
from src.llm.prompts import biasmedqa_messages  # noqa: E402
from src.training.consistency import letter_logprobs, pair_loss  # noqa: E402
from src.training.pairs import ARMS, build_pairs  # noqa: E402
from src.utils.provenance import provenance  # noqa: E402

BIASES = ["false_consensus", "frequency", "recency"]
LETTERS = "ABCD"
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


# --- evaluation ---------------------------------------------------------------------------------------------------
def eval_conditions(mode: str) -> list[str]:
    if mode == "base_extra":
        return [f"clean+edu:{b}" for b in BIASES] + [f"{b}+helpful+edu" for b in BIASES]
    return ["clean"] + BIASES + [f"{b}+helpful" for b in BIASES]


def build_eval(item: dict, cond: str):
    """Returns (messages, template_ids, cue_target)."""
    if cond == "clean":
        return (*biasmedqa_messages(item), None)
    if cond.startswith("clean+edu:"):
        return (*biasmedqa_messages(item, education=cond.split(":")[1]), None)
    bias, *mods = cond.split("+")
    x = add_cue(item, bias, helpful="helpful" in mods)
    return (*biasmedqa_messages(x, education=bias if "edu" in mods else None), x["cue_target"])


def evaluate(m, items: list[dict], conds: list[str], rec_path: Path, batch_size: int) -> dict[str, list[str]]:
    """Append one record per (question, condition); conditions already complete in rec_path are skipped."""
    done = set()
    if rec_path.exists():
        cnt: dict[str, int] = {}
        for r in map(json.loads, rec_path.open()):
            cnt[r["cond"]] = cnt.get(r["cond"], 0) + 1
        done = {c for c, k in cnt.items() if k == len(items)}
        keep = [l for l in rec_path.open() if json.loads(l)["cond"] in done]
        rec_path.write_text("".join(keep))
    templates = {}
    for cond in conds:
        if cond in done:
            continue
        tc = time.time()
        built = [build_eval(it, cond) for it in items]
        templates[cond] = built[0][1]
        res = m.letter_probs([b[0] for b in built], LETTERS, batch_size=batch_size)
        with rec_path.open("a") as f:
            for it, b, r in zip(items, built, res):
                pred = max(r["probs"], key=r["probs"].get)
                f.write(json.dumps({"qid": it["id"], "cond": cond, "gold": it["answer_idx"], "pred": pred,
                                    "correct": pred == it["answer_idx"], "cue_target": b[2],
                                    "probs": r["probs"], "letter_mass": r["letter_mass"]}) + "\n")
        print(f"eval {cond:28s} {time.time() - tc:6.1f}s", flush=True)
    return templates


def summarise(rec_path: Path) -> dict:
    by: dict[str, list[dict]] = {}
    for r in map(json.loads, rec_path.open()):
        by.setdefault(r["cond"], []).append(r)
    out = {"accuracy": {c: 100 * float(np.mean([r["correct"] for r in v])) for c, v in by.items()},
           "mean_letter_mass": {c: float(np.mean([r["letter_mass"] for r in v])) for c, v in by.items()},
           "cue_follow_rate": {c: 100 * float(np.mean([r["pred"] == r["cue_target"] for r in v]))
                               for c, v in by.items() if v[0]["cue_target"] is not None}}
    out["n"] = len(next(iter(by.values())))
    return out


# --- training -----------------------------------------------------------------------------------------------------
def train(m, pairs: list[dict], args, log_path: Path) -> dict:
    from peft import LoraConfig, get_peft_model
    from transformers import get_cosine_schedule_with_warmup

    cuda = m.device == "cuda"
    model = m.model
    model.config.use_cache = False
    if args.grad_ckpt:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.05,
                                             target_modules=LORA_TARGETS, task_type="CAUSAL_LM"))
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable params: {n_train / 1e6:.1f} M", flush=True)
    m.model = model
    letter_ids = m._letter_ids(LETTERS)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    steps = math.ceil(len(pairs) * args.epochs / (args.micro_pairs * args.accum))
    sched = get_cosine_schedule_with_warmup(opt, max(1, int(0.05 * steps)), steps)
    scaler = torch.amp.GradScaler("cuda", enabled=cuda)
    rng = random.Random(args.train_seed)
    torch.manual_seed(args.train_seed)
    model.train()
    t0, step, micro, agg, n_nonfinite = time.time(), 0, 0, [], 0
    order = [i for _ in range(math.ceil(args.epochs)) for i in rng.sample(range(len(pairs)), len(pairs))]
    order = order[:int(len(pairs) * args.epochs)]
    with log_path.open("w") as logf:
        for s in range(0, len(order), args.micro_pairs):
            batch = [pairs[i] for i in order[s:s + args.micro_pairs]]
            gold = torch.tensor([LETTERS.index(p["clean"]["answer_idx"]) for p in batch], device=m.device)
            msgs = [biasmedqa_messages(p["clean"])[0] for p in batch]
            has_cued = batch[0]["cued"] is not None
            if has_cued:
                msgs += [biasmedqa_messages(p["cued"])[0] for p in batch]
            enc = m._encode(msgs)  # left padding -> position -1 is the answer position for every row
            with torch.autocast("cuda", dtype=torch.float16, enabled=cuda):
                logits = model(**enc, logits_to_keep=1).logits[:, -1, :]
            lp = letter_logprobs(logits, letter_ids)
            k = len(batch)
            out = pair_loss(lp[:k], lp[k:] if has_cued else None, gold, lam=args.lam)
            if not torch.isfinite(out["loss"]):  # fp16 overflow guard: skip this micro-batch, count it
                n_nonfinite += 1
                print(f"non-finite loss at micro-batch {micro} (skipped)", flush=True)
                if n_nonfinite > 20:
                    raise RuntimeError("too many non-finite losses: fp16 instability")
                continue
            scaler.scale(out["loss"] / args.accum).backward()
            agg.append({key: float(v.detach()) for key, v in out.items() if v is not None})
            micro += 1
            if micro % args.accum == 0 or s + args.micro_pairs >= len(order):
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()
                step += 1
                if step % args.log_every == 0 or step == steps:
                    row = {"step": step, "of": steps, "lr": sched.get_last_lr()[0],
                           "elapsed_min": round((time.time() - t0) / 60, 2),
                           **{key: float(np.mean([a[key] for a in agg if key in a])) for key in agg[0]}}
                    logf.write(json.dumps(row) + "\n")
                    logf.flush()
                    print(json.dumps(row), flush=True)
                    agg = []
    model.eval()
    return {"steps": step, "train_minutes": round((time.time() - t0) / 60, 2), "trainable_params": n_train,
            "nonfinite_microbatches_skipped": n_nonfinite}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", default=None, help="pin to the stage-1 hub commit")
    ap.add_argument("--mode", choices=["train", "base_extra"], default="train")
    ap.add_argument("--arm", choices=list(ARMS), default="all7")
    ap.add_argument("--n_pairs", type=int, default=2000)
    ap.add_argument("--pair_seed", type=int, default=1)
    ap.add_argument("--train_seed", type=int, default=0)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lam", type=float, default=1.0, help="KL consistency weight")
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--micro_pairs", type=int, default=4)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--grad_ckpt", type=int, default=1)
    ap.add_argument("--log_every", type=int, default=5)
    ap.add_argument("--n_eval", type=int, default=300)
    ap.add_argument("--eval_seed", type=int, default=0, help="must match stage 1 (same 300 dev questions)")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--load_4bit", type=int, default=None, help="default: 1 on CUDA, 0 otherwise")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from src.llm.hf_runner import HFModel

    run = "base_extra" if args.mode == "base_extra" else f"{args.arm}_s{args.train_seed}"
    out = Path(args.out or ROOT / "results/pilot/killA2" / args.model.replace("/", "__")) / run
    out.mkdir(parents=True, exist_ok=True)
    # adapters stay out of results/ (weights are not zipped or committed); a custom --out keeps its own adapter
    adapter_dir = (Path(args.out) / run if args.out else ROOT / "experiments/killA2" / args.model.replace("/", "__") / run) / "adapter"
    t0 = time.time()
    m = HFModel.load(args.model, load_4bit=None if args.load_4bit is None else bool(args.load_4bit),
                     revision=args.revision)
    print(f"loaded {args.model} on {m.device} rev {m.revision} in {time.time() - t0:.0f}s", flush=True)

    train_info = None
    if args.mode == "train":
        if (adapter_dir / "adapter_config.json").exists():
            from peft import PeftModel
            m.model = PeftModel.from_pretrained(m.model, adapter_dir).eval()
            train_info = json.loads((out / "train_info.json").read_text())
            print("adapter found: skipping training", flush=True)
        else:
            pairs = build_pairs(load_split("train"), args.n_pairs, args.arm, seed=args.pair_seed)
            n_helpful = sum(bool(p["cued"] and p["cued"]["cue_helpful"]) for p in pairs)
            train_info = {**train(m, pairs, args, out / "train_log.jsonl"), "n_pairs": len(pairs),
                          "n_helpful_pairs": n_helpful, "templates": ARMS[args.arm],
                          "first_pair_ids": [p["clean"]["id"] for p in pairs[:5]]}
            m.model.save_pretrained(adapter_dir)
            (out / "train_info.json").write_text(json.dumps(train_info, indent=1))

    items = sample(load_split("dev"), args.n_eval, args.eval_seed)
    templates = evaluate(m, items, eval_conditions(args.mode), out / "records.jsonl", args.batch_size)
    summary = summarise(out / "records.jsonl")
    summary["train"] = train_info
    summary["provenance"] = provenance(vars(args), __file__, seed=args.train_seed, device=m.device, extra={
        "model": args.model, "model_revision": m.revision, "templates": templates, "split_eval": "dev",
        "split_train": "train", "runtime_s": round(time.time() - t0, 1)})
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({"accuracy": summary["accuracy"]}, indent=1))


if __name__ == "__main__":
    main()
