"""QLoRA training loop shared by all Phase 5 trained arms (docs/preregistration.md B1, B4).

Objectives (selected by `objective`):
  cift      CE(clean -> gold) + CE(cued -> gold) + lam * KL(stopgrad p_clean || p_cued)      (ours; lam=0 = no-KL ablation)
  sft_clean CE(clean -> gold)                                                              (plain fine-tuning)
  bct       CE(cued -> base model's own clean answer)                                      (Chua et al. 2024)
All arms use identical optimisation settings. The training seed fixes the LoRA initialisation and the data order.
Derived from scripts/kill_test_a2.py (kill test, frozen); difference: the seed is set BEFORE the adapter is created.
"""
from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch

from src.llm.prompts import biasmedqa_messages
from src.training.consistency import bct_loss, letter_logprobs, pair_loss

LETTERS = "ABCD"
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
DEFAULTS = {"lr": 2e-4, "epochs": 1.0, "lora_r": 16, "micro_pairs": 4, "accum": 4, "lam": 1.0, "grad_ckpt": 1,
            "log_every": 5}


def train(m, pairs: list[dict], objective: str, seed: int, log_path: Path, own_answers: list[str] | None = None,
          **hp) -> dict:
    """m: HFModel (model replaced in place by the PEFT model). pairs: from src.training.pairs.build_pairs.
    own_answers: BCT targets (base-model clean argmax letter per pair)."""
    from peft import LoraConfig, get_peft_model
    from transformers import get_cosine_schedule_with_warmup

    hp = {**DEFAULTS, **hp}
    cuda = m.device == "cuda"
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    model = m.model
    model.config.use_cache = False
    if hp["grad_ckpt"]:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=hp["lora_r"], lora_alpha=2 * hp["lora_r"], lora_dropout=0.05,
                                             target_modules=LORA_TARGETS, task_type="CAUSAL_LM"))
    m.model = model
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    letter_ids = m._letter_ids(LETTERS)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=hp["lr"], weight_decay=0.0)
    n_items = int(len(pairs) * hp["epochs"])
    steps = math.ceil(n_items / (hp["micro_pairs"] * hp["accum"]))
    sched = get_cosine_schedule_with_warmup(opt, max(1, int(0.05 * steps)), steps)
    scaler = torch.amp.GradScaler("cuda", enabled=cuda)
    rng = random.Random(seed)
    order = [i for _ in range(math.ceil(hp["epochs"])) for i in rng.sample(range(len(pairs)), len(pairs))][:n_items]
    model.train()
    t0, step, micro, agg, nonfinite = time.time(), 0, 0, [], 0
    with log_path.open("w") as logf:
        for s in range(0, len(order), hp["micro_pairs"]):
            idx = order[s:s + hp["micro_pairs"]]
            batch = [pairs[i] for i in idx]
            k = len(batch)
            if objective == "bct":
                msgs = [biasmedqa_messages(p["cued"])[0] for p in batch]
            else:
                msgs = [biasmedqa_messages(p["clean"])[0] for p in batch]
                if objective == "cift":
                    msgs += [biasmedqa_messages(p["cued"])[0] for p in batch]
            enc = m._encode(msgs)  # left padding: position -1 is the answer position in every row
            with torch.autocast("cuda", dtype=torch.float16, enabled=cuda):
                logits = model(**enc, logits_to_keep=1).logits[:, -1, :]
            lp = letter_logprobs(logits, letter_ids)
            if objective == "bct":
                tgt = torch.tensor([LETTERS.index(own_answers[i]) for i in idx], device=m.device)
                out = bct_loss(lp, tgt)
            else:
                gold = torch.tensor([LETTERS.index(p["clean"]["answer_idx"]) for p in batch], device=m.device)
                out = pair_loss(lp[:k], lp[k:] if objective == "cift" else None, gold, lam=hp["lam"])
            if not torch.isfinite(out["loss"]):  # fp16 overflow guard: skip and count
                nonfinite += 1
                if nonfinite > 20:
                    raise RuntimeError("too many non-finite losses: fp16 instability")
                continue
            scaler.scale(out["loss"] / hp["accum"]).backward()
            agg.append({key: float(v.detach()) for key, v in out.items() if v is not None})
            micro += 1
            if micro % hp["accum"] == 0 or s + hp["micro_pairs"] >= len(order):
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); sched.step()
                step += 1
                if step % hp["log_every"] == 0 or step == steps:
                    row = {"step": step, "of": steps, "lr": sched.get_last_lr()[0],
                           "elapsed_min": round((time.time() - t0) / 60, 2),
                           **{key: float(np.mean([a[key] for a in agg if key in a])) for key in agg[0]}}
                    logf.write(json.dumps(row) + "\n"); logf.flush()
                    print(json.dumps(row), flush=True)
                    agg = []
    model.eval()
    model.config.use_cache = True
    return {"objective": objective, "seed": seed, "steps": step, "train_minutes": round((time.time() - t0) / 60, 2),
            "trainable_params": n_train, "nonfinite_microbatches_skipped": nonfinite, "hyperparameters": hp}
