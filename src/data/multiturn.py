"""MedQA-MultiTurnRobust (Manczak et al., arXiv 2510.12255; HF dynamoai-ml/MedQA-USMLE-4-MultiTurnRobust, MIT) for the
Phase 5 transfer test (docs/preregistration.md B5). We use its single-turn `misleading_context` paragraph (an evidence-style
manipulation pointing to `target_answer_misleading_context`), prepended to the question. This is a different style from
our one-sentence cues.
`prepare()` writes data/raw/multiturn/items.jsonl once (pinned revision); `load()` reads it.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data/raw/multiturn/items.jsonl"
HF_REPO, HF_REVISION = "dynamoai-ml/MedQA-USMLE-4-MultiTurnRobust", "8f345fc82b278e79e1aef40ca16e0831f5b446e2"


def prepare(token: str | None = None) -> dict:
    from datasets import load_dataset
    ds = load_dataset(HF_REPO, revision=HF_REVISION, token=token)["train"]
    RAW.parent.mkdir(parents=True, exist_ok=True)
    kept, dropped = 0, []
    with RAW.open("w") as f:
        for i, r in enumerate(ds):
            t = (r["target_answer_misleading_context"] or "").strip()
            ok = t in r["options"] and r["options"].get(t) and t != r["answer_idx"] and (r["misleading_context"] or "").strip()
            if not ok:
                dropped.append(i)
                continue
            f.write(json.dumps({"id": f"mtr-{i:04d}", "question": r["question"].strip(),
                                "options": {k: v.strip() for k, v in r["options"].items()}, "answer_idx": r["answer_idx"],
                                "meta_info": r["meta_info"], "misleading_context": r["misleading_context"].strip(),
                                "misleading_target": t}, ensure_ascii=False) + "\n")
            kept += 1
    return {"rows": len(ds), "kept": kept, "dropped_invalid_target": dropped}


def load() -> list[dict]:
    if not RAW.exists():
        raise FileNotFoundError(f"{RAW} missing: run python -c 'from src.data.multiturn import prepare; prepare()'")
    return [json.loads(l) for l in RAW.open()]


def with_context(item: dict) -> dict:
    """The misleading-context condition: the paragraph is placed before the question."""
    return {**item, "question": f"{item['misleading_context']}\n\n{item['question']}", "cue_target": item["misleading_target"]}
