"""No-correct-option transforms (axis B).

- `nota_replace`: Bedi et al. 2025 (JAMA Netw Open, doi:10.1001/jamanetworkopen.2025.26021) — the correct option's
  text is replaced by "None of the other answers"; its letter stays the gold answer.
- `remove_answer`: Handler et al. 2026 (arXiv 2605.15000) style — the correct option is deleted and no NOTA option is
  offered; remaining options are re-lettered A.. in original order; gold answer is "NONE" (abstain / none correct).
"""
from __future__ import annotations

NOTA_TEXT = "None of the other answers"
NONE = "NONE"


def nota_replace(item: dict, text: str = NOTA_TEXT) -> dict:
    opts = dict(item["options"])
    opts[item["answer_idx"]] = text
    return {**item, "options": opts, "variant": "nota", "id": f"{item['id']}|nota"}


def remove_answer(item: dict) -> dict:
    kept = [v for k, v in sorted(item["options"].items()) if k != item["answer_idx"]]
    opts = {chr(ord("A") + i): v for i, v in enumerate(kept)}
    return {**item, "options": opts, "answer_idx": NONE, "variant": "removed", "id": f"{item['id']}|removed"}


def original(item: dict) -> dict:
    return {**item, "variant": "original", "id": f"{item['id']}|original"}
