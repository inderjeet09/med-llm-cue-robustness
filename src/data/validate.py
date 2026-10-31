"""Automatic checks for generated question variants (Phase 2). Each check returns a list of issue strings; an empty
list means the variant is valid. Used by scripts/make_variants.py and the unit tests.

Checked:
  every variant  - options are 4 (or 3 for `removed`), non-empty, no duplicate option texts
  clean / cue    - the gold answer is present and its text is unchanged from the source item
  cue variants   - misleading target != gold, helpful target == gold; the cue sentence contains the target option
                   text; the source question text is preserved apart from the inserted sentence
  nota           - the gold letter now holds the NOTA text; the original correct text is gone; other options unchanged
  removed        - gold is NONE; the original correct text is gone; 3 options remain
Flagged (not invalid, reported for the clinician review / limitations):
  NOTA ambiguity - another option already reads like "none/all of the above", "both", "neither"
"""
from __future__ import annotations

import re

from src.data.bias import TEMPLATES
from src.data.nota import NONE, NOTA_TEXT

_AMBIG = re.compile(r"\b(none of the above|all of the above|both|neither|none)\b", re.I)


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def check_options(item: dict, n_expected: int) -> list[str]:
    issues = []
    opts = item["options"]
    if len(opts) != n_expected:
        issues.append(f"{len(opts)} options (expected {n_expected})")
    if any(not str(v).strip() for v in opts.values()):
        issues.append("empty option text")
    if len({_norm(v) for v in opts.values()}) != len(opts):
        issues.append("duplicate option texts")
    return issues


def check_variant(v: dict, src: dict) -> list[str]:
    """v: a generated variant with field `variant` in {clean, mis:<bias>, help:<bias>, nota, removed}."""
    kind = v["variant"]
    gold_text = src["options"][src["answer_idx"]]
    if kind == "removed":
        issues = check_options(v, len(src["options"]) - 1)
        if v["answer_idx"] != NONE:
            issues.append("removed variant must have gold NONE")
        if any(_norm(t) == _norm(gold_text) for t in v["options"].values()):
            issues.append("correct option text still present")
        return issues
    issues = check_options(v, len(src["options"]))
    if v["answer_idx"] != src["answer_idx"]:
        issues.append("gold letter changed")
    if kind == "nota":
        if v["options"][v["answer_idx"]] != NOTA_TEXT:
            issues.append("gold letter does not hold the NOTA text")
        if any(_norm(t) == _norm(gold_text) for t in v["options"].values()):
            issues.append("correct option text still present")
        if any(v["options"][k] != src["options"][k] for k in src["options"] if k != src["answer_idx"]):
            issues.append("a distractor changed")
        return issues
    if v["options"] != src["options"]:
        issues.append("options changed")
    if kind == "clean":
        if v["question"] != src["question"]:
            issues.append("question text changed")
        return issues
    mode, bias = kind.split(":")
    target = v["cue_target"]
    if mode == "mis" and target == src["answer_idx"]:
        issues.append("misleading cue points to the correct answer")
    if mode == "help" and target != src["answer_idx"]:
        issues.append("helpful cue does not point to the correct answer")
    cue = TEMPLATES[bias].replace("[X]", f"{target}: {src['options'][target]}")
    if cue not in v["question"]:
        issues.append("cue sentence (with target option text) not found in question")
    if _norm(v["question"].replace(cue, " ")) != _norm(src["question"]):
        issues.append("question text changed beyond the inserted cue")
    return issues


def nota_ambiguous(src: dict) -> bool:
    """True if a distractor already reads like a catch-all option, which makes the NOTA variant ambiguous."""
    return any(_AMBIG.search(t) for k, t in src["options"].items() if k != src["answer_idx"])
