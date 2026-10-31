"""Extract the chosen option letter (or an explicit 'none of the options') from a free-text model response.

Returns a letter "A".."E", "NONE" (the model says no option is correct / abstains), or None (unparseable).
Priority: a lone letter response > an explicit "Answer: X" line (last one wins) > "NONE" phrases > a final
"(X)" / "X:" mention. Unparseable responses are counted as wrong in accuracy and reported separately.
"""
from __future__ import annotations

import re

# The letter itself is case-sensitive ((?-i:...)): otherwise "the answer is a beta blocker" parsed as option A.
# Re-parsing every shipped free-text reply with this fix changes no prediction.
_ANSWER = re.compile(r"(?:final answer|answer)\s*(?:is)?\s*[:\-]?\s*\**\(?((?-i:[A-E]))\)?\**(?![A-Za-z])", re.I)
_ANSWER_NONE = re.compile(r"(?:final answer|answer)\s*(?:is)?\s*[:\-]?\s*\**\s*(none|abstain|no correct)", re.I)
_NONE_PHRASES = re.compile(
    r"\b(none of the (?:options|choices|answers|above|listed)|no (?:correct|valid) (?:option|answer|choice)"
    r"|none is correct|i abstain|cannot select any)\b", re.I)
_LONE = re.compile(r"^\W*\(?([A-E])\)?\W*$")
_MENTION = re.compile(r"(?:^|[\s(])([A-E])(?:\)|:|\.)(?:\s|$)")


def parse_answer(text: str | None, valid: str = "ABCDE") -> str | None:
    if not text:
        return None
    t = text.strip()
    m = _LONE.match(t)
    if m and m.group(1) in valid:
        return m.group(1)
    letters = [x for x in _ANSWER.findall(t) if x.upper() in valid]
    nones = _ANSWER_NONE.findall(t)
    if letters or nones:
        # whichever explicit answer statement comes last
        last_letter = max((mm.start() for mm in _ANSWER.finditer(t) if mm.group(1).upper() in valid), default=-1)
        last_none = max((mm.start() for mm in _ANSWER_NONE.finditer(t)), default=-1)
        return "NONE" if last_none > last_letter else letters[-1].upper()
    if _NONE_PHRASES.search(t[-300:]):
        return "NONE"
    mentions = [x for x in _MENTION.findall(t[-200:]) if x in valid]
    return mentions[-1] if mentions else None
