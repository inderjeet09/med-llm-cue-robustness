"""Build chat messages from the versioned templates in prompts/ (the only place prompt text lives).

Every builder returns (messages, template_ids) so result files can record exactly which templates produced them.
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

from src.data.bias import EDUCATION, add_cue

ROOT = Path(__file__).resolve().parents[2]
PROMPT_DIR = ROOT / "prompts"
BIAS_NAMES = {"self_diagnosis": "self-diagnosis bias", "recency": "recency bias", "confirmation": "confirmation bias",
              "frequency": "frequency bias", "cultural": "cultural bias", "status_quo": "status quo bias",
              "false_consensus": "false consensus bias", "authority_senior": "authority bias",
              "authority_expert": "authority bias", "referral_note": "anchoring bias", "prelim_report": "anchoring bias",
              "cds_tool": "automation bias", "ai_assistant": "automation bias", "convenience_time": "convenience bias",
              "convenience_cost": "convenience bias"}


@lru_cache(maxsize=None)
def template(name: str) -> str:
    return (PROMPT_DIR / f"{name}.txt").read_text().strip()


def options_block(options: dict, sep: str = ": ") -> str:
    return "\n".join(f"{k}{sep}{v}" for k, v in sorted(options.items()))


# --- axis A: BiasMedQA letter-only format (Schmidgall et al. layout: instruction, question, options, ### Answer) ---
def _biasmedqa_block(item: dict) -> str:
    return f"### Question: {item['question']}\n### Options:\n{options_block(item['options'])}\n### Answer:"


def biasmedqa_messages(item: dict, education: str | None = None,
                       fewshot: tuple[dict, dict] | None = None, fewshot_bias: str | None = None,
                       warning: str | None = None):
    """education: bias key whose education addendum is appended to the instruction.
    fewshot: (negative_example, positive_example) — already cue-injected train items; negative is answered with the
    cue's (wrong) target, positive with the gold answer (Schmidgall few-shot mitigation)."""
    ids = ["biasmedqa_instruction_v1"]
    instr = template("biasmedqa_instruction_v1")
    if education:
        instr = f"{instr} {EDUCATION[education]}"
        ids.append(f"education:{education}")
    if warning:  # a versioned warning template, e.g. "generic_warning_v1" (Phase 5 realistic prompt baseline)
        instr = f"{instr} {template(warning)}"
        ids.append(warning)
    parts = [f"### Instruction: {instr}"]
    if fewshot:
        neg, pos = fewshot
        name = BIAS_NAMES[fewshot_bias or neg["bias"]]
        parts.append(f"The following is an example of incorrectly classifying based on {name}.\n"
                     f"{_biasmedqa_block(neg)} {neg['cue_target']}")
        parts.append(f"The following is an example of correctly classifying based on {name}.\n"
                     f"{_biasmedqa_block(pos)} {pos['answer_idx']}")
        ids.append(f"fewshot:{neg['id']},{pos['id']}")
    parts.append(_biasmedqa_block(item))
    return [{"role": "user", "content": "\n\n".join(parts)}], ids


def fewshot_pair(train: list[dict], bias: str, seed: int = 0) -> tuple[dict, dict]:
    """Deterministic negative/positive demonstration pair from TRAIN for one bias."""
    import random

    a, b = random.Random(f"{seed}|{bias}").sample(range(len(train)), 2)
    return add_cue(train[a], bias), add_cue(train[b], bias)


# --- faithful BiasMedQA reproduction (Phase 3, D017): the paper's Methods layout, verbatim headers ---
def _faithful_block(item: dict) -> str:
    opts = ", ".join(f"{k}: {v}" for k, v in sorted(item["options"].items()))  # options on one line, as in the paper
    return f"### Question: {item['question']}\n\n### Options: {opts}\n\n### Answer:"


def faithful_messages(item: dict, education: str | None = None, demos: list[tuple[dict, str, str]] | None = None):
    """Schmidgall et al. 2024 prompt. demos: [(cue-injected TRAIN item, answer letter, "incorrect"|"correct")].
    One demo = one-shot ("### Example:"), two = few-shot ("### Example 1/2:"), then "### Instruction: Now please
    answer the next question correctly." before the test question."""
    ids = ["biasmedqa_faithful_instruction_v1"]
    instr = template("biasmedqa_faithful_instruction_v1")
    if education:
        instr = f"{instr} {EDUCATION[education]}"
        ids.append(f"education:{education}")
    parts = [f"### Instruction: {instr}"]
    if demos:
        for i, (ex, ans, kind) in enumerate(demos, 1):
            head = template(f"biasmedqa_demo_{kind}_v1")
            label = "### Example:" if len(demos) == 1 else f"### Example {i}:"
            parts.append(f"{label} {head}\n\n{_faithful_block(ex)} {ans}")
            ids.append(f"demo_{kind}:{ex['id']}")
        parts.append(f"### Instruction: {template('biasmedqa_next_question_v1')}")
        ids.append("biasmedqa_next_question_v1")
    parts.append(_faithful_block(item))
    return [{"role": "user", "content": "\n\n".join(parts)}], ids


# --- axis B ---
def cot_messages(item: dict):
    return [{"role": "system", "content": template("cot_mcq_v1")},
            {"role": "user", "content": f"{item['question']}\n\n{options_block(item['options'], '. ')}"}], ["cot_mcq_v1"]


def handler_messages(item: dict, safety: bool):
    name = "handler_safety_v1" if safety else "handler_baseline_v1"
    L = sorted(item["options"])
    letters = ", ".join(L[:-1]) + f", or {L[-1]}"
    sys_ = template(name).format(letters=letters, letter_range=f"{L[0]}–{L[-1]}", letters_slash="/".join(L))
    user = f"{item['question']}\n\nOptions:\n{options_block(item['options'], '. ')}"
    return [{"role": "system", "content": sys_}, {"role": "user", "content": user}], [name]


def gtv_step1_messages(item: dict):
    return [{"role": "system", "content": template("gtv_step1_v1")},
            {"role": "user", "content": item["question"]}], ["gtv_step1_v1"]


_MY_ANSWER = re.compile(r"my answer\s*:\s*(.+)", re.I)


def extract_own_answer(text: str | None) -> str:
    if not text:
        return ""
    m = _MY_ANSWER.findall(text)
    return (m[-1] if m else text.strip().splitlines()[-1]).strip().strip('"*').strip()


def gtv_step2_messages(item: dict, own_answer: str):
    sys_ = template("gtv_step2_v1").format(own_answer=own_answer)
    user = f"{item['question']}\n\nOptions:\n{options_block(item['options'], '. ')}"
    return [{"role": "system", "content": sys_}, {"role": "user", "content": user}], ["gtv_step2_v1"]


def parse_handler_json(text: str | None, valid: str) -> str | None:
    """Handler-format reply -> letter, 'NONE' (abstained) or None (unparseable)."""
    if not text:
        return None
    if text.strip().strip('`"').strip().lower() in ("null", "none"):  # bare null = abstained
        return "NONE"
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        try:
            obj = json.loads(m.group(0))
            ch = str(obj.get("choice") or "").strip().upper()
            if ch in valid and len(ch) == 1:
                return ch
            if ch in ("", "NULL", "NONE"):
                return "NONE"
        except json.JSONDecodeError:
            pass
    mc = re.search(r'"choice"\s*:\s*"?([A-Za-z]+)"?', text)
    if mc:
        ch = mc.group(1).upper()
        return ch if (len(ch) == 1 and ch in valid) else ("NONE" if ch in ("NULL", "NONE") else None)
    return None
