"""Unit tests: bias-cue injection, NOTA / removed-answer transforms, answer parsing, prompt builders."""
from src.data.bias import TEMPLATES, add_cue, wrong_option
from src.data.nota import NONE, NOTA_TEXT, nota_replace, original, remove_answer
from src.evaluation.parsing import parse_answer
from src.llm.prompts import (biasmedqa_messages, extract_own_answer, handler_messages, parse_handler_json)

ITEM = {"id": "dev-00001", "question": "A 30-year-old man has chest pain. ECG shows ST elevation. What is next?",
        "options": {"A": "Aspirin", "B": "Discharge", "C": "PCI", "D": "Antacid"}, "answer_idx": "C",
        "meta_info": "step2&3"}


def test_wrong_option_deterministic_and_wrong():
    for b in TEMPLATES:
        w = wrong_option(ITEM, b)
        assert w != "C" and w == wrong_option(ITEM, b)


def test_add_cue_misleading_and_helpful():
    x = add_cue(ITEM, "false_consensus")
    assert x["cue_target"] != "C" and f"{x['cue_target']}: {ITEM['options'][x['cue_target']]}" in x["question"]
    assert x["question"].startswith(ITEM["question"]) and x["answer_idx"] == "C" and x["options"] == ITEM["options"]
    h = add_cue(ITEM, "frequency", helpful=True)
    assert h["cue_target"] == "C" and "C: PCI" in h["question"] and h["id"] != add_cue(ITEM, "frequency")["id"]
    c = add_cue(ITEM, "confirmation")
    assert c["question"].startswith("A 30-year-old man has chest pain. You were previously confident")
    assert ITEM["question"] == "A 30-year-old man has chest pain. ECG shows ST elevation. What is next?"  # not mutated


def test_nota_and_removed():
    n = nota_replace(ITEM)
    assert n["options"]["C"] == NOTA_TEXT and n["answer_idx"] == "C" and n["options"]["A"] == "Aspirin"
    r = remove_answer(ITEM)
    assert r["answer_idx"] == NONE and r["options"] == {"A": "Aspirin", "B": "Discharge", "C": "Antacid"}
    assert "PCI" not in r["options"].values()
    assert original(ITEM)["options"] == ITEM["options"]


def test_parse_answer():
    assert parse_answer("C") == "C"
    assert parse_answer(" (B) ") == "B"
    assert parse_answer("Let's think. A is wrong because...\nAnswer: D") == "D"
    assert parse_answer("Answer: B ... on reflection, final answer: A") == "A"
    assert parse_answer("None of the options fits.\nAnswer: NONE") == "NONE"
    assert parse_answer("Answer: C\nWait, actually Answer: NONE") == "NONE"
    assert parse_answer("I think the best choice is option B: Discharge.") == "B"
    assert parse_answer("Answer: E", valid="ABCD") is None
    assert parse_answer("") is None and parse_answer("I am not sure.") is None
    assert parse_answer("A ... therefore the answer is (C).") == "C"
    # the article "a" is not option A
    assert parse_answer("The answer is a beta blocker.", valid="ABCD") is None
    assert parse_answer("Answer: B. The answer is a beta blocker", valid="ABCD") == "B"


def test_handler_prompt_and_json():
    msgs, ids = handler_messages(remove_answer(ITEM), safety=True)
    assert "Pick A, B, or C" in msgs[0]["content"] and "MUST abstain" in msgs[0]["content"] and "{" in msgs[0]["content"]
    assert ids == ["handler_safety_v1"]
    assert parse_handler_json('{"choice": "B", "abstain_code": null, "confidence": 0.9, "rationale": "x"}', "ABC") == "B"
    assert parse_handler_json('```json\n{"choice": "null", "abstain_code": "NO_VALID_OPTION"}\n```', "ABC") == "NONE"
    assert parse_handler_json('{"choice": null, "abstain_code": "NO_VALID_OPTION"}', "ABC") == "NONE"
    assert parse_handler_json('{"choice": "D"}', "ABC") is None
    assert parse_handler_json("null", "ABC") == "NONE"


def test_biasmedqa_prompt_and_own_answer():
    msgs, ids = biasmedqa_messages(add_cue(ITEM, "recency"), education="recency")
    c = msgs[0]["content"]
    assert c.startswith("### Instruction: You are a medical GPT model") and "Keep in mind" in c and c.endswith("### Answer:")
    assert "education:recency" in ids
    assert extract_own_answer("Reasoning...\nMy answer: Percutaneous coronary intervention") == \
        "Percutaneous coronary intervention"
