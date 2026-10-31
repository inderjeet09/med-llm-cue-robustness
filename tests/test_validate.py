"""Unit tests for the Phase 2 variant validator: every variant of a clean item passes; each defect is caught."""
from src.data.bias import TEMPLATES, add_cue
from src.data.nota import nota_replace, remove_answer
from src.data.validate import check_variant, nota_ambiguous

SRC = {"id": "dev-00001", "question": "A 34-year-old woman has fever and a rash. Which is the most likely diagnosis?",
       "options": {"A": "Measles", "B": "Rubella", "C": "Scarlet fever", "D": "Roseola"}, "answer_idx": "C"}


def _cue(bias, helpful):
    x = add_cue(SRC, bias, helpful=helpful)
    return {**x, "variant": f"{'help' if helpful else 'mis'}:{bias}"}


def test_all_variants_of_clean_item_pass():
    vs = [{**SRC, "variant": "clean"}, {**nota_replace(SRC), "variant": "nota"},
          {**remove_answer(SRC), "variant": "removed"}]
    vs += [_cue(b, h) for b in TEMPLATES for h in (False, True)]
    assert all(check_variant(v, SRC) == [] for v in vs)


def test_defects_are_caught():
    bad_target = {**_cue("frequency", False), "cue_target": "C"}
    assert any("points to the correct" in m for m in check_variant(bad_target, SRC))
    edited = _cue("authority_senior", False)
    edited = {**edited, "question": edited["question"].replace("fever", "chills")}
    assert any("beyond the inserted cue" in m for m in check_variant(edited, SRC))
    wrong_help = {**_cue("cds_tool", True), "cue_target": "A"}
    assert check_variant(wrong_help, SRC)
    leaky_nota = {**nota_replace(SRC), "variant": "nota"}
    leaky_nota["options"] = {**leaky_nota["options"], "A": "Scarlet fever"}
    assert any("still present" in m for m in check_variant(leaky_nota, SRC))
    dup = {**SRC, "variant": "clean", "options": {**SRC["options"], "D": "Measles"}}
    assert any("duplicate" in m for m in check_variant(dup, {**SRC, "options": dup["options"]}))


def test_nota_ambiguity_flag():
    assert not nota_ambiguous(SRC)
    assert nota_ambiguous({**SRC, "options": {**SRC["options"], "D": "Both A and B"}})
