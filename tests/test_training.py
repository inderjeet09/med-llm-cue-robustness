"""Unit tests for method-A training pairs and the consistency loss."""
import pytest
import torch

from src.training.consistency import letter_logprobs, pair_loss
from src.training.pairs import ARMS, build_pairs


def _train(n=60):
    return [{"id": f"train-{i:05d}", "question": f"A patient presents with sign {i}. What is the diagnosis?",
             "options": {"A": "a", "B": "b", "C": "c", "D": "d"}, "answer_idx": "ABCD"[i % 4]} for i in range(n)]


def test_pairs_train_only_and_targets():
    with pytest.raises(ValueError):
        build_pairs([{**_train(1)[0], "id": "dev-00001"}], 1, "all7")
    pairs = build_pairs(_train(), 45, "all7")
    assert pairs == build_pairs(_train(), 45, "all7")                       # deterministic
    for p in pairs:
        c, q = p["clean"], p["cued"]
        assert q["bias"] in ARMS["all7"]
        if q["cue_helpful"]:
            assert q["cue_target"] == c["answer_idx"]
        else:
            assert q["cue_target"] != c["answer_idx"]
    n_help = sum(p["cued"]["cue_helpful"] for p in pairs)
    assert 0 < n_help < len(pairs) / 2                                       # ~1/3 helpful


def test_heldout_arm_never_sees_eval_cues():
    used = {p["cued"]["bias"] for p in build_pairs(_train(), 60, "heldout4")}
    assert used and not used & {"false_consensus", "frequency", "recency"}
    assert all(p["cued"] is None for p in build_pairs(_train(), 10, "sft_clean"))


def test_loss_kl_zero_when_identical_and_gradient_flows_to_cued_only():
    torch.manual_seed(0)
    letter_ids = [[0], [1], [2], [3, 4]]                                     # letter D has two token variants
    logits_clean = torch.randn(3, 10, requires_grad=True)
    lp = letter_logprobs(logits_clean, letter_ids)
    gold = torch.tensor([0, 1, 3])
    same = pair_loss(lp, lp, gold)
    assert abs(float(same["kl"])) < 1e-6
    logits_cued = torch.randn(3, 10, requires_grad=True)
    out = pair_loss(lp, letter_logprobs(logits_cued, letter_ids), gold, lam=1.0)
    assert float(out["kl"]) > 0
    # KL target is detached: its gradient w.r.t. the clean logits comes only from CE(clean)
    g_full = torch.autograd.grad(out["loss"], logits_clean, retain_graph=True)[0]
    g_ce = torch.autograd.grad(-lp[torch.arange(3), gold].mean(), logits_clean)[0]
    assert torch.allclose(g_full, g_ce, atol=1e-6)
    assert pair_loss(lp, None, gold)["kl"] is None


def test_plan_batches_respects_budgets_and_covers_all():
    from src.llm.hf_runner import plan_batches
    ntok = [300, 1500, 320, 900, 1600, 310, 305, 2000, 1000]
    b = plan_batches(ntok, max_batch=4, max_tokens=3000)
    assert sorted(j for x in b for j in x) == list(range(len(ntok)))          # every prompt exactly once
    assert all(len(x) <= 4 and max(ntok[j] for j in x) * len(x) <= 3000 or len(x) == 1 for x in b)
    assert plan_batches([5000], 8, 1000) == [[0]]                             # over-long prompt alone


def test_phase5_pair_arms_and_bct_loss():
    from src.training.consistency import bct_loss
    nohelp = build_pairs(_train(), 45, "all7", helpful_every=0)
    assert all(not p["cued"]["cue_helpful"] for p in nohelp)
    assert {p["cued"]["bias"] for p in build_pairs(_train(), 60, "all15")} - set(ARMS["all7"])   # new cues present
    lp = letter_logprobs(torch.tensor([[2.0, 0.0, 0.0, 0.0], [0.0, 0.0, 3.0, 0.0]]), [[0], [1], [2], [3]])
    out = bct_loss(lp, torch.tensor([0, 2]))
    assert float(out["loss"]) > 0 and out["kl"] is None


def test_generic_warning_in_prompt():
    from src.llm.prompts import biasmedqa_messages
    item = {"id": "dev-1", "question": "Q?", "options": {"A": "a", "B": "b", "C": "c", "D": "d"}, "answer_idx": "A"}
    msgs, ids = biasmedqa_messages(item, warning="generic_warning_v1")
    assert "not clinical evidence" in msgs[0]["content"] and "generic_warning_v1" in ids
