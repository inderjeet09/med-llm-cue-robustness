"""Loss for method A: gold-label SFT on clean and cued questions + KL consistency between their answer distributions.

Works on the next-token logits at the answer position only (the same quantity `HFModel.letter_probs` evaluates), so
training and evaluation measure exactly the same thing.
    CE  = -log P(gold letter) over the full vocabulary, pooling the letter's token variants ("A", " A")
    KL  = KL( stopgrad(p_clean) || p_cued ) over the A-D distribution renormalised to the letters
"""
from __future__ import annotations

import torch


def letter_logprobs(last_logits: torch.Tensor, letter_ids: list[list[int]]) -> torch.Tensor:
    """[B, vocab] logits -> [B, n_letters] log P(letter) under the full-vocabulary softmax."""
    logp = torch.log_softmax(last_logits.float(), dim=-1)
    return torch.stack([torch.logsumexp(logp[:, ids], dim=-1) for ids in letter_ids], dim=-1)


def pair_loss(lp_clean: torch.Tensor, lp_cued: torch.Tensor | None, gold: torch.Tensor, lam: float = 1.0) -> dict:
    """lp_*: [B, n_letters] from `letter_logprobs`; gold: [B] letter indices. lp_cued=None -> clean SFT only."""
    rows = torch.arange(len(gold), device=gold.device)
    ce_clean = -lp_clean[rows, gold].mean()
    if lp_cued is None:
        return {"loss": ce_clean, "ce_clean": ce_clean.detach(), "ce_cued": None, "kl": None}
    ce_cued = -lp_cued[rows, gold].mean()
    q = torch.log_softmax(lp_clean.detach(), dim=-1)          # renormalised over letters, no gradient (target)
    p = torch.log_softmax(lp_cued, dim=-1)
    kl = (q.exp() * (q - p)).sum(-1).mean()
    loss = ce_clean + ce_cued + lam * kl
    return {"loss": loss, "ce_clean": ce_clean.detach(), "ce_cued": ce_cued.detach(), "kl": kl.detach()}


def bct_loss(lp_cued: torch.Tensor, own_clean_answer: torch.Tensor) -> dict:
    """BCT (Chua et al., arXiv 2403.05518) adapted to letter answers: cross-entropy of the CUED prompt towards the base
    model's own argmax answer on the CLEAN prompt (no gold labels, no consistency term)."""
    rows = torch.arange(len(own_clean_answer), device=own_clean_answer.device)
    ce = -lp_cued[rows, own_clean_answer].mean()
    return {"loss": ce, "ce_cued": ce.detach(), "ce_clean": None, "kl": None}
