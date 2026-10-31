"""Training pairs for method A (counterfactual-invariance fine-tuning). TRAIN split only (integrity rule 3).

Each pair = (clean question, cued question) built from the SAME train item. The cue is misleading (points to a wrong
option) for most pairs and helpful (points to the correct option) for the rest, so the model learns to judge hints
rather than to ignore or invert every hint. All choices are deterministic hashes of (item id, arm, seed).
Pre-registered in docs/preregistration.md (stage 2).
"""
from __future__ import annotations

import hashlib

from src.data.bias import BIASMEDQA, TEMPLATES, add_cue
from src.data.medqa import EXCLUDED_TRAIN_IDS, sample

# Template sets per training arm. heldout4 never sees the three cues used in evaluation.
ARMS: dict[str, list[str]] = {
    "all7": list(BIASMEDQA),
    "heldout4": ["self_diagnosis", "confirmation", "cultural", "status_quo"],
    "sft_clean": [],
    "all15": list(TEMPLATES),   # Phase 5 optional ablation: no held-out cue types
}


def _h(*parts) -> int:
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest(), 16)


def build_pairs(train: list[dict], n: int, arm: str, seed: int = 1, helpful_every: int = 3) -> list[dict]:
    """n pairs {"clean": item, "cued": item | None}. Every `helpful_every`-th pair (by hash) gets a helpful cue;
    helpful_every=0 means no helpful cues at all (Phase 5 ablation)."""
    if any(not it["id"].startswith("train-") for it in train):
        raise ValueError("training pairs must come from the TRAIN split only")
    biases = ARMS[arm]
    pool = [it for it in train if it["id"] not in EXCLUDED_TRAIN_IDS]  # source defects (D016)
    pairs = []
    for it in sample(pool, n, seed):
        if not biases:
            pairs.append({"clean": it, "cued": None})
            continue
        h = _h(it["id"], arm, seed)
        bias = biases[h % len(biases)]
        helpful = bool(helpful_every) and (h // 1009) % helpful_every == 0
        pairs.append({"clean": it, "cued": add_cue(it, bias, helpful=helpful)})
    return pairs
