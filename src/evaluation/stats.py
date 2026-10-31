"""Paired statistics over questions, and the two-level (training seed x question) bootstrap for fine-tuned results.

`two_level_bootstrap` is adapted from ~/Desktop/ecg-research scripts/analyze_final.py `boot2` (same author, commit
7394049). Why it is needed (docs/DECISIONS.md D012, carried over from ecg-research D014): a bootstrap over test items only
treats the trained models as fixed; comparing two runs of the SAME method that way gave a spurious p < 0.001.
Resampling training seeds as well gave the correct, non-significant result."""
from __future__ import annotations

from typing import Callable

import numpy as np


def paired_bootstrap_diff(a: np.ndarray, b: np.ndarray, n_boot: int = 2000, seed: int = 0) -> dict:
    """Mean of (a - b) over items with a percentile 95% CI, resampling items with replacement.
    a, b: per-item scores (e.g. 0/1 correctness) for the SAME items under two conditions."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    boots = d[idx].mean(1)
    return {"diff": float(d.mean()), "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
            "n": int(len(d))}


def mcnemar_exact(a: np.ndarray, b: np.ndarray) -> dict:
    """Exact McNemar test on paired binary outcomes."""
    from scipy.stats import binomtest

    a, b = np.asarray(a, bool), np.asarray(b, bool)
    n01, n10 = int((~a & b).sum()), int((a & ~b).sum())
    p = binomtest(n10, n01 + n10, 0.5).pvalue if n01 + n10 else 1.0
    return {"a_only": n10, "b_only": n01, "p": float(p)}


def two_level_bootstrap(S_a: np.ndarray, S_b: np.ndarray, key_fn: Callable[[np.ndarray], float] | None = None,
                        n_boot: int = 1000, seed: int = 0) -> dict:
    """Difference key(A) - key(B) with a CI that includes training-seed variance.

    S_a, S_b: arrays of shape (n_seeds, n_items) with per-seed, per-item scores (e.g. 0/1 correct) on the SAME items.
    Each replicate resamples seeds (with replacement, within each arm) AND items (shared across arms, keeping the
    pairing). key_fn maps a seed-averaged per-item vector to a scalar (default: mean = accuracy).
    """
    key_fn = key_fn or (lambda v: float(np.mean(v)))
    S_a, S_b = np.asarray(S_a, float), np.asarray(S_b, float)
    assert S_a.shape[1] == S_b.shape[1], "both arms must be scored on the same items"
    rng = np.random.default_rng(seed)
    obs = key_fn(S_a.mean(0)) - key_fn(S_b.mean(0))
    n_items = S_a.shape[1]
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        sa = rng.integers(0, len(S_a), len(S_a))
        sb = rng.integers(0, len(S_b), len(S_b))
        r = rng.integers(0, n_items, n_items)
        diffs[i] = key_fn(S_a[sa].mean(0)[r]) - key_fn(S_b[sb].mean(0)[r])
    return {"diff": float(obs), "ci95": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))],
            "p_value": float(min(1.0, 2 * min((diffs <= 0).mean(), (diffs >= 0).mean()))), "n_boot": n_boot,
            "type": "two-level (seeds+items)"}
