"""Unit tests for the statistics helpers."""
import numpy as np

from src.evaluation.stats import mcnemar_exact, paired_bootstrap_diff, two_level_bootstrap


def test_paired_bootstrap_and_mcnemar():
    a = np.array([1] * 80 + [0] * 20)
    b = np.array([1] * 50 + [0] * 50)
    d = paired_bootstrap_diff(a, b)
    assert abs(d["diff"] - 0.30) < 1e-9 and d["ci95"][0] > 0
    assert mcnemar_exact(a, b)["p"] < 1e-6 and mcnemar_exact(a, a)["p"] == 1.0


def test_two_level_identical_methods_not_significant():
    rng = np.random.default_rng(1)
    # two arms drawn from the SAME process (3 seeds x 300 items): must not look significant
    base = rng.random(300)
    A = (rng.random((3, 300)) < base).astype(float)
    B = (rng.random((3, 300)) < base).astype(float)
    r = two_level_bootstrap(A, B, seed=0)
    assert r["ci95"][0] <= 0 <= r["ci95"][1] and r["p_value"] > 0.05


def test_two_level_detects_real_gap():
    rng = np.random.default_rng(2)
    A = (rng.random((3, 300)) < 0.8).astype(float)
    B = (rng.random((3, 300)) < 0.5).astype(float)
    r = two_level_bootstrap(A, B, seed=0)
    assert r["ci95"][0] > 0 and r["p_value"] < 0.01
