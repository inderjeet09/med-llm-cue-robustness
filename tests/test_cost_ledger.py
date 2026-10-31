"""Unit tests for cost accounting and the Azure client's cache/budget logic (no network)."""
from pathlib import Path

import pytest

from src.llm.azure_client import cache_key
from src.utils import cost_ledger as cl


def test_rates_and_basis():
    r, b = cl.load_rates("gpt-4.1-mini", "global", None)
    assert b == "global" and r == {"input": 0.40, "cached_input": 0.10, "output": 1.60}
    assert cl.load_rates("gpt-4.1-mini", "regional", "westeurope")[1] == "regional:westeurope"
    # unknown deployment type / region -> highest price, never lower
    rc, bc = cl.load_rates("gpt-4.1-mini", "", None)
    assert bc == "conservative" and rc["input"] >= r["input"] and rc["output"] >= r["output"]
    assert cl.load_rates("gpt-4.1-mini", "regional", "atlantis")[1] == "conservative"
    with pytest.raises(KeyError):
        cl.load_rates("gpt-unknown", "global", None)


def test_cost_formula():
    r = {"input": 0.40, "cached_input": 0.10, "output": 1.60}
    assert cl.cost_usd(1_000_000, 0, 0, r) == pytest.approx(0.40)
    assert cl.cost_usd(1_000_000, 1_000_000, 0, r) == pytest.approx(0.10)
    assert cl.cost_usd(1000, 200, 500, r) == pytest.approx((800 * 0.40 + 200 * 0.10 + 500 * 1.60) / 1e6)


def test_ledger_and_budget(tmp_path: Path):
    led = tmp_path / "ledger.csv"
    cl.append_row({"experiment": "t", "cost_usd": "0.5", "cached": "no"}, led)
    cl.append_row({"experiment": "t", "cost_usd": "0", "cached": "yes"}, led)
    assert len(cl.read_ledger(led)) == 2 and cl.ledger_total(led) == pytest.approx(0.5)
    cl.check_budget(0.4, 0.0, 1.0, 1.0, led)            # 0.5 + 0.4 <= 1.0: fine
    with pytest.raises(cl.BudgetExceeded):
        cl.check_budget(0.6, 0.0, 10.0, 1.0, led)       # global cap
    with pytest.raises(cl.BudgetExceeded):
        cl.check_budget(0.2, 0.9, 1.0, 10.0, led)       # run cap


def test_cache_key_stable_and_sensitive():
    m = [{"role": "user", "content": "hi"}]
    a = cache_key("dep", m, {"temperature": 0, "max_tokens": 5})
    assert a == cache_key("dep", m, {"max_tokens": 5, "temperature": 0})   # param order irrelevant
    assert a != cache_key("dep", m, {"temperature": 0, "max_tokens": 6})
    assert a != cache_key("dep2", m, {"temperature": 0, "max_tokens": 5})
