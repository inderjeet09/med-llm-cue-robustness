"""API cost accounting: prices from configs/pricing.yaml, the call ledger CSV, and budget caps.

In simple terms: every Azure call (paid or served from our disk cache) becomes one row in
results/costs/api_ledger.csv. The cost of a row is computed from the token counts the API returned and the
verified prices in configs/pricing.yaml. Before a paid call, `check_budget` refuses to proceed if the
worst-case cost of that call would push the run or the whole project past its cap.
"""
from __future__ import annotations

import csv
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
PRICING_PATH = ROOT / "configs" / "pricing.yaml"
BUDGET_PATH = ROOT / "configs" / "budget.yaml"
LEDGER_PATH = ROOT / "results" / "costs" / "api_ledger.csv"

LEDGER_FIELDS = [
    "timestamp", "git_commit", "script", "phase", "experiment", "deployment", "model_version",
    "prompt_tokens", "cached_prompt_tokens", "completion_tokens", "cached", "cost_usd", "price_basis",
    "system_fingerprint", "finish_reason", "cache_key",
]


class BudgetExceeded(RuntimeError):
    """Raised before a call that could exceed the per-run or global cap."""


def load_rates(model: str, deployment_type: str | None, region: str | None,
               pricing_path: Path = PRICING_PATH) -> tuple[dict, str]:
    """Return ({input, cached_input, output} in USD per 1M tokens, price_basis label).

    Falls back to the 'conservative' (highest listed) rates when the deployment type or region is unknown,
    so costs are over- rather than under-stated.
    """
    table = yaml.safe_load(pricing_path.read_text())["models"]
    if model not in table:
        raise KeyError(f"No verified price for model '{model}' in {pricing_path}; verify and add it first.")
    m = table[model]
    dt = (deployment_type or "").strip().lower()
    if dt in ("global", "datazone"):
        return m[dt], dt
    if dt == "regional" and region and region.strip().lower() in m["regional"]:
        r = region.strip().lower()
        return m["regional"][r], f"regional:{r}"
    return m["conservative"], "conservative"


def cost_usd(prompt_tokens: int, cached_prompt_tokens: int, completion_tokens: int, rates: dict) -> float:
    """Cost of one call. Azure bills cached prompt tokens (provider-side prompt caching) at the cached rate."""
    uncached = max(prompt_tokens - cached_prompt_tokens, 0)
    return (uncached * rates["input"] + cached_prompt_tokens * rates["cached_input"]
            + completion_tokens * rates["output"]) / 1e6


def append_row(row: dict, ledger_path: Path = LEDGER_PATH) -> None:
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    new = not ledger_path.exists() or ledger_path.stat().st_size == 0
    with ledger_path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LEDGER_FIELDS)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in LEDGER_FIELDS})


def read_ledger(ledger_path: Path = LEDGER_PATH) -> list[dict]:
    if not ledger_path.exists():
        return []
    with ledger_path.open(newline="") as f:
        return list(csv.DictReader(f))


def ledger_total(ledger_path: Path = LEDGER_PATH) -> float:
    """Total USD actually spent (rows served from our disk cache cost 0)."""
    return sum(float(r["cost_usd"] or 0) for r in read_ledger(ledger_path))


def load_budget(budget_path: Path = BUDGET_PATH) -> dict:
    return yaml.safe_load(budget_path.read_text())


def check_budget(worst_case_usd: float, run_spent_usd: float, run_cap_usd: float, global_cap_usd: float,
                 ledger_path: Path = LEDGER_PATH) -> None:
    """Raise BudgetExceeded if this call's worst-case cost would break either cap."""
    if run_spent_usd + worst_case_usd > run_cap_usd:
        raise BudgetExceeded(f"Run cap ${run_cap_usd:.2f} would be exceeded "
                             f"(spent ${run_spent_usd:.4f} + next call up to ${worst_case_usd:.4f}).")
    total = ledger_total(ledger_path)
    if total + worst_case_usd > global_cap_usd:
        raise BudgetExceeded(f"Global cap ${global_cap_usd:.2f} would be exceeded "
                             f"(project total ${total:.4f} + next call up to ${worst_case_usd:.4f}).")


def estimate_batch_cost(n_calls: int, avg_prompt_tokens: float, avg_completion_tokens: float,
                        rates: dict) -> float:
    """Planning estimate for a batch (shown before runs; never reported as a result)."""
    return n_calls * cost_usd(int(avg_prompt_tokens), 0, int(avg_completion_tokens), rates)
