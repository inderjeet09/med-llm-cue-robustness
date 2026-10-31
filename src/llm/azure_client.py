"""The ONLY way this project calls Azure OpenAI.

Guarantees:
  1. Disk cache keyed by sha256(deployment, messages, params): an identical request is never paid twice.
     Cache files: results/costs/api_cache/<key[:2]>/<key>.json (gitignored; human-readable JSON).
  2. Every call (paid or cache hit) is logged to results/costs/api_ledger.csv.
  3. Cost from configs/pricing.yaml (verified prices only).
  4. Per-run and global budget caps are checked BEFORE every paid call (worst case = generous prompt-token
     estimate + max_tokens of output).
  5. temperature=0 and a fixed seed by default; system_fingerprint and the returned model version are recorded.

Usage:
    from src.llm.azure_client import AzureClient
    client = AzureClient(experiment="pilot_testB", phase="1.5")
    out = client.chat([{"role": "user", "content": "..."}], max_tokens=16)
    out["text"], out["cost_usd"], out["cached"]
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import sys
import threading
import time
from pathlib import Path

from dotenv import load_dotenv

from src.utils.cost_ledger import (LEDGER_PATH, ROOT, append_row, check_budget, cost_usd, load_budget,
                                   load_rates)
from src.utils.provenance import _git

CACHE_DIR = ROOT / "results" / "costs" / "api_cache"
PRICED_MODEL = "gpt-4.1-mini"  # key into configs/pricing.yaml; change only after verifying a new price


def cache_key(deployment: str, messages: list[dict], params: dict) -> str:
    blob = json.dumps({"deployment": deployment, "messages": messages, "params": params},
                      sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _worst_case_prompt_tokens(messages: list[dict]) -> int:
    # Budget guard only: ~1 token per 3 characters plus per-message overhead over-estimates English text.
    return sum(len(str(m.get("content", ""))) // 3 + 8 for m in messages) + 16


class AzureClient:
    def __init__(self, experiment: str, phase: str = "", run_cap_usd: float | None = None,
                 cache_dir: Path = CACHE_DIR, ledger_path: Path = LEDGER_PATH, offline: bool = False):
        load_dotenv(ROOT / ".env")
        self.experiment, self.phase = experiment, phase
        self.cache_dir, self.ledger_path = Path(cache_dir), Path(ledger_path)
        self.deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "")
        budget = load_budget()
        self.run_cap = float(run_cap_usd if run_cap_usd is not None else budget["default_run_cap_usd"])
        self.global_cap = float(budget["global_cap_usd"])
        self.rates, self.price_basis = load_rates(PRICED_MODEL, os.environ.get("AZURE_OPENAI_DEPLOYMENT_TYPE"),
                                                  os.environ.get("AZURE_OPENAI_REGION"))
        self.run_spent = 0.0
        self._lock = threading.Lock()  # chat() may be called from a thread pool
        self.commit = _git("rev-parse", "--short", "HEAD")
        if self.commit == "unknown":
            self.commit = os.environ.get("CODE_COMMIT", "unknown")
        self.script = Path(sys.argv[0]).name if sys.argv and sys.argv[0] else "interactive"
        self._client = None
        self.offline = offline  # True: cache only, never call the API (for re-analysis)

    # -- internals -----------------------------------------------------------------------------------
    def _api(self):
        if self._client is None:
            from openai import AzureOpenAI
            # Accept AZURE_OPENAI_API_ENDPOINT as an alias for the documented AZURE_OPENAI_ENDPOINT.
            if not os.environ.get("AZURE_OPENAI_ENDPOINT") and os.environ.get("AZURE_OPENAI_API_ENDPOINT"):
                os.environ["AZURE_OPENAI_ENDPOINT"] = os.environ["AZURE_OPENAI_API_ENDPOINT"]
            missing = [v for v in ("AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_DEPLOYMENT",
                                   "AZURE_OPENAI_API_VERSION") if not os.environ.get(v)]
            if missing:
                raise RuntimeError(f"Missing in .env: {', '.join(missing)}")
            self._client = AzureOpenAI(azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
                                       api_key=os.environ["AZURE_OPENAI_API_KEY"],
                                       api_version=os.environ["AZURE_OPENAI_API_VERSION"])
        return self._client

    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / key[:2] / f"{key}.json"

    def _log(self, rec: dict, cached: bool, cost: float) -> None:
        u = rec.get("usage") or {}
        with self._lock:
            self._append(rec, u, cached, cost)

    def _append(self, rec: dict, u: dict, cached: bool, cost: float) -> None:
        append_row({
            "timestamp": _dt.datetime.now().isoformat(timespec="seconds"), "git_commit": self.commit,
            "script": self.script, "phase": self.phase, "experiment": self.experiment,
            "deployment": rec.get("deployment", self.deployment), "model_version": rec.get("model", ""),
            "prompt_tokens": u.get("prompt_tokens", 0), "cached_prompt_tokens": u.get("cached_prompt_tokens", 0),
            "completion_tokens": u.get("completion_tokens", 0), "cached": "yes" if cached else "no",
            "cost_usd": f"{cost:.8f}", "price_basis": self.price_basis,
            "system_fingerprint": rec.get("system_fingerprint", ""), "finish_reason": rec.get("finish_reason", ""),
            "cache_key": rec.get("cache_key", ""),
        }, self.ledger_path)

    # -- public ----------------------------------------------------------------------------------------
    def chat(self, messages: list[dict], max_tokens: int = 512, temperature: float = 0.0, seed: int = 0,
             max_retries: int = 6, **params) -> dict:
        """One chat completion. Returns dict(text, finish_reason, usage, model, system_fingerprint,
        cached, cost_usd, cache_key)."""
        req = {"max_tokens": max_tokens, "temperature": temperature, "seed": seed, **params}
        key = cache_key(self.deployment, messages, req)
        path = self._cache_path(key)
        if path.exists():
            rec = json.loads(path.read_text())
            self._log(rec, cached=True, cost=0.0)
            return {**rec, "cached": True, "cost_usd": 0.0}
        if self.offline:
            raise LookupError(f"offline mode and no cached response for key {key}")

        worst = cost_usd(_worst_case_prompt_tokens(messages), 0, max_tokens, self.rates)
        with self._lock:
            check_budget(worst, self.run_spent, self.run_cap, self.global_cap, self.ledger_path)
            self.run_spent += worst  # reserve worst case; corrected to the real cost below

        from openai import APIConnectionError, APITimeoutError, BadRequestError, InternalServerError, RateLimitError
        for attempt in range(max_retries):
            try:
                resp = self._api().chat.completions.create(model=self.deployment, messages=messages, **req)
                break
            except (RateLimitError, APIConnectionError, APITimeoutError, InternalServerError):
                if attempt == max_retries - 1:
                    raise
                time.sleep(min(2 ** attempt * 2, 60))
            except BadRequestError as e:  # e.g. Azure content filter on the prompt: not cached, logged at $0
                with self._lock:
                    self.run_spent -= worst
                rec = {"deployment": self.deployment, "text": None, "finish_reason": "error:" + str(e.code),
                       "usage": {}, "model": "", "system_fingerprint": "", "cache_key": key}
                self._log(rec, cached=False, cost=0.0)
                return {**rec, "cached": False, "cost_usd": 0.0}

        ch = resp.choices[0]
        details = getattr(resp.usage, "prompt_tokens_details", None)
        usage = {"prompt_tokens": resp.usage.prompt_tokens, "completion_tokens": resp.usage.completion_tokens,
                 "cached_prompt_tokens": (getattr(details, "cached_tokens", 0) or 0) if details else 0}
        rec = {"deployment": self.deployment, "request": {"messages": messages, "params": req},
               "text": ch.message.content, "finish_reason": ch.finish_reason, "usage": usage,
               "model": resp.model, "system_fingerprint": resp.system_fingerprint or "", "cache_key": key,
               "created": _dt.datetime.now().isoformat(timespec="seconds")}
        cost = cost_usd(usage["prompt_tokens"], usage["cached_prompt_tokens"], usage["completion_tokens"], self.rates)
        if ch.finish_reason != "content_filter":
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(rec, ensure_ascii=False, indent=1))
        with self._lock:
            self.run_spent += cost - worst
        self._log(rec, cached=False, cost=cost)
        return {**rec, "cached": False, "cost_usd": cost}
