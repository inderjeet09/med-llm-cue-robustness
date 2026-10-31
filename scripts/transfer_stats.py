"""Statistics of the MedQA-MultiTurnRobust transfer set cited in the paper, saved to results/ so that the paper numbers can
be regenerated without the raw dataset (which is not redistributed). Usage: python scripts/transfer_stats.py"""
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.multiturn import HF_REVISION, load  # noqa: E402
from src.utils.provenance import provenance  # noqa: E402

items = load()
out = {"n_items": len(items), "context_words_median": statistics.median(len(x["misleading_context"].split()) for x in items),
       "hf_revision": HF_REVISION, "provenance": provenance({}, __file__)}
(ROOT / "results/data/transfer_stats.json").write_text(json.dumps(out, indent=1))
print(out["n_items"], out["context_words_median"])
