"""Download MedQA-US (pinned bigbio revision) and verify the official split counts.
4-option -> data/raw/medqa_us4/ (default, used everywhere); 5-option -> data/raw/medqa_us5/ (faithful BiasMedQA
reproduction only, D017).

Usage: python scripts/prepare_medqa.py [--options 5]
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.medqa import HF_CONFIG, HF_CONFIG5, HF_REVISION, RAW_DIR, RAW_DIR5, prepare  # noqa: E402
from src.utils.provenance import provenance  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--options", type=int, choices=[4, 5], default=4)
args = ap.parse_args()
raw, cfg = (RAW_DIR5, HF_CONFIG5) if args.options == 5 else (RAW_DIR, HF_CONFIG)
counts = prepare(raw, cfg)
meta = {"counts": counts, "hf_revision": HF_REVISION, "hf_config": cfg, "provenance": provenance(script=__file__)}
(raw / "PREPARED.json").write_text(json.dumps(meta, indent=1))
print(counts)
