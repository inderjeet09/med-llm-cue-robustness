"""MedQA-USMLE 4-option loader (official splits; docs/DECISIONS.md D006).

Source: Hugging Face `bigbio/med_qa`, config `med_qa_en_4options_source`, pinned revision below. Upstream data:
Jin et al. 2021 (Applied Sciences, doi:10.3390/app11146421), MIT licence.
`prepare()` writes data/raw/medqa_us4/{train,dev,test}.jsonl once; `load_split()` reads those files.

Item format used everywhere in this project:
    {"id": "dev-00017", "question": str, "options": {"A": str, ...}, "answer_idx": "C", "meta_info": "step1"}
"""
from __future__ import annotations

import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw" / "medqa_us4"
RAW_DIR5 = ROOT / "data" / "raw" / "medqa_us5"   # 5-option version: only for the faithful BiasMedQA reproduction (D017)
HF_REPO, HF_CONFIG = "bigbio/med_qa", "med_qa_en_4options_source"
HF_CONFIG5 = "med_qa_en_source"
HF_REVISION = "484a6c066fe8e75c83edea0c88b5169316714fcd"  # checked 2026-10-17
SPLITS = {"train": "train", "dev": "validation", "test": "test"}
EXPECTED = {"train": 10178, "dev": 1272, "test": 1273}  # Jin et al. 2021, Table 3
# Source-data defects found by scripts/make_variants.py (two identical option texts): never used for training (D016).
EXCLUDED_TRAIN_IDS = frozenset({"train-01318", "train-03960"})


def prepare(out_dir: Path = RAW_DIR, config: str = HF_CONFIG) -> dict:
    """Download the pinned revision and write one jsonl per split. Returns the row counts."""
    from datasets import load_dataset

    ds = load_dataset(HF_REPO, config, revision=HF_REVISION)
    out_dir.mkdir(parents=True, exist_ok=True)
    counts = {}
    for ours, theirs in SPLITS.items():
        rows = ds[theirs]
        with (out_dir / f"{ours}.jsonl").open("w") as f:
            for i, r in enumerate(rows):
                item = {"id": f"{ours}-{i:05d}", "question": r["question"].strip(),
                        "options": {o["key"]: o["value"].strip() for o in r["options"]},
                        "answer_idx": r["answer_idx"], "meta_info": r["meta_info"]}
                assert item["options"][item["answer_idx"]] == r["answer"].strip(), item["id"]
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        counts[ours] = len(rows)
    if counts != EXPECTED:
        raise ValueError(f"split counts {counts} != expected {EXPECTED}")
    return counts


def load_split(split: str, raw_dir: Path = RAW_DIR) -> list[dict]:
    path = raw_dir / f"{split}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing — run `python scripts/prepare_medqa.py` first")
    with path.open() as f:
        return [json.loads(line) for line in f]


def sample(items: list[dict], n: int, seed: int) -> list[dict]:
    """Deterministic subsample (order preserved by original index)."""
    idx = sorted(random.Random(seed).sample(range(len(items)), n))
    return [items[i] for i in idx]
