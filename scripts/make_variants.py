"""Phase 2: generate and validate every question variant for the dev and test splits.

Per source question: clean; misleading and helpful cue for each of the 15 templates (7 BiasMedQA + 8 new); nota;
removed. Written to data/processed/medqa_us4_variants/{dev,test}.jsonl (gitignored; regenerate with this script).
Train variants are NOT written to disk: training pairs are built deterministically at train time
(src/training/pairs.py); the same transforms are validated here on every train item in memory.

Also checks MedQA itself for exact duplicate questions within and across splits (possible leakage).
Writes results/data/variants_manifest.json (counts, issues, duplicates, file hashes, template hash).
Usage: python scripts/make_variants.py
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.bias import BIASMEDQA, EDUCATION, FAMILY_OF, NEW_BIASES, TEMPLATES, add_cue  # noqa: E402
from src.data.medqa import load_split  # noqa: E402
from src.data.nota import nota_replace, remove_answer  # noqa: E402
from src.data.validate import check_variant, nota_ambiguous  # noqa: E402
from src.utils.provenance import provenance  # noqa: E402

OUT_DIR = ROOT / "data/processed/medqa_us4_variants"


def variants_of(item: dict) -> list[dict]:
    src_id = item["id"]
    out = [{**item, "variant": "clean", "src_id": src_id, "id": f"{src_id}|clean"}]
    for bias in TEMPLATES:
        for mode, helpful in (("mis", False), ("help", True)):
            x = add_cue(item, bias, helpful=helpful)
            out.append({**x, "variant": f"{mode}:{bias}", "family": FAMILY_OF[bias], "src_id": src_id,
                        "id": f"{src_id}|{mode}:{bias}"})
    out.append({**nota_replace(item), "variant": "nota", "src_id": src_id})
    out.append({**remove_answer(item), "variant": "removed", "src_id": src_id})
    return out


def norm_q(q: str) -> str:
    return re.sub(r"\s+", " ", q.strip().lower())


def main() -> None:
    splits = {s: load_split(s) for s in ("train", "dev", "test")}
    manifest: dict = {"templates_sha256": hashlib.sha256(json.dumps([TEMPLATES, EDUCATION], sort_keys=True)
                                                         .encode()).hexdigest(),
                      "biasmedqa": BIASMEDQA, "new_biases": NEW_BIASES, "splits": {}}

    # 1) duplicates inside MedQA (exact, whitespace/case-normalised question stems)
    where = defaultdict(list)
    for s, items in splits.items():
        for it in items:
            where[norm_q(it["question"])].append(f"{s}:{it['id']}")
    dup_groups = [v for v in where.values() if len(v) > 1]
    cross = [v for v in dup_groups if len({x.split(":")[0] for x in v}) > 1]
    manifest["medqa_duplicates"] = {
        "groups_total": len(dup_groups),
        "groups_within_split_only": len(dup_groups) - len(cross),
        "groups_across_splits": len(cross),
        "train_ids_overlapping_dev_or_test": sorted({x.split(":")[1] for v in cross for x in v
                                                     if x.startswith("train:")}),
        "examples_across": cross[:10]}

    # 2) generate + validate
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for s, items in splits.items():
        issues: Counter = Counter()
        examples: dict[str, str] = {}
        n_rows, ids = 0, set()
        f = (OUT_DIR / f"{s}.jsonl").open("w") if s != "train" else None
        for it in items:
            for v in variants_of(it):
                for msg in check_variant(v, it):
                    key = f"{v['variant'].split(':')[0]}: {msg}"
                    issues[key] += 1
                    examples.setdefault(key, v["id"])
                if v["id"] in ids:
                    issues["duplicate variant id"] += 1
                ids.add(v["id"])
                n_rows += 1
                if f:
                    f.write(json.dumps(v, ensure_ascii=False) + "\n")
        if f:
            f.close()
        manifest["splits"][s] = {
            "source_questions": len(items), "variant_rows": n_rows, "written": f is not None,
            "issues": dict(issues), "issue_examples": examples,
            "nota_ambiguous_questions": sum(nota_ambiguous(it) for it in items),
            "file_sha256": hashlib.sha256((OUT_DIR / f"{s}.jsonl").read_bytes()).hexdigest() if f else None}
        print(f"{s:5s} {len(items):6d} questions -> {n_rows:7d} variants | issues: {dict(issues) or 'none'} | "
              f"NOTA-ambiguous: {manifest['splits'][s]['nota_ambiguous_questions']}", flush=True)

    print(f"MedQA duplicate stems: {manifest['medqa_duplicates']['groups_total']} groups, "
          f"{manifest['medqa_duplicates']['groups_across_splits']} across splits")
    manifest["provenance"] = provenance({}, __file__)
    out = ROOT / "results/data/variants_manifest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=1))
    print("->", out.relative_to(ROOT))


if __name__ == "__main__":
    main()
