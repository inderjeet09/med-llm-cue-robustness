"""Unpack a Kaggle batch zip into the repository, check provenance, and log GPU-hours.

Adapted from ~/Desktop/ecg-research/scripts/ingest_kaggle.py (same author). Added: provenance commit check against
the queue's commit and an automatic row in results/costs/gpu_ledger.csv (sum of job wall-minutes; each job uses one
GPU, so job-minutes = GPU-minutes).
  results/...      -> results/...
  experiments/X    -> experiments/kaggle/X
Usage: python scripts/ingest_kaggle.py cloud/killtest1_results.zip [--notes "..."]
       python scripts/ingest_kaggle.py cloud/quick_check_results.zip --dest reruns   # reviewers: keep the shipped
       results untouched; files go to reruns/results/..., no GPU-ledger row is written
"""
import argparse
import csv
import datetime as _dt
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("zip")
ap.add_argument("--notes", default="")
ap.add_argument("--dest", default=None, help="unpack under this folder instead of the repository (no ledger row)")
a = ap.parse_args()
BASE = Path(a.dest).resolve() if a.dest else ROOT
z = zipfile.ZipFile(a.zip)
n = 0
for name in z.namelist():
    if name.endswith("/"):
        continue
    p = Path(name)
    if p.parts[0] == "experiments":
        dest = BASE / "experiments" / "kaggle" / Path(*p.parts[1:])
    elif p.parts[0] == "results":
        dest = BASE / p
    else:  # job_status.json, queue file
        dest = BASE / "results" / "kaggle_job_status" / f"{Path(a.zip).stem}_{p.name}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(z.read(name))
    n += 1
print(f"extracted {n} files")
st = [x for x in z.namelist() if x.endswith("job_status.json")]
if st:
    s = json.loads(z.read(st[0]))
    ok = sum(v["ok"] for v in s["status"].values())
    print(f"queue {s['queue']} commit {s['commit']}: {ok}/{len(s['status'])} jobs ok")
    for k, v in s["status"].items():
        print(f"  {'OK ' if v['ok'] else 'FAIL'} {k:60s} {v['minutes']:6.1f} min  gpu{v['gpu']}  attempts {v['attempts']}")
    # provenance check: every summary.json from this zip should carry the queue's commit
    bad = []
    for name in z.namelist():
        if name.endswith("summary.json"):
            prov = json.loads(z.read(name)).get("provenance", {})
            if not str(s["commit"]).startswith(str(prov.get("git_commit", ""))[:7]) and \
                    not str(prov.get("git_commit", "")).startswith(str(s["commit"])):
                bad.append((name, prov.get("git_commit")))
    print("provenance commit check:", "OK" if not bad else f"MISMATCH {bad}")
    gpu_h = sum(v["minutes"] for v in s["status"].values()) / 60
    if a.dest:
        print(f"--dest given: {gpu_h:.2f} GPU-hours not logged; results under {BASE}")
        raise SystemExit(0)
    led = ROOT / "results" / "costs" / "gpu_ledger.csv"
    rows = list(csv.DictReader(led.open())) if led.exists() else []
    if any(r["batch"] == Path(a.zip).stem and r["commit"] == str(s["commit"]) for r in rows):  # re-runs = new commit
        print("gpu_ledger: batch already logged; not adding again")
    else:
        with led.open("a", newline="") as f:
            csv.writer(f).writerow([_dt.date.today().isoformat(), Path(a.zip).stem, s["commit"],
                                    s.get("gpu_count", ""), f"{gpu_h:.2f}", a.notes or f"queue {s['queue']}"])
        print(f"gpu_ledger: +{gpu_h:.2f} GPU-hours")
