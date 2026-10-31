# Copied from ~/Desktop/ecg-research/src/utils/provenance.py (same author, commit 7394049).
# Changes: env fallback renamed ECG_CODE_COMMIT -> CODE_COMMIT; added extra= hook for model/prompt/cost fields; torch import is optional.
"""Provenance record attached to every result file (integrity rule 2: commit, args, seed, device, runtime)."""
from __future__ import annotations

import datetime as _dt
import os
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def provenance(args: dict | None = None, script: str | None = None, seed: int | None = None,
               device: str | None = None, extra: dict | None = None) -> dict:
    """Git commit (and whether the tree is dirty), command line, args, seed, device, date, versions."""
    import numpy as np

    in_git = _git("rev-parse", "--is-inside-work-tree") == "true"
    commit = _git("rev-parse", "HEAD") if in_git else "unknown"
    if commit == "unknown":  # cloud runs (cloud/): no git; the bundle's commit is passed via env
        commit = os.environ.get("CODE_COMMIT", "unknown")
    info = {
        "git_commit": commit,
        # None = unknown (no git checkout, e.g. on Kaggle, where the code comes from a bundle of a clean commit).
        # Shipped results made before this fix record git_dirty=true on Kaggle only because "unknown" was truthy.
        "git_dirty": bool(_git("status", "--porcelain", "--untracked-files=no")) if in_git else None,
        # Added later: which tracked files differ from HEAD, and whether any of them is code.
        # (Before this, appending to the tracked experiments/logs/reproduction.log made every run "dirty".)
        "git_dirty_files": _git("diff", "--name-only", "HEAD").splitlines() if in_git else None,
        "git_code_dirty": bool(_git("diff", "--name-only", "HEAD", "--", "src", "scripts", "configs", "tests"))
        if in_git else None,
        "script": str(Path(script).resolve().relative_to(ROOT)) if script else None,
        "command": " ".join([Path(sys.executable).name, *sys.argv]),
        "args": args,
        "seed": seed,
        "device": device,
        "date": _dt.datetime.now().isoformat(timespec="seconds"),
        "host": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
    }
    try:
        import torch

        info["torch"] = torch.__version__
    except ImportError:
        pass
    # GPU-stack versions (not recorded before this addition; read from package metadata, nothing is imported)
    from importlib.metadata import PackageNotFoundError, version
    libs = {}
    for pkg in ("transformers", "peft", "accelerate", "bitsandbytes", "datasets"):
        try:
            libs[pkg] = version(pkg)
        except PackageNotFoundError:
            libs[pkg] = None
    info["libraries"] = libs
    if extra:
        info.update(extra)
    return info
