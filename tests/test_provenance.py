"""Unit tests for the provenance record: without a git checkout the dirty flags are unknown, not true."""
from src.utils import provenance as pv


def test_no_git_means_unknown_not_dirty(monkeypatch):
    monkeypatch.setattr(pv, "_git", lambda *a: "unknown")       # what _git returns when git is unavailable
    monkeypatch.setenv("CODE_COMMIT", "abc1234")
    info = pv.provenance()
    assert info["git_commit"] == "abc1234"
    assert info["git_dirty"] is None and info["git_dirty_files"] is None and info["git_code_dirty"] is None


def test_clean_checkout_is_not_dirty(monkeypatch):
    answers = {("rev-parse", "--is-inside-work-tree"): "true", ("rev-parse", "HEAD"): "deadbeef"}
    monkeypatch.setattr(pv, "_git", lambda *a: answers.get(a, ""))
    info = pv.provenance()
    assert info["git_commit"] == "deadbeef" and info["git_dirty"] is False and info["git_dirty_files"] == []


def test_library_versions_recorded():
    libs = pv.provenance()["libraries"]
    assert set(libs) == {"transformers", "peft", "accelerate", "bitsandbytes", "datasets"}
