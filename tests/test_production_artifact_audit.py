from __future__ import annotations

import pytest

from src import production_artifact_audit as audit


def test_current_workspace_sha_matches_ci_pin(monkeypatch, tmp_path):
    expected = "a" * 40
    monkeypatch.setenv("GITHUB_SHA", expected)
    monkeypatch.setattr(
        audit.subprocess,
        "check_output",
        lambda *args, **kwargs: expected + "\n",
    )

    assert audit.current_workspace_sha(tmp_path) == expected


def test_current_workspace_sha_fails_closed_on_ci_mismatch(monkeypatch, tmp_path):
    expected = "a" * 40
    actual = "b" * 40
    monkeypatch.setenv("GITHUB_SHA", expected)
    monkeypatch.setattr(
        audit.subprocess,
        "check_output",
        lambda *args, **kwargs: actual,
    )

    with pytest.raises(SystemExit, match="workspace SHA mismatch"):
        audit.current_workspace_sha(tmp_path)


def test_current_workspace_sha_allows_local_unpinned_execution(monkeypatch, tmp_path):
    actual = "c" * 40
    monkeypatch.delenv("GITHUB_SHA", raising=False)
    monkeypatch.setattr(
        audit.subprocess,
        "check_output",
        lambda *args, **kwargs: actual,
    )

    assert audit.current_workspace_sha(tmp_path) == actual
