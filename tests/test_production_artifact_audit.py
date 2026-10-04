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


def test_current_workspace_sha_allows_ci_ledger_child(monkeypatch, tmp_path):
    expected = "a" * 40
    actual = "b" * 40
    monkeypatch.setenv("GITHUB_SHA", expected)
    monkeypatch.setattr(
        audit.subprocess,
        "check_output",
        lambda *args, **kwargs: actual,
    )
    monkeypatch.setattr(
        audit.subprocess,
        "run",
        lambda *args, **kwargs: type("Completed", (), {"returncode": 0})(),
    )

    assert audit.current_workspace_sha(tmp_path) == actual


def test_current_workspace_sha_fails_closed_on_unrelated_ci_workspace(monkeypatch, tmp_path):
    expected = "a" * 40
    actual = "b" * 40
    monkeypatch.setenv("GITHUB_SHA", expected)
    monkeypatch.setattr(
        audit.subprocess,
        "check_output",
        lambda *args, **kwargs: actual,
    )
    monkeypatch.setattr(
        audit.subprocess,
        "run",
        lambda *args, **kwargs: type("Completed", (), {"returncode": 1})(),
    )

    with pytest.raises(SystemExit, match="not descended from checkout SHA"):
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


def test_current_workspace_sha_rejects_invalid_ci_pin(monkeypatch, tmp_path):
    monkeypatch.setenv("GITHUB_SHA", "not-a-sha")
    monkeypatch.setattr(
        audit.subprocess,
        "check_output",
        lambda *args, **kwargs: "a" * 40 + "\n",
    )

    with pytest.raises(SystemExit, match="GITHUB_SHA is not a valid 40-character Git SHA"):
        audit.current_workspace_sha(tmp_path)


def test_build_provenance_fails_closed_on_missing_required_policy_file(monkeypatch, tmp_path):
    monkeypatch.delenv("GITHUB_SHA", raising=False)
    monkeypatch.setattr(audit, "current_workspace_sha", lambda root: "c" * 40)
    (tmp_path / "PROJECT_INSTRUCTIONS.md").write_text("instructions\n", encoding="utf-8")
    (tmp_path / "requirements.txt").write_text("pytest\n", encoding="utf-8")

    with pytest.raises(SystemExit, match="missing required policy file: docs/PROJECT_SOURCE.md"):
        audit.build_provenance(tmp_path)


def test_build_provenance_records_all_required_policy_hashes(monkeypatch, tmp_path):
    monkeypatch.delenv("GITHUB_SHA", raising=False)
    monkeypatch.setattr(audit, "current_workspace_sha", lambda root: "d" * 40)
    (tmp_path / "PROJECT_INSTRUCTIONS.md").write_text("instructions\n", encoding="utf-8")
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs/PROJECT_SOURCE.md").write_text("source\n", encoding="utf-8")
    (tmp_path / "requirements.txt").write_text("pytest\n", encoding="utf-8")

    provenance = audit.build_provenance(tmp_path)

    assert provenance["git_sha"] == "d" * 40
    assert provenance["checkout_sha"] == "d" * 40
    assert provenance["workspace_sha"] == "d" * 40
    assert provenance["workspace_derived_from_checkout_sha"] is False
    assert set(provenance["policy_files"]) == {
        "PROJECT_INSTRUCTIONS.md",
        "docs/PROJECT_SOURCE.md",
        "requirements.txt",
    }
    assert all(len(value) == 64 for value in provenance["policy_files"].values())
