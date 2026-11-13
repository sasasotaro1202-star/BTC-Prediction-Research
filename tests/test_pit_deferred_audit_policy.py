from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.pit_deferred_audit_policy import refresh_required


NOW = datetime(2026, 10, 8, 13, 0, tzinfo=timezone.utc)


def write_report(tmp_path: Path, value: str) -> Path:
    path = tmp_path / "pit_oos_audit.json"
    path.write_text(value, encoding="utf-8")
    return path


def test_missing_report_requires_refresh(tmp_path):
    assert refresh_required(tmp_path / "missing.json", now=NOW) is True


def test_malformed_report_requires_refresh(tmp_path):
    assert refresh_required(write_report(tmp_path, "{bad"), now=NOW) is True


def test_missing_timestamp_requires_refresh(tmp_path):
    path = write_report(tmp_path, '{"status":"PASS"}')
    assert refresh_required(path, now=NOW) is True


def test_recent_report_skips_refresh(tmp_path):
    path = write_report(
        tmp_path,
        '{"generated_at_utc":"2026-10-08T12:50:00+00:00","status":"PASS"}',
    )
    assert refresh_required(path, now=NOW, max_age_seconds=900) is False


def test_boundary_age_is_still_fresh(tmp_path):
    path = write_report(
        tmp_path,
        '{"generated_at_utc":"2026-10-08T12:45:00+00:00","status":"PASS"}',
    )
    assert refresh_required(path, now=NOW, max_age_seconds=900) is False


def test_old_report_requires_refresh(tmp_path):
    path = write_report(
        tmp_path,
        '{"generated_at_utc":"2026-10-08T12:44:59+00:00","status":"PASS"}',
    )
    assert refresh_required(path, now=NOW, max_age_seconds=900) is True


def test_future_dated_report_requires_refresh(tmp_path):
    path = write_report(
        tmp_path,
        '{"generated_at_utc":"2026-10-08T14:00:00+00:00","status":"PASS"}',
    )
    assert refresh_required(path, now=NOW, max_age_seconds=900) is True


def test_invalid_age_argument_fails_closed(tmp_path):
    with pytest.raises(ValueError, match="max_age_seconds_must_be_nonnegative"):
        refresh_required(tmp_path / "missing.json", now=NOW, max_age_seconds=-1)
