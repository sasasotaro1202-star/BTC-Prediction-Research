from datetime import datetime, timezone
from io import BytesIO
import csv
import io
import zipfile

from gdelt_exogenous_research import parse_slice


def _row(published: str, title: str, url: str = "https://example.test/btc") -> list[str]:
    row = [""] * 27
    row[0] = "record-1"
    row[1] = published
    row[3] = "Example"
    row[4] = url
    row[8] = ""
    row[15] = "2.0,0,0,0,0,0"
    row[26] = f"<PAGE_TITLE>{title}</PAGE_TITLE>"
    return row


def _zip(rows: list[list[str]]) -> bytes:
    raw = io.StringIO()
    w = csv.writer(raw, delimiter="\t", lineterminator="\n")
    w.writerows(rows)
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("sample.gkg.csv", raw.getvalue())
    return buf.getvalue()


def test_parser_uses_slice_time_as_conservative_availability_bound():
    available = datetime(2026, 9, 21, 12, 15, tzinfo=timezone.utc)
    rows = [
        _row("20260921120000", "Bitcoin ETF news"),
        _row("20260921130000", "Future bitcoin news", "https://example.test/future"),
    ]
    events = parse_slice(_zip(rows), available)
    assert len(events) == 1
    assert events[0]["event_type"] == "news"
    assert events[0]["available_at"] == available.isoformat()
    assert events[0]["sentiment"] == 0.2
    assert events[0]["research_only"] is True
    assert events[0]["production_changed"] is False


def test_policy_and_macro_are_classified_separately_for_research_diagnostics():
    available = datetime(2026, 9, 21, 12, 15, tzinfo=timezone.utc)
    rows = [
        _row("20260921120000", "SEC regulation of bitcoin"),
        _row("20260921120000", "Federal Reserve interest rate and bitcoin",
             "https://example.test/macro"),
    ]
    events = parse_slice(_zip(rows), available)
    assert {x["event_type"] for x in events} == {"policy", "macro"}


def test_coverage_requires_every_15_minute_slice_in_lookback():
    from datetime import timedelta
    from gdelt_exogenous_research import coverage_complete, _slice_stamps_for_window
    t = datetime(2026, 9, 21, 12, 0, tzinfo=timezone.utc)
    required = set(_slice_stamps_for_window(t, timedelta(hours=1)))
    assert len(required) == 5
    assert coverage_complete(required, t, timedelta(hours=1))
    assert coverage_complete(required - {min(required)}, t, timedelta(hours=1)) is False


def test_coverage_normalizes_prediction_timezone():
    from datetime import timedelta
    from gdelt_exogenous_research import coverage_complete, _slice_stamps_for_window
    t = datetime(2026, 9, 21, 14, 0, tzinfo=timezone.utc)
    equivalent = datetime(2026, 9, 21, 16, 0, tzinfo=timezone(timedelta(hours=2)))
    required = set(_slice_stamps_for_window(t, timedelta(hours=1)))
    assert coverage_complete(required, t, timedelta(hours=1))
    assert coverage_complete(required, equivalent, timedelta(hours=1))
