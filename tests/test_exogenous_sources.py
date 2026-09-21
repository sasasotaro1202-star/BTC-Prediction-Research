from datetime import datetime, timezone
from exogenous_sources import _parse_seen_time

def test_gdelt_seen_time_is_timezone_aware():
    value = _parse_seen_time("20260921T120530Z")
    assert value == datetime(2026, 9, 21, 12, 5, 30, tzinfo=timezone.utc)

def test_gdelt_bad_seen_time_fails_closed():
    assert _parse_seen_time("bad") is None
