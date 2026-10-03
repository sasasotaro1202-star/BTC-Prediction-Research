import csv
import io
import zipfile
from datetime import datetime, timezone
from unittest.mock import patch

from src import historical_research_runner as runner


def _zip(rows):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        payload = io.StringIO()
        writer = csv.writer(payload)
        writer.writerows(rows)
        zf.writestr("BTCUSDT-metrics-test.csv", payload.getvalue())
    return buf.getvalue()


def test_metrics_archive_url_is_official_usdm_daily_metrics():
    urls = runner._metrics_candidate_urls(
        "BTCUSDT",
        datetime(2026, 9, 1, tzinfo=timezone.utc).date(),
    )
    assert urls[0] == (
        "https://data.binance.vision/data/futures/um/daily/metrics/"
        "BTCUSDT/BTCUSDT-metrics-2026-09-01.zip"
    )


def test_metrics_oi_rows_parse_create_time_and_conservatively_shift_snapshot():
    raw = _zip([
        ["create_time", "symbol", "sum_open_interest"],
        ["2026-09-01 00:00:00", "BTCUSDT", "100.5"],
        ["2026-09-01 00:05:00", "BTCUSDT", "101.25"],
    ])
    start = int(datetime(2026, 9, 1, tzinfo=timezone.utc).timestamp() * 1000)
    end = start + 11 * 60_000
    with patch.object(runner, "_get_zip", return_value=(raw, "fixture")):
        rows = runner._metrics_oi_zip_rows(
            ["fixture"], start, end, "BTCUSDT", shift_ms=5 * 60_000
        )
    assert [r["timestamp"] for r in rows] == [start + 5 * 60_000, start + 10 * 60_000]
    assert [float(r["sumOpenInterest"]) for r in rows] == [100.5, 101.25]


def test_resilient_runner_routes_open_interest_to_archive_without_live_rest_call():
    url = (
        "https://fapi.binance.com/futures/data/openInterestHist?"
        "symbol=BTCUSDT&period=15m&startTime=1788134400000&endTime=1788584400000&limit=500"
    )
    expected = [{"timestamp": 1788134700000, "sumOpenInterest": "100", "symbol": "BTCUSDT"}]
    with patch.object(runner, "_archive_oi_fallback", return_value=expected) as archive, \
         patch.object(runner, "_ORIGINAL_REQ_JSON") as original:
        out = runner.resilient_req_json(url)
    assert out == expected
    archive.assert_called_once_with(url)
    original.assert_not_called()


def test_archive_oi_fallback_deduplicates_deterministically():
    raw = _zip([
        ["create_time", "symbol", "sum_open_interest"],
        ["2026-09-01 00:00:00", "BTCUSDT", "100"],
        ["2026-09-01 00:05:00", "BTCUSDT", "101"],
    ])
    start = int(datetime(2026, 9, 1, tzinfo=timezone.utc).timestamp() * 1000)
    end = start + 10 * 60_000
    with patch.object(runner, "_get_zip", return_value=(raw, "fixture")), \
         patch.object(runner, "_safe_end", return_value=end):
        rows = runner._archive_oi_fallback(
            "https://fapi.binance.com/futures/data/openInterestHist?"
            f"symbol=BTCUSDT&period=15m&startTime={start}&endTime={end}&limit=500"
        )
    assert len(rows) == 2
    assert [r["timestamp"] for r in rows] == [start + 5 * 60_000, start + 10 * 60_000]
