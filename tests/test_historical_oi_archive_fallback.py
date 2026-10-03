import csv
import io
import zipfile
from unittest.mock import patch

from src import historical_research_runner as runner


def _zip(rows):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        payload = io.StringIO()
        writer = csv.writer(payload)
        writer.writerow(["create_time", "symbol", "sum_open_interest", "sum_open_interest_value"])
        writer.writerows(rows)
        zf.writestr("BTCUSDT-metrics-test.csv", payload.getvalue())
    return buf.getvalue()


def test_oi_metrics_archive_parser_normalizes_and_filters_rows():
    payload = _zip([
        ["1759536000000", "BTCUSDT", "100.0", "1000000.0"],
        ["1759536300000", "BTCUSDT", "101.0", "1010000.0"],
        ["1759540000000", "BTCUSDT", "not-a-number", "0"],
    ])
    with patch.object(runner, "_get_zip", return_value=(payload, "test")):
        out = runner._oi_metrics_zip_rows(
            ["ignored"],
            1759536000000,
            1759539000000,
        )
    assert [row["timestamp"] for row in out] == [1759536000000, 1759536300000]
    assert [row["sumOpenInterest"] for row in out] == ["100.0", "101.0"]
    assert all(row["pit_status"] == "NON_STRICT_ARCHIVE_TIMING" for row in out)


def test_resilient_oi_451_uses_archive_fallback_without_zero_imputation():
    source_error = RuntimeError(
        "request failed: https://fapi.binance.com/futures/data/openInterestHist?"
        "symbol=BTCUSDT&period=15m&startTime=1&endTime=2&limit=500: HTTP Error 451:"
    )
    archived = [{"timestamp": 1, "sumOpenInterest": "100.0"}]
    with patch.object(runner, "_ORIGINAL_REQ_JSON", side_effect=source_error),          patch.object(runner, "_archive_oi_metrics_fallback", return_value=archived) as fallback:
        out = runner.resilient_req_json(
            "https://fapi.binance.com/futures/data/openInterestHist?"
            "symbol=BTCUSDT&period=15m&startTime=1&endTime=2&limit=500"
        )
    fallback.assert_called_once()
    assert out == archived
    assert out[0]["sumOpenInterest"] != "0"
