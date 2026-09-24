import csv
import io
import zipfile
from datetime import datetime, timezone
from unittest.mock import patch

from src import market_data


def _zip(rows):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        payload = io.StringIO()
        writer = csv.writer(payload)
        for row in rows:
            writer.writerow(row)
        zf.writestr("BTCUSDT-1m-test.csv", payload.getvalue())
    return buf.getvalue()


def _rows(start_ms, count):
    return [
        [start_ms + i * 60_000, "100.0", "101.0", "99.0", "100.5", "10.0"]
        for i in range(count)
    ]


def test_daily_archive_returns_closed_contiguous_rows():
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    start = now_ms - 150 * 60_000
    payload = _zip(_rows(start, 150))

    class Response:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return None
        def read(self):
            return payload

    with patch.object(market_data, "urlopen", return_value=Response()):
        rows = market_data.binance_archive_daily_rows(120)

    assert len(rows) == 120
    assert all(rows[i][0] - rows[i - 1][0] == 60_000 for i in range(1, len(rows)))
    assert rows[-1][0] + 60_000 <= now_ms


def test_daily_archive_rejects_crc_corruption_and_fails_closed():
    class Response:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return None
        def read(self):
            return b"not a zip"

    with patch.object(market_data, "urlopen", return_value=Response()):
        try:
            market_data.binance_archive_daily_rows(120)
        except RuntimeError as exc:
            assert "insufficient contiguous rows" in str(exc)
        else:
            raise AssertionError("corrupt archive must fail closed")


def test_daily_archive_never_accepts_current_open_candle():
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    start = now_ms - 20 * 60_000
    rows = _rows(start, 21)
    rows[-1][0] = now_ms
    payload = _zip(rows)

    class Response:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return None
        def read(self):
            return payload

    with patch.object(market_data, "urlopen", return_value=Response()):
        try:
            market_data.binance_archive_daily_rows(40)
        except RuntimeError:
            pass
        else:
            raise AssertionError("insufficient historical rows must defer")
