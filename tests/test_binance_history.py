from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import pytest

from src import binance_history as bh


def _zip_bytes(rows):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        text = "\n".join(",".join(map(str, row)) for row in rows) + "\n"
        z.writestr("BTCUSDT-1m-test.csv", text)
    return buf.getvalue()


def test_rows_from_zip_ignores_headers_and_malformed_rows():
    raw = _zip_bytes([
        ["open_time", "open", "high", "low", "close", "volume"],
        [1000, 10, 11, 9, 10.5, 2],
        ["bad", "x", "x", "x", "x", "x"],
        [2000, 20, 21, 19, 20.5, 3],
    ])
    assert bh._rows_from_zip(raw, "unit") == [
        [1000, 10.0, 11.0, 9.0, 10.5, 2.0],
        [2000, 20.0, 21.0, 19.0, 20.5, 3.0],
    ]


def test_load_day_uses_cache_and_rejects_future_candles(tmp_path, monkeypatch):
    monkeypatch.setattr(bh, "CACHE_DIR", Path(tmp_path))
    # Keep one candle safely closed and one candle still open relative to the
    # mocked clock.  Cache loading must re-apply the closed-candle filter.
    monkeypatch.setattr(bh.time, "time", lambda: 2_000_000 / 1000)
    path = bh._cached_day(__import__("datetime").datetime(1970, 1, 1))
    path.write_text(json.dumps([
        [1_900_000, 1, 1, 1, 1, 1],
        [1_950_000, 1, 1, 1, 1, 1],
    ]), encoding="utf-8")

    rows, source = bh._load_day(__import__("datetime").datetime(1970, 1, 1))
    assert source == "cache"
    assert [r[0] for r in rows] == [1_900_000]


def test_archive_rows_deduplicates_sorts_and_returns_recent_target(monkeypatch):
    monkeypatch.setattr(
        bh,
        "_month_rows",
        lambda month: [
            [300, 3, 3, 3, 3, 3],
            [100, 1, 1, 1, 1, 1],
            [200, 2, 2, 2, 2, 2],
            [200, 22, 22, 22, 22, 22],
        ],
    )
    # Two month attempts would normally overlap; the function must deduplicate.
    rows = bh.binance_archive_rows(target=3)
    assert [r[0] for r in rows] == [100, 200, 300]
    assert rows[-1][1] == 3.0


def test_archive_rows_fails_closed_when_not_enough_history(monkeypatch):
    monkeypatch.setattr(bh, "_month_rows", lambda month: [])
    monkeypatch.setattr(bh, "_load_day", lambda day: ([], "mock"))
    with pytest.raises(RuntimeError, match="need 5"):
        bh.binance_archive_rows(target=5)
