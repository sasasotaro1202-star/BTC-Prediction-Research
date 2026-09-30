"""Reliable free BTC 1-minute historical archive loader.

Monthly Binance Vision archives are tried first. If they are unavailable or do
not contain enough closed candles, the loader walks backward over completed UTC
days using the public daily archive and S3 mirror. Failed days are recorded,
not fatal. Returned rows are deduplicated, sorted, and guaranteed closed.
"""
from __future__ import annotations

import csv
import io
import json
import time
import urllib.request
from urllib.parse import urlencode
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / "data" / "historical_research" / "archive_cache"
UA = "BTC-Prediction-Research/archive/1.0"
REST_HOSTS = ("fapi.binance.com", "fapi1.binance.com", "fapi2.binance.com", "fapi3.binance.com", "fapi4.binance.com")
REST_BATCH = 1000
REST_MAX_PAGES = 48


def _download(url: str, timeout: int = 45) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    last = None
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except Exception as exc:
            last = exc
            if attempt < 3:
                time.sleep(min(6.0, 0.8 * (attempt + 1)))
    raise last


def _rows_from_zip(raw: bytes, label: str):
    rows = []
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        names = [n for n in z.namelist() if n.lower().endswith(".csv")]
        if not names:
            raise RuntimeError(f"{label}: archive contains no CSV")
        with z.open(names[0]) as fh:
            for row in csv.reader(io.TextIOWrapper(fh, encoding="utf-8")):
                if len(row) < 6 or not row[0].strip().isdigit():
                    continue
                try:
                    rows.append([int(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[4]), float(row[5])])
                except (TypeError, ValueError):
                    continue
    return rows


def _closed(rows):
    now_ms = int(time.time() * 1000)
    return [r for r in rows if int(r[0]) + 60_000 <= now_ms]


def _month_url(dt: datetime) -> str:
    name = f"BTCUSDT-1m-{dt.year:04d}-{dt.month:02d}.zip"
    return f"https://data.binance.vision/data/futures/um/monthly/klines/BTCUSDT/1m/{name}"


def _day_urls(dt: datetime):
    name = f"BTCUSDT-1m-{dt:%Y-%m-%d}.zip"
    path = f"data/futures/um/daily/klines/BTCUSDT/1m/{name}"
    return [f"https://data.binance.vision/{path}", f"https://s3-ap-northeast-1.amazonaws.com/data.binance.vision/{path}"]


def _contiguous_suffix(rows, interval_ms: int = 60_000):
    """Return only the newest strictly contiguous candle suffix."""
    ordered = sorted({int(r[0]): r for r in rows}.values(), key=lambda r: int(r[0]))
    if not ordered:
        return []
    start = len(ordered) - 1
    while start > 0 and int(ordered[start][0]) - int(ordered[start - 1][0]) == int(interval_ms):
        start -= 1
    return ordered[start:]


def _cached_day(day: datetime):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"BTCUSDT-1m-{day:%Y-%m-%d}.json"


def _load_day(day: datetime):
    path = _cached_day(day)
    if path.is_file():
        try:
            obj = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(obj, list):
                return _closed(obj), "cache"
        except Exception:
            pass
    errors = []
    for url in _day_urls(day):
        try:
            rows = _closed(_rows_from_zip(_download(url), day.strftime("%Y-%m-%d")))
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(rows, separators=(",", ":")), encoding="utf-8")
            tmp.replace(path)
            return rows, url
        except Exception as exc:
            errors.append(f"{type(exc).__name__}:{exc}")
    raise RuntimeError(" | ".join(errors))


def _month_rows(month: datetime):
    return _closed(_rows_from_zip(_download(_month_url(month)), month.strftime("%Y-%m")))


def _rest_klines(target: int):
    """Recover recent closed BTCUSDT 1m candles from public Binance Futures REST."""
    target = int(target)
    if target <= 0:
        return []
    rows = []
    end_ms = int(time.time() * 1000) - 60_000
    errors = []
    for _ in range(REST_MAX_PAGES):
        page = None
        for host in REST_HOSTS:
            url = "https://" + host + "/fapi/v1/klines?" + urlencode({
                "symbol": "BTCUSDT",
                "interval": "1m",
                "limit": REST_BATCH,
                "endTime": end_ms,
            })
            try:
                page = json.loads(_download(url, timeout=30).decode("utf-8"))
                if isinstance(page, list):
                    break
                errors.append(host + ":non_list_response")
            except Exception as exc:
                errors.append(host + ":" + type(exc).__name__)
                page = None
        if not isinstance(page, list) or not page:
            break
        parsed = []
        for item in page:
            if not isinstance(item, list) or len(item) < 11:
                continue
            try:
                parsed.append([
                    int(item[0]),
                    float(item[1]),
                    float(item[2]),
                    float(item[3]),
                    float(item[4]),
                    float(item[5]),
                ])
            except (TypeError, ValueError):
                continue
        closed = _closed(parsed)
        rows.extend(closed)
        rows = list({int(row[0]): row for row in rows}.values())
        contiguous = _contiguous_suffix(rows)
        if len(contiguous) >= target:
            return contiguous[-target:]
        earliest = min((int(row[0]) for row in parsed), default=None)
        if earliest is None:
            break
        end_ms = earliest - 1
        if len(page) < REST_BATCH:
            break
    if len(_contiguous_suffix(rows)) >= target:
        return _contiguous_suffix(rows)[-target:]
    if errors:
        return []
    return []


def binance_archive_rows(target: int = 30_000):
    """Return target recent closed BTCUSDT 1m candles from free archives.
    
    Freshness-first policy:
    1. Use the current-month monthly archive when available.
    2. If it is unavailable, consume recent completed daily archives first.
    3. Fill any remaining history from prior completed monthly archives.
    4. Preserve closed-candle, dedupe, and contiguous-suffix guarantees.
    """
    target = int(target)
    if target <= 0:
        return []

    rows, errors = [], []
    now = datetime.now(timezone.utc)
    month = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

    # Current month is special: an unpublished/stale monthly archive must not
    # cause a large older month to satisfy the row-count target first.
    current_month_ok = False
    try:
        current_rows = _month_rows(month)
        if current_rows:
            current_month_ok = True
            rows.extend(current_rows)
            rows = list({int(r[0]): r for r in rows}.values())
    except Exception as exc:
        errors.append(f"monthly:{month:%Y-%m}:{type(exc).__name__}:{exc}")

    if not current_month_ok and len(rows) < target:
        # Freshness-first recovery: build the newest available closed suffix
        # from daily archives before considering older monthly history.
        day = (now - timedelta(days=1)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        for _ in range(45):
            try:
                day_rows, _ = _load_day(day)
                rows.extend(day_rows)
            except Exception as exc:
                errors.append(
                    f"daily:{day:%Y-%m-%d}:{type(exc).__name__}:{exc}"
                )
            rows = list({int(r[0]): r for r in rows}.values())
            contiguous = _contiguous_suffix(rows)
            if len(contiguous) >= target:
                return contiguous[-target:]
            day -= timedelta(days=1)

    # Fill the remainder from completed prior monthly archives. Current month
    # is excluded because it was already attempted above.
    previous_month = (month - timedelta(days=1)).replace(
        day=1, hour=0, minute=0, second=0, microsecond=0
    )
    m = previous_month
    for _ in range(6):
        try:
            rows.extend(_month_rows(m))
            rows = list({int(r[0]): r for r in rows}.values())
        except Exception as exc:
            errors.append(f"monthly:{m:%Y-%m}:{type(exc).__name__}:{exc}")
        contiguous = _contiguous_suffix(rows)
        if len(contiguous) >= target:
            return contiguous[-target:]
        m = (m - timedelta(days=1)).replace(day=1)

    # Final daily recovery is useful when the current monthly archive exists but
    # completed monthly history is still insufficient.
    if len(rows) < target:
        day = (now - timedelta(days=1)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        for _ in range(45):
            try:
                day_rows, _ = _load_day(day)
                rows.extend(day_rows)
            except Exception as exc:
                errors.append(
                    f"daily:{day:%Y-%m-%d}:{type(exc).__name__}:{exc}"
                )
            rows = list({int(r[0]): r for r in rows}.values())
            contiguous = _contiguous_suffix(rows)
            if len(contiguous) >= target:
                return contiguous[-target:]
            day -= timedelta(days=1)

    rows = _closed(rows)
    contiguous = _contiguous_suffix(rows)
    if len(contiguous) < target:
        rest_rows = _rest_klines(target)
        rest_contiguous = _contiguous_suffix(rest_rows)
        if len(rest_contiguous) >= target:
            return rest_contiguous[-target:]
    if len(contiguous) < target:
        raise RuntimeError(
            f"archive returned only {len(contiguous)} contiguous closed rows; "
            f"need {target}; errors={errors[-10:]}"
        )
    return contiguous[-target:]


