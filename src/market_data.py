"""BTC-only resilient public market-data adapters for GitHub Actions."""
from __future__ import annotations
import asyncio, csv, io, json, time, zipfile, math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError
from datetime import datetime, timezone, timedelta
from binance_ws import capture_closed_klines, contiguous_suffix as ws_contiguous_suffix, load_cache as load_binance_ws_cache

UA = "BTC-Prediction-Research/8.0"
ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "historical_research" / "btc_bootstrap_1m.json"
CACHE_MAX_AGE_MS = 15 * 60 * 1000
BINANCE_WS_CACHE = ROOT / "data" / "binance_ws_1m.json"
BINANCE_WS_MAX_AGE_MS = 180 * 1000


def http_json(url: str, timeout: int = 12):
    req = Request(url, headers={"User-Agent": UA, "Accept": "application/json"})
    with urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def _error_label(exc: Exception) -> str:
    """Return a stable, non-secret diagnostic label for live-source failures."""
    if isinstance(exc, HTTPError):
        return f"HTTPError:{exc.code}"
    if isinstance(exc, URLError):
        reason = getattr(exc, "reason", None)
        return f"URLError:{type(reason).__name__}" if reason is not None else "URLError"
    return type(exc).__name__


def _get(url: str, attempts: int = 3):
    last = None
    for i in range(attempts):
        try:
            return http_json(url)
        except Exception as e:
            last = e
            if i + 1 < attempts:
                time.sleep(min(3.0, 0.8 * (i + 1)))
    raise last


def _binance(path: str, params: dict):
    return _get(f"https://{path}?{urlencode(params)}")


def coinbase_rows(limit: int = 300):
    end = int(time.time())
    start = end - min(limit, 300) * 60
    rows = _get(f"https://api.exchange.coinbase.com/products/BTC-USD/candles?{urlencode({'granularity': 60, 'start': start, 'end': end})}")
    return sorted([[int(r[0]) * 1000, float(r[3]), float(r[2]), float(r[1]), float(r[4]), float(r[5])] for r in rows], key=lambda r: r[0])


def kraken_rows(limit: int = 720):
    since = int(time.time()) - min(limit, 720) * 60
    payload = _get(f"https://api.kraken.com/0/public/OHLC?{urlencode({'pair': 'XBTUSD', 'interval': 1, 'since': since})}")
    result = payload.get("result", {})
    key = next((k for k in result.keys() if k != "last"), None)
    rows = result.get(key, []) if key else []
    return sorted([[int(r[0]) * 1000, float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[6])] for r in rows], key=lambda r: r[0])


def binance_archive_month(month: datetime):
    """Read Binance's static monthly UM-futures 1m archive."""
    name = f"BTCUSDT-1m-{month.year:04d}-{month.month:02d}.zip"
    url = f"https://data.binance.vision/data/futures/um/monthly/klines/BTCUSDT/1m/{name}"
    req = Request(url, headers={"User-Agent": UA})
    with urlopen(req, timeout=45) as r:
        raw = r.read()
    rows = []
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        csv_names = [n for n in z.namelist() if n.lower().endswith('.csv')]
        if not csv_names:
            raise RuntimeError(f"archive contains no csv: {name}")
        with z.open(csv_names[0]) as fh:
            for r in csv.reader(io.TextIOWrapper(fh, encoding='utf-8')):
                if not r or not r[0].isdigit() or len(r) < 6:
                    continue
                try:
                    rows.append([int(r[0]), float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[5])])
                except (TypeError, ValueError):
                    continue
    now_ms = int(time.time() * 1000)
    return [r for r in rows if r[0] + 60000 <= now_ms]


def binance_archive_rows(target: int):
    now = datetime.now(timezone.utc).replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    months = [now]
    prev = (now - timedelta(days=1)).replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    months.append(prev)
    rows = []
    errors = []
    for month in months:
        try:
            rows.extend(binance_archive_month(month))
            if len(rows) >= target:
                break
        except Exception as e:
            errors.append(f"{month.strftime('%Y-%m')}:{type(e).__name__}")
    rows = sorted({r[0]: r for r in rows}.values(), key=lambda r: r[0])
    if len(rows) < target:
        raise RuntimeError(f"archive returned {len(rows)} rows, need {target}; errors={errors}")
    return rows[-target:]


def binance_klines(spot: bool = False, limit: int = 120):
    host = "api.binance.com/api/v3/klines" if spot else "fapi.binance.com/fapi/v1/klines"
    return _binance(host, {"symbol": "BTCUSDT", "interval": "1m", "limit": limit})


def bybit_klines(limit: int = 120):
    return _bybit("kline", {"category": "linear", "symbol": "BTCUSDT", "interval": "1", "limit": min(limit, 1000)})


def _bybit(path: str, params: dict):
    return _get(f"https://api.bybit.com/v5/market/{path}?{urlencode(params)}")


def closed_binance(rows):
    now = int(time.time() * 1000)
    return [r for r in rows if int(r[0]) + 60000 <= now]


def closed_bybit(payload):
    now = int(time.time() * 1000)
    rows = sorted(payload.get("result", {}).get("list", []), key=lambda r: int(r[0]))
    return [[int(r[0]), float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[5])] for r in rows if int(r[0]) + 60000 <= now]


def _latest_contiguous_suffix(rows, minimum: int = 40):
    """Return the newest contiguous 1-minute suffix, or [] if too short."""
    rows = sorted(rows, key=lambda r: int(r[0]))
    if not rows:
        return []
    end = len(rows) - 1
    start = end
    while start > 0 and int(rows[start][0]) - int(rows[start - 1][0]) == 60_000:
        start -= 1
    suffix = rows[start:end + 1]
    return suffix if len(suffix) >= minimum else []


def _latest_row(rows):
    return max(rows, key=lambda r: int(r[0])) if rows else None


def cache_rows(limit=120):
    try:
        obj = json.loads(CACHE.read_text(encoding="utf-8"))
        rows = obj.get("rows", [])
        rows = sorted([[int(r[0]), float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[5])] for r in rows], key=lambda r: r[0])
        now = int(time.time() * 1000)
        rows = [r for r in rows if r[0] + 60000 <= now]
        created_raw = obj.get("created_at_utc", "")
        created_ms = None
        if created_raw:
            created_ms = int(datetime.fromisoformat(created_raw.replace("Z", "+00:00")).timestamp() * 1000)
        age_ms = None if created_ms is None else max(0, now - created_ms)
        fresh = age_ms is not None and age_ms <= CACHE_MAX_AGE_MS
        return _latest_contiguous_suffix(rows, min(limit, len(rows))), created_raw, age_ms, fresh
    except Exception:
        return [], "", None, False


def _ws_series_rows(rows):
    return [
        [
            int(r["open_time_ms"]),
            float(r["open"]),
            float(r["high"]),
            float(r["low"]),
            float(r["close"]),
            float(r["volume"]),
            float(r["taker_buy_base"]),
            int(r["event_time_ms"]),
            int(r["retrieved_at_ms"]),
        ]
        for r in rows
    ]


def _fresh_ws_suffix(rows, minimum: int = 40):
    suffix = ws_contiguous_suffix(rows, minimum=minimum)
    if not suffix:
        return []
    now_ms = int(time.time() * 1000)
    latest = suffix[-1]
    retrieved = int(latest.get("retrieved_at_ms", latest["open_time_ms"]))
    return suffix if now_ms - retrieved <= BINANCE_WS_MAX_AGE_MS else []


def _capture_ws_suffix(existing, timeout_seconds: float = 62.0):
    incoming = []
    try:
        incoming = asyncio.run(capture_closed_klines(timeout_seconds))
    except Exception:
        incoming = []
    if not incoming:
        return _fresh_ws_suffix(existing, 40)
    by_open = {int(row["open_time_ms"]): row for row in existing}
    by_open.update({int(row["open_time_ms"]): row for row in incoming})
    merged = [by_open[key] for key in sorted(by_open)]
    return _fresh_ws_suffix(merged, 40)


def _parallel_result_calls(calls):
    """Run independent public market-data calls concurrently.

    Each callable keeps its own bounded retry/timeout behavior. A failed source
    returns an exception object so the caller can apply the existing fail-closed
    fallback policy without conflating source failures.
    """
    if not calls:
        return {}
    results = {}
    with ThreadPoolExecutor(max_workers=min(3, len(calls))) as pool:
        future_map = {pool.submit(fn): key for key, fn in calls.items()}
        for future, key in ((f, future_map[f]) for f in future_map):
            try:
                results[key] = future.result()
            except Exception as exc:
                results[key] = exc
    return results


def resilient_1m_series(limit: int = 120):
    status = {}

    # A fresh Binance Futures WebSocket cache is a first-class production input.
    # Hosted runners can be geo-restricted from Binance REST, so do not spend
    # the live-cycle latency budget on a REST request when an independently
    # validated WS checkpoint already satisfies the same 40-bar PIT boundary.
    ws_cache = load_binance_ws_cache(BINANCE_WS_CACHE, max(120, limit))
    ws_suffix_primary = _fresh_ws_suffix(ws_cache, 40)

    parallel_calls = {
        "bybit": lambda: closed_bybit(bybit_klines(limit)),
        "binance_spot": lambda: closed_binance(binance_klines(True, limit)),
    }
    if not ws_suffix_primary:
        parallel_calls["binance_futures"] = lambda: closed_binance(binance_klines(False, limit))

    parallel = _parallel_result_calls(parallel_calls)

    by = []
    by_current = None
    try:
        payload = parallel["bybit"]
        if isinstance(payload, Exception):
            raise payload
        by_raw = payload
        by = _latest_contiguous_suffix(by_raw, 40)
        by_current = _latest_row(by_raw)
        # If kline data is empty/fragmented, obtain the current linear ticker once.
        if not by_current:
            try:
                ticker = bybit_mark_price()
                ticker_rows = ticker.get("result", {}).get("list", []) if isinstance(ticker, dict) else []
                if ticker_rows:
                    row = ticker_rows[0]
                    px = float(row.get("lastPrice") or row.get("markPrice"))
                    if math.isfinite(px) and px > 0:
                        by_current = [int(time.time() * 1000), px, px, px, px, 0.0]
            except Exception:
                by_current = None
        status["bybit_futures"] = "ok" if by else ("ok_current_only" if by_current else "non_contiguous_or_insufficient")
        if not by and by_current:
            by = [by_current]
    except Exception as e:
        by = []
        by_current = None
        try:
            ticker = bybit_mark_price()
            rows = ticker.get("result", {}).get("list", []) if isinstance(ticker, dict) else []
            if rows:
                row = rows[0]
                px = float(row.get("lastPrice") or row.get("markPrice"))
                if px > 0:
                    by_current = [int(time.time() * 1000), px, px, px, px, 0.0]
        except Exception:
            by_current = None
        status["bybit_futures"] = "ok_current_only" if by_current else f"error:{_error_label(e)}"

    if ws_suffix_primary:
        fut = _ws_series_rows(ws_suffix_primary[-limit:])
        status["binance_futures"] = "ok"
        status["binance_futures_transport"] = "websocket"
        status["binance_futures_ws_event_time_ms"] = int(ws_suffix_primary[-1]["event_time_ms"])
        status["binance_futures_ws_retrieved_at_ms"] = int(ws_suffix_primary[-1]["retrieved_at_ms"])
    else:
        fut = []
        fut_error = parallel.get("binance_futures")
        if isinstance(fut_error, Exception):
            status["binance_futures"] = f"error:{_error_label(fut_error)}"
        else:
            fut = _latest_contiguous_suffix(fut_error, 40)
            status["binance_futures"] = "ok" if fut else "non_contiguous_or_insufficient"

    spot = []
    spot_error = parallel.get("binance_spot")
    if isinstance(spot_error, Exception):
        status["binance_spot"] = f"error:{_error_label(spot_error)}"
    else:
        spot = _latest_contiguous_suffix(spot_error, 40)
        status["binance_spot"] = "ok" if spot else "non_contiguous_or_insufficient"

    # When Binance REST is unavailable, recover the same Binance Futures product
    # through its public WebSocket market stream.
    if len(fut) < 40:
        ws_cache = load_binance_ws_cache(BINANCE_WS_CACHE, max(120, limit))
        ws_suffix = _fresh_ws_suffix(ws_cache, 40)
        if not ws_suffix:
            ws_suffix = _capture_ws_suffix(ws_cache, 62.0)
        if len(ws_suffix) >= 40:
            fut = _ws_series_rows(ws_suffix[-limit:])
            status["binance_futures"] = "ok"
            status["binance_futures_transport"] = "websocket"
            status["binance_futures_ws_event_time_ms"] = int(ws_suffix[-1]["event_time_ms"])
            status["binance_futures_ws_retrieved_at_ms"] = int(ws_suffix[-1]["retrieved_at_ms"])
        else:
            status["binance_futures_ws_contiguous_rows"] = len(ws_suffix)

    if len(fut) >= 40 and status.get("binance_futures_transport") == "websocket":
        status["price_feature_fallback"] = "none"
    elif len(fut) < 40 and len(by) >= 40:
        fut = by
        status["price_feature_fallback"] = "bybit"
    elif len(fut) < 40:
        try:
            cb = _latest_contiguous_suffix(coinbase_rows(min(300, max(120, limit))), 40)
            if len(cb) >= 40:
                fut = cb
                status["price_feature_fallback"] = "coinbase"
            else:
                raise RuntimeError("insufficient contiguous Coinbase candles")
        except Exception as e:
            status["coinbase_futures"] = f"error:{_error_label(e)}"
            try:
                kr = _latest_contiguous_suffix(kraken_rows(max(120, limit)), 40)
                if len(kr) >= 40:
                    fut = kr
                    status["price_feature_fallback"] = "kraken"
                else:
                    raise RuntimeError("insufficient contiguous Kraken candles")
            except Exception as e2:
                status["kraken_futures"] = f"error:{_error_label(e2)}"
                cached, created, age_ms, fresh = cache_rows(limit)
                status["cache_created_at_utc"] = created
                status["cache_age_seconds"] = None if age_ms is None else int(age_ms / 1000)
                if len(cached) >= 40 and fresh:
                    fut = cached
                    status["price_feature_fallback"] = "fresh_bootstrap_cache"
                elif len(cached) >= 40:
                    status["price_feature_fallback"] = "stale_cache_rejected"
                else:
                    status["price_feature_fallback"] = "none"
    else:
        status["price_feature_fallback"] = "none"
    status["spot_fallback"] = "unavailable" if len(spot) < 40 else "none"
    if not by and by_current:
        by = [by_current]
    status["bybit_series_available"] = len(by) >= 40
    status["bybit_current_price_available"] = by_current is not None
    status["live_series_fresh"] = status["price_feature_fallback"] in {"none", "bybit", "coinbase", "kraken", "fresh_bootstrap_cache"}
    return fut, spot, by, status


def binance_depth():
    return _binance("fapi.binance.com/fapi/v1/depth", {"symbol": "BTCUSDT", "limit": 50})


def bybit_depth():
    return _bybit("orderbook", {"category": "linear", "symbol": "BTCUSDT", "limit": 50})


def binance_premium():
    return _binance("fapi.binance.com/fapi/v1/premiumIndex", {"symbol": "BTCUSDT"})


def binance_oi():
    return _binance("fapi.binance.com/fapi/v1/openInterest", {"symbol": "BTCUSDT"})


def binance_taker():
    # Taker-flow is a Binance Futures endpoint; keep the fapi host explicit.
    return _binance("fapi.binance.com/futures/data/takerBuySellVol", {"symbol": "BTCUSDT", "period": "5m", "limit": 1})


def bybit_funding():
    return _bybit("funding/history", {"category": "linear", "symbol": "BTCUSDT", "limit": 1})


def bybit_mark_price():
    return _bybit("tickers", {"category": "linear", "symbol": "BTCUSDT"})


def target_close_binance(target_iso: str):
    target = datetime.fromisoformat(target_iso.replace("Z", "+00:00"))
    ts = int(target.timestamp() * 1000)
    start = ts - 60000
    try:
        rows = _binance("fapi.binance.com/fapi/v1/klines", {"symbol": "BTCUSDT", "interval": "1m", "startTime": start, "endTime": ts, "limit": 2})
        for row in rows:
            if int(row[0]) == start:
                return float(row[4]), "binance"
    except Exception:
        pass
    try:
        payload = _bybit("kline", {"category": "linear", "symbol": "BTCUSDT", "interval": "1", "start": start, "end": ts, "limit": 2})
        for row in payload.get("result", {}).get("list", []):
            if int(row[0]) == start:
                return float(row[4]), "bybit"
    except Exception:
        pass
    try:
        for row in coinbase_rows(10):
            if row[0] == start:
                return float(row[4]), "coinbase"
    except Exception:
        pass
    try:
        for row in kraken_rows(10):
            if row[0] == start:
                return float(row[4]), "kraken"
    except Exception:
        pass
    return None, "unavailable"
