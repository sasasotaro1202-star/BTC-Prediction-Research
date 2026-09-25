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


BINANCE_FUTURES_REST_HOSTS = (
    "fapi.binance.com",
    "fapi1.binance.com",
    "fapi2.binance.com",
    "fapi3.binance.com",
    "fapi4.binance.com",
)


def _binance(path: str, params: dict):
    return _get(f"https://{path}?{urlencode(params)}")


def _binance_futures(path: str, params: dict):
    """Query Binance USD-M Futures with same-product REST host failover.
    
    The canonical fapi host is tried first; mirror hosts are only transport
    failover and do not change the underlying Binance Futures product.
    """
    last = None
    for host in BINANCE_FUTURES_REST_HOSTS:
        try:
            return _get(f"https://{host}/{path}?{urlencode(params)}")
        except Exception as exc:
            last = exc
    if last is not None:
        raise last
    raise RuntimeError("binance_futures_rest_hosts_empty")


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




def _archive_daily_urls(day: datetime) -> list[str]:
    stamp = day.strftime("%Y-%m-%d")
    name = f"BTCUSDT-1m-{stamp}.zip"
    rel = f"data/futures/um/daily/klines/BTCUSDT/1m/{name}"
    return [
        f"https://data.binance.vision/{rel}",
        f"https://s3-ap-northeast-1.amazonaws.com/data.binance.vision/{rel}",
    ]


def binance_archive_daily_rows(target: int = 120) -> list[list[float]]:
    """Load the newest closed BTCUSDT 1m Futures candles from Binance Vision daily files.

    This is the same Binance USD-M product as the Futures REST endpoint. Because
    the archive is retrieved during the current prediction cycle, its
    retrieval/availability timestamp is conservatively treated as the current
    cutoff; candle event times remain the exchange candle open times.
    """
    target = max(40, min(int(target), 1500))
    now = datetime.now(timezone.utc)
    days = [now, now - timedelta(days=1)]
    rows: list[list[float]] = []
    errors: list[str] = []

    for day in days:
        for url in _archive_daily_urls(day):
            try:
                req = Request(url, headers={"User-Agent": UA})
                with urlopen(req, timeout=20) as response:
                    payload = response.read()
                with zipfile.ZipFile(io.BytesIO(payload)) as zf:
                    if zf.testzip() is not None:
                        raise RuntimeError("archive_zip_crc_failed")
                    names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
                    if not names:
                        raise RuntimeError("archive_contains_no_csv")
                    with zf.open(names[0]) as fh:
                        reader = csv.reader(io.TextIOWrapper(fh, encoding="utf-8", newline=""))
                        for raw in reader:
                            if len(raw) < 6:
                                continue
                            try:
                                open_ms = int(raw[0])
                                o = float(raw[1])
                                h = float(raw[2])
                                low = float(raw[3])
                                c = float(raw[4])
                                volume = float(raw[5])
                            except (TypeError, ValueError):
                                continue
                            if open_ms <= 0 or min(o, h, low, c) <= 0 or volume < 0:
                                continue
                            if h < max(o, c) or low > min(o, c):
                                continue
                            if open_ms + 60_000 > int(now.timestamp() * 1000):
                                continue
                            rows.append([open_ms, o, h, low, c, volume])
                if len(rows) >= target:
                    break
            except Exception as exc:
                errors.append(f"{day.strftime('%Y-%m-%d')}:{type(exc).__name__}")
        if len(rows) >= target:
            break

    dedup = {int(r[0]): r for r in rows}
    ordered = [dedup[k] for k in sorted(dedup)]
    suffix = _latest_contiguous_suffix(ordered, target)
    if not suffix:
        raise RuntimeError(
            f"Binance Vision daily archive returned insufficient contiguous rows: "
            f"{len(ordered)} need {target}; errors={errors}"
        )
    return suffix[-target:]

def binance_archive_daily_taker_rows(target: int = 5) -> list[dict]:
    """Load recent closed BTCUSDT 1m Futures taker-buy volumes from Binance Vision.

    The daily archive contains the same kline taker-buy-base field used by the
    Futures kline API. The archive retrieval timestamp is used as the conservative
    availability boundary; closed/future rows and gaps are rejected.
    """
    target = max(5, min(int(target), 120))
    now = datetime.now(timezone.utc)
    now_ms = int(now.timestamp() * 1000)
    days = [now, now - timedelta(days=1)]
    rows: list[dict] = []
    errors: list[str] = []

    for day in days:
        for url in _archive_daily_urls(day):
            try:
                req = Request(url, headers={"User-Agent": UA})
                with urlopen(req, timeout=20) as response:
                    payload = response.read()
                with zipfile.ZipFile(io.BytesIO(payload)) as zf:
                    if zf.testzip() is not None:
                        raise RuntimeError("archive_zip_crc_failed")
                    names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
                    if not names:
                        raise RuntimeError("archive_contains_no_csv")
                    with zf.open(names[0]) as fh:
                        reader = csv.reader(io.TextIOWrapper(fh, encoding="utf-8", newline=""))
                        for raw in reader:
                            if len(raw) < 10:
                                continue
                            try:
                                open_ms = int(raw[0])
                                volume = float(raw[5])
                                close_ms = int(raw[6])
                                taker_buy = float(raw[9])
                            except (TypeError, ValueError, IndexError):
                                continue
                            if (
                                open_ms <= 0
                                or close_ms <= 0
                                or volume <= 0
                                or taker_buy < 0
                                or taker_buy > volume + 1e-9
                                or open_ms + 60_000 > now_ms
                                or close_ms > now_ms
                            ):
                                continue
                            rows.append(
                                {
                                    "open_time_ms": open_ms,
                                    "volume": volume,
                                    "taker_buy_base": taker_buy,
                                    "event_time_ms": close_ms,
                                    "retrieved_at_ms": now_ms,
                                }
                            )
                if len(rows) >= target:
                    break
            except Exception as exc:
                errors.append(f"{day.strftime('%Y-%m-%d')}:{type(exc).__name__}")
        if len(rows) >= target:
            break

    dedup = {int(r["open_time_ms"]): r for r in rows}
    ordered = [dedup[k] for k in sorted(dedup)]
    if len(ordered) < target:
        raise RuntimeError(
            f"Binance Vision daily archive returned insufficient taker rows: "
            f"{len(ordered)} need {target}; errors={errors}"
        )
    suffix = ordered[-target:]
    for left, right in zip(suffix, suffix[1:]):
        if int(right["open_time_ms"]) - int(left["open_time_ms"]) != 60_000:
            raise RuntimeError("Binance Vision daily archive taker rows are non-contiguous")
    return suffix


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
    if spot:
        return _binance("api.binance.com/api/v3/klines", {"symbol": "BTCUSDT", "interval": "1m", "limit": limit})
    return _binance_futures("fapi/v1/klines", {"symbol": "BTCUSDT", "interval": "1m", "limit": limit})


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


def _fresh_closed_candle_suffix(rows, minimum: int = 40, max_age_ms: int = BINANCE_WS_MAX_AGE_MS):
    """Return contiguous closed candles only when the latest close boundary is fresh."""
    suffix = _latest_contiguous_suffix(rows, minimum)
    if not suffix:
        return []
    try:
        now_ms = int(time.time() * 1000)
        latest_open_ms = int(suffix[-1][0])
    except (TypeError, ValueError, IndexError):
        return []
    latest_event_ms = latest_open_ms + 60_000 - 1
    age = now_ms - latest_event_ms
    if age < -60_000 or age > int(max_age_ms):
        return []
    return suffix


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
    """Return a contiguous WS suffix only when retrieval and exchange event times are fresh."""
    suffix = ws_contiguous_suffix(rows, minimum=minimum)
    if not suffix:
        return []
    now_ms = int(time.time() * 1000)
    latest = suffix[-1]
    retrieved = int(latest.get("retrieved_at_ms", latest["open_time_ms"]))
    event_time = int(latest.get("event_time_ms", latest["open_time_ms"]))
    retrieved_age = now_ms - retrieved
    event_age = now_ms - event_time
    # A post-retrieval delay must not make an old market event look fresh.
    # Reject future timestamps conservatively as well.
    if retrieved_age < 0 or retrieved_age > BINANCE_WS_MAX_AGE_MS:
        return []
    if event_age < -60_000 or event_age > BINANCE_WS_MAX_AGE_MS:
        return []
    return suffix
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

    # Prefer a fresh Binance Futures WS cache when one is available. This keeps
    # the production price/target venue independent of REST reachability and
    # avoids spending the live-cycle budget on a known-good market-data source.
    ws_cache = load_binance_ws_cache(BINANCE_WS_CACHE, max(120, limit))
    ws_suffix = _fresh_ws_suffix(ws_cache, 40)
    fut = _ws_series_rows(ws_suffix[-limit:]) if len(ws_suffix) >= 40 else []
    if fut:
        status["binance_futures"] = "ok"
        status["binance_futures_transport"] = "websocket"
        status["binance_futures_ws_event_time_ms"] = int(ws_suffix[-1]["event_time_ms"])
        status["binance_futures_ws_retrieved_at_ms"] = int(ws_suffix[-1]["retrieved_at_ms"])
    else:
        status["binance_futures"] = "not_loaded_from_websocket"

    parallel_calls = {
        "bybit": lambda: closed_bybit(bybit_klines(limit)),
        "binance_spot": lambda: closed_binance(binance_klines(True, limit)),
    }
    if not fut:
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

    if not fut:
        fut_error = parallel.get("binance_futures")
        if isinstance(fut_error, Exception):
            status["binance_futures"] = f"error:{_error_label(fut_error)}"
        elif fut_error is not None:
            raw_fut = _latest_contiguous_suffix(fut_error, 40)
            fut = _fresh_closed_candle_suffix(raw_fut, 40)
            status["binance_futures"] = "ok" if fut else "stale_or_insufficient"
            status["binance_futures_transport"] = "rest"

    spot = []
    spot_error = parallel.get("binance_spot")
    if isinstance(spot_error, Exception):
        status["binance_spot"] = f"error:{_error_label(spot_error)}"
    else:
        spot = _latest_contiguous_suffix(spot_error, 40)
        status["binance_spot"] = "ok" if spot else "non_contiguous_or_insufficient"

    # When neither the fresh cache nor Binance REST provides a usable Futures
    # history, recover the same Binance product through a bounded WebSocket capture.
    if len(fut) < 40:
        live_ws = load_binance_ws_cache(BINANCE_WS_CACHE, max(120, limit))
        live_ws_suffix = _fresh_ws_suffix(live_ws, 40)
        if not live_ws_suffix:
            live_ws_suffix = _capture_ws_suffix(live_ws, 62.0)
        if len(live_ws_suffix) >= 40:
            fut = _ws_series_rows(live_ws_suffix[-limit:])
            status["binance_futures"] = "ok"
            status["binance_futures_transport"] = "websocket"
            status["binance_futures_ws_event_time_ms"] = int(live_ws_suffix[-1]["event_time_ms"])
            status["binance_futures_ws_retrieved_at_ms"] = int(live_ws_suffix[-1]["retrieved_at_ms"])
        else:
            status["binance_futures_ws_contiguous_rows"] = len(live_ws_suffix)

    # Binance Vision is a same-product, closed-candle archive. It is used only
    # to reconstruct historical Futures candles after the live REST/WS paths fail.
    # Each archive observation is conservatively treated as available at the
    # current acquisition cutoff; no post-cutoff candle is accepted.
    if len(fut) < 40:
        try:
            archive_rows = binance_archive_daily_rows(max(120, limit))
            fresh_archive = _fresh_closed_candle_suffix(archive_rows, 40)
            if len(fresh_archive) >= 40:
                fut = fresh_archive[-limit:]
                status["binance_futures"] = "ok"
                status["binance_futures_transport"] = "binance_vision_daily_archive"
                status["binance_futures_archive_retrieved_at_ms"] = int(
                    time.time() * 1000
                )
            else:
                status["binance_futures"] = "stale_or_insufficient_archive"
        except Exception as exc:
            status["binance_futures_archive"] = f"error:{_error_label(exc)}"

    if len(fut) >= 40 and status.get("binance_futures_transport") == "websocket":
        status["price_feature_fallback"] = "none"
    elif len(fut) < 40 and len(by) >= 40:
        fresh_by = _fresh_closed_candle_suffix(by, 40)
        if len(fresh_by) >= 40:
            fut = fresh_by
            status["price_feature_fallback"] = "bybit"
        else:
            status["price_feature_fallback"] = "none"
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
    return _binance_futures("fapi/v1/depth", {"symbol": "BTCUSDT", "limit": 50})


def bybit_depth():
    return _bybit("orderbook", {"category": "linear", "symbol": "BTCUSDT", "limit": 50})


def binance_premium():
    return _binance_futures("fapi/v1/premiumIndex", {"symbol": "BTCUSDT"})


def binance_oi():
    return _binance_futures("fapi/v1/openInterest", {"symbol": "BTCUSDT"})


def derive_binance_taker_from_closed_klines(
    rows,
    window: int = 5,
    cutoff_ms: int | None = None,
):
    """Derive taker imbalance from already-fetched closed Binance Futures klines.

    Supports normalized Binance WebSocket rows and raw Binance Futures REST
    kline rows. Only closed, contiguous bars available at the prediction cutoff
    are used. Invalid, future, unavailable, or gapped data fails closed.
    """
    try:
        window = int(window)
    except (TypeError, ValueError):
        return None
    if window <= 0:
        return None

    now_ms = int(time.time() * 1000)
    cutoff = now_ms if cutoff_ms is None else int(cutoff_ms)
    normalized = []

    for row in rows or []:
        try:
            if isinstance(row, dict):
                open_ms = int(row["open_time_ms"])
                volume = float(row["volume"])
                taker_buy = float(row["taker_buy_base"])
                event_ms = int(row.get("event_time_ms", open_ms + 59_999))
                retrieved_ms = int(row.get("retrieved_at_ms", now_ms))
            else:
                values = list(row)
                if len(values) >= 10:
                    open_ms = int(values[0])
                    close_ms = int(values[6])
                    volume = float(values[5])
                    taker_buy = float(values[9])
                    event_ms = close_ms
                    retrieved_ms = now_ms
                elif len(values) >= 9:
                    open_ms = int(values[0])
                    volume = float(values[5])
                    taker_buy = float(values[6])
                    event_ms = int(values[7])
                    retrieved_ms = int(values[8])
                else:
                    continue
        except (KeyError, TypeError, ValueError, IndexError):
            continue

        if (
            open_ms <= 0
            or volume <= 0
            or taker_buy < 0
            or taker_buy > volume + 1e-9
            or event_ms <= 0
            or retrieved_ms <= 0
            or open_ms + 60_000 > cutoff
            or event_ms > cutoff
            or retrieved_ms > cutoff
        ):
            continue

        normalized.append(
            {
                "open_time_ms": open_ms,
                "volume": volume,
                "taker_buy_base": taker_buy,
                "event_time_ms": event_ms,
                "retrieved_at_ms": retrieved_ms,
            }
        )

    normalized.sort(key=lambda r: r["open_time_ms"])
    if len(normalized) < window:
        return None

    suffix = normalized[-window:]
    for left, right in zip(suffix, suffix[1:]):
        if right["open_time_ms"] - left["open_time_ms"] != 60_000:
            return None

    total_volume = sum(r["volume"] for r in suffix)
    total_taker_buy = sum(r["taker_buy_base"] for r in suffix)
    if not math.isfinite(total_volume) or total_volume <= 0:
        return None

    imbalance = 2.0 * total_taker_buy / total_volume - 1.0
    if not math.isfinite(imbalance):
        return None

    return {
        "taker_imbalance": float(imbalance),
        "event_time_ms": max(r["event_time_ms"] for r in suffix),
        "retrieved_at_ms": max(r["retrieved_at_ms"] for r in suffix),
        "rows": window,
        "source": "binance_futures_closed_klines",
    }


def binance_taker():
    return _binance_futures("futures/data/takerBuySellVol", {"symbol": "BTCUSDT", "period": "5m", "limit": 1})


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
