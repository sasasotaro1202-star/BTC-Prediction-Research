"""Free Binance USD-M Futures WebSocket helpers with strict closed-bar handling.

This module is deliberately transport-only. It never invents missing observations:
- kline rows are accepted only when Binance marks them closed;
- every row keeps the exchange event time and local receipt time;
- depth is a bounded partial-book snapshot (20 levels);
- mark price keeps Binance's event time and reported funding rate;
- persisted cache contains only Binance WebSocket observations.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import websockets

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE = ROOT / "data" / "binance_ws_1m.json"
DEFAULT_DEPTH_CACHE = ROOT / "data" / "binance_ws_depth.json"
# Binance USDⓈ-M Futures migrated public market streams to category-specific
# WebSocket bases in 2026. Legacy /ws and /stream URLs were retired on
# 2026-04-23, so production transport starts directly on the current endpoint
# and only falls back to the current combined-stream form.
KLINE_URL = "wss://fstream.binance.com/market/ws/btcusdt@kline_1m"
KLINE_FALLBACK_URLS = (
    "wss://fstream.binance.com/market/stream?streams=btcusdt@kline_1m",
)
DEPTH_URL = "wss://fstream.binance.com/public/ws/btcusdt@depth20@100ms"
DEPTH_FALLBACK_URLS = (
    "wss://fstream.binance.com/public/stream?streams=btcusdt@depth20@100ms",
)
MARK_URL = "wss://fstream.binance.com/market/ws/btcusdt@markPrice@1s"
MARK_FALLBACK_URLS = (
    "wss://fstream.binance.com/market/stream?streams=btcusdt@markPrice@1s",
)
SCHEMA_VERSION = 1
MAX_CACHE_ROWS = 720


def utc_iso_from_ms(value: int) -> str:
    return datetime.fromtimestamp(int(value) / 1000, timezone.utc).isoformat()


def _unwrap(message: Any) -> dict[str, Any]:
    if not isinstance(message, dict):
        return {}
    data = message.get("data")
    return data if isinstance(data, dict) else message


def parse_kline_message(message: Any, received_at_ms: int | None = None) -> dict[str, Any] | None:
    data = _unwrap(message)
    if data.get("e") != "kline":
        return None
    k = data.get("k")
    if not isinstance(k, dict) or k.get("i") != "1m" or k.get("x") is not True:
        return None
    try:
        row = {
            "open_time_ms": int(k["t"]),
            "open": float(k["o"]),
            "high": float(k["h"]),
            "low": float(k["l"]),
            "close": float(k["c"]),
            "volume": float(k["v"]),
            "taker_buy_base": float(k["V"]),
            "event_time_ms": int(data["E"]),
            "retrieved_at_ms": int(received_at_ms if received_at_ms is not None else time.time() * 1000),
        }
    except (KeyError, TypeError, ValueError):
        return None
    vals = [row["open"], row["high"], row["low"], row["close"], row["volume"], row["taker_buy_base"]]
    if not all(math.isfinite(float(v)) for v in vals):
        return None
    if min(row["open"], row["high"], row["low"], row["close"]) <= 0 or row["volume"] < 0:
        return None
    if not (row["low"] <= min(row["open"], row["close"]) <= max(row["open"], row["close"]) <= row["high"]):
        return None
    if row["taker_buy_base"] < 0 or row["taker_buy_base"] > row["volume"] + 1e-9:
        return None
    if row["event_time_ms"] < row["open_time_ms"]:
        return None
    return row


def parse_depth_message(message: Any, received_at_ms: int | None = None) -> dict[str, Any] | None:
    """Parse current Binance USD-M Partial/Depth-Update payloads safely.

    The current public Partial Book Depth stream uses b/a arrays with
    exchange event timestamps E/T. Keep accepting the normalized
    bids/asks shape for already-persisted cache data and tests.
    """
    data = _unwrap(message)
    event_type = data.get("e")
    if event_type not in {"depthUpdate", None}:
        return None

    bids = data.get("b")
    asks = data.get("a")
    if bids is None and asks is None:
        bids = data.get("bids")
        asks = data.get("asks")
    if not isinstance(bids, list) or not isinstance(asks, list) or not bids or not asks:
        return None

    clean_bids: list[list[float]] = []
    clean_asks: list[list[float]] = []
    try:
        for side, out in ((bids, clean_bids), (asks, clean_asks)):
            for level in side[:20]:
                if not isinstance(level, (list, tuple)) or len(level) < 2:
                    return None
                price, qty = float(level[0]), float(level[1])
                if not (math.isfinite(price) and math.isfinite(qty) and price > 0 and qty >= 0):
                    return None
                out.append([price, qty])
    except (TypeError, ValueError, IndexError):
        return None
    if not clean_bids or not clean_asks:
        return None
    if clean_bids[0][0] > clean_asks[0][0]:
        return None

    symbol = data.get("s") or data.get("ps")
    if symbol is not None and str(symbol).upper() != "BTCUSDT":
        return None
    symbol_type = data.get("st")
    if symbol_type is not None:
        try:
            if int(symbol_type) != 1:
                return None
        except (TypeError, ValueError):
            return None

    try:
        event_time_ms = int(data["E"])
    except (KeyError, TypeError, ValueError):
        # Exchange event time is required for strict PIT provenance. Never
        # substitute local receipt time for an exchange timestamp.
        return None
    retrieved = int(received_at_ms if received_at_ms is not None else time.time() * 1000)
    if event_time_ms > retrieved + 60_000:
        return None
    update_id = data.get("u", data.get("lastUpdateId", 0))
    try:
        update_id = int(update_id)
    except (TypeError, ValueError):
        return None
    return {
        "bids": clean_bids,
        "asks": clean_asks,
        "last_update_id": update_id,
        "event_time_ms": event_time_ms,
        "retrieved_at_ms": retrieved,
    }

def depth_midprice(snapshot: dict[str, Any]) -> float:
    bids=snapshot.get("bids") if isinstance(snapshot,dict) else None
    asks=snapshot.get("asks") if isinstance(snapshot,dict) else None
    if not bids or not asks:
        raise ValueError("depth_snapshot_missing_sides")
    bid=float(bids[0][0]); ask=float(asks[0][0])
    if not (math.isfinite(bid) and math.isfinite(ask) and bid>0 and ask>0 and bid<=ask):
        raise ValueError("depth_snapshot_invalid_top_of_book")
    return (bid+ask)/2.0


def write_depth_cache(snapshot: dict[str, Any], path: Path = DEFAULT_DEPTH_CACHE) -> None:
    path.parent.mkdir(parents=True,exist_ok=True)
    stored_snapshot=dict(snapshot)
    # Preserve the parser discriminator so a serialized snapshot can be
    # revalidated exactly like the live Binance depthUpdate message.
    stored_snapshot.setdefault("e","depthUpdate")
    if "E" not in stored_snapshot and "event_time_ms" in stored_snapshot:
        stored_snapshot["E"] = int(stored_snapshot["event_time_ms"])
    payload={
        "schema_version":SCHEMA_VERSION,
        "source":"Binance USD-M Futures WebSocket",
        "stream":"btcusdt@depth20@100ms",
        "snapshot":stored_snapshot,
        "updated_at_utc":datetime.now(timezone.utc).isoformat(),
    }
    temp=path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temp.write_text(json.dumps(payload,sort_keys=True,separators=(",",":"))+"\n",encoding="utf-8")
    os.replace(temp,path)


def load_depth_cache(path: Path = DEFAULT_DEPTH_CACHE, max_age_ms: int = 180_000) -> dict[str, Any] | None:
    try:
        obj=json.loads(path.read_text(encoding="utf-8"))
        if obj.get("schema_version") != SCHEMA_VERSION or obj.get("source") != "Binance USD-M Futures WebSocket":
            return None
        if obj.get("stream") != "btcusdt@depth20@100ms":
            return None
        snap=obj.get("snapshot")
        if not isinstance(snap,dict):
            return None
        parse_input=dict(snap)
        parse_input.setdefault("e","depthUpdate")
        if "E" not in parse_input and "event_time_ms" in parse_input:
            parse_input["E"]=parse_input["event_time_ms"]
        parsed=parse_depth_message(
            parse_input,
            int(parse_input.get("retrieved_at_ms",0)),
        )
        if parsed is None:
            return None
        now_ms=int(time.time()*1000)
        retrieved_age=now_ms-int(parsed["retrieved_at_ms"])
        event_age=now_ms-int(parsed["event_time_ms"])
        if retrieved_age < 0 or retrieved_age > int(max_age_ms):
            return None
        if event_age < 0 or event_age > int(max_age_ms):
            return None
        return parsed
    except (OSError,ValueError,TypeError,KeyError):
        return None


def parse_mark_price_message(message: Any, received_at_ms: int | None = None) -> dict[str, Any] | None:
    data = _unwrap(message)
    if data.get("e") != "markPriceUpdate":
        return None
    try:
        event_time_ms = int(data["E"])
        mark_price = float(data["p"])
        funding_rate = float(data["r"])
    except (KeyError, TypeError, ValueError):
        return None
    if not all(math.isfinite(v) for v in (mark_price, funding_rate)) or mark_price <= 0:
        return None
    return {
        "mark_price": mark_price,
        "funding_rate": funding_rate,
        "event_time_ms": event_time_ms,
        "retrieved_at_ms": int(received_at_ms if received_at_ms is not None else time.time() * 1000),
    }


def row_to_market_data(row: dict[str, Any]) -> list[float | int]:
    return [
        int(row["open_time_ms"]),
        float(row["open"]),
        float(row["high"]),
        float(row["low"]),
        float(row["close"]),
        float(row["volume"]),
    ]


def load_cache(path: Path = DEFAULT_CACHE, limit: int = MAX_CACHE_ROWS) -> list[dict[str, Any]]:
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, ValueError, TypeError):
        return []
    rows = obj.get("rows") if isinstance(obj, dict) else None
    if not isinstance(rows, list):
        return []
    out = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        try:
            normalized = {
                "open_time_ms": int(row["open_time_ms"]),
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": float(row["volume"]),
                "taker_buy_base": float(row["taker_buy_base"]),
                "event_time_ms": int(row["event_time_ms"]),
                "retrieved_at_ms": int(row["retrieved_at_ms"]),
            }
        except (KeyError, TypeError, ValueError):
            continue
        if parse_kline_message({"e": "kline", "E": normalized["event_time_ms"], "k": {
            "t": normalized["open_time_ms"], "o": normalized["open"], "h": normalized["high"],
            "l": normalized["low"], "c": normalized["close"], "v": normalized["volume"],
            "V": normalized["taker_buy_base"], "i": "1m", "x": True,
        }}, normalized["retrieved_at_ms"]) is not None:
            out.append(normalized)
    out.sort(key=lambda r: r["open_time_ms"])
    dedup = {r["open_time_ms"]: r for r in out}
    return list(sorted(dedup.values(), key=lambda r: r["open_time_ms"]))[-int(limit):]


def merge_cache(
    existing: list[dict[str, Any]],
    incoming: list[dict[str, Any]],
    max_rows: int = MAX_CACHE_ROWS,
) -> list[dict[str, Any]]:
    merged = {int(r["open_time_ms"]): r for r in existing if isinstance(r, dict)}
    for row in incoming:
        if isinstance(row, dict):
            merged[int(row["open_time_ms"])] = row
    return [merged[k] for k in sorted(merged)[-int(max_rows):]]


def contiguous_suffix(rows: list[dict[str, Any]], minimum: int = 40) -> list[dict[str, Any]]:
    ordered = sorted(rows, key=lambda r: int(r["open_time_ms"]))
    if not ordered:
        return []
    start = len(ordered) - 1
    while start > 0 and int(ordered[start]["open_time_ms"]) - int(ordered[start - 1]["open_time_ms"]) == 60_000:
        start -= 1
    suffix = ordered[start:]
    return suffix if len(suffix) >= int(minimum) else []


def taker_imbalance(rows: list[dict[str, Any]], window: int = 5) -> tuple[float, int] | None:
    suffix = contiguous_suffix(rows, minimum=window)
    if len(suffix) < window:
        return None
    suffix = suffix[-window:]
    total = 0.0
    buy = 0.0
    latest_event = 0
    for row in suffix:
        volume = float(row["volume"])
        taker_buy = float(row["taker_buy_base"])
        if volume <= 0 or taker_buy < 0 or taker_buy > volume + 1e-9:
            return None
        total += volume
        buy += taker_buy
        latest_event = max(latest_event, int(row["event_time_ms"]))
    if total <= 0:
        return None
    return (2.0 * buy / total - 1.0, latest_event)


async def _collect_url(url: str, timeout_seconds: float, parser) -> list[dict[str, Any]]:
    deadline = time.monotonic() + float(timeout_seconds)
    out: list[dict[str, Any]] = []
    try:
        async with websockets.connect(
            url,
            ping_interval=20,
            ping_timeout=10,
            open_timeout=10,
            close_timeout=5,
            max_size=2_000_000,
        ) as ws:
            while time.monotonic() < deadline:
                remaining = max(0.25, deadline - time.monotonic())
                try:
                    # Kline updates arrive frequently. Treat a silent transport as
                    # unhealthy after 45s so the remaining window can fail over
                    # without forcing a full reconnect at every checkpoint.
                    raw = await asyncio.wait_for(ws.recv(), timeout=min(remaining, 45.0))
                except (asyncio.TimeoutError, websockets.exceptions.ConnectionClosed):
                    break
                received_ms = int(time.time() * 1000)
                try:
                    msg = json.loads(raw)
                except (TypeError, ValueError):
                    continue
                parsed = parser(msg, received_ms)
                if parsed is not None:
                    out.append(parsed)
    except Exception:
        return out
    return out


async def _collect_with_fallback(urls, timeout_seconds: float, parser) -> tuple[list[dict[str, Any]], str | None]:
    """Try the documented raw stream first, then a legacy-compatible path.

    A fallback is used only when the primary endpoint yields no parsed
    observations. Parser-level validation remains authoritative, so an
    endpoint can never turn malformed/future data into an accepted row.
    """
    for url in urls:
        rows = await _collect_url(url, timeout_seconds, parser)
        if rows:
            return rows, url
    return [], None


async def _stream_url(
    url: str,
    timeout_seconds: float,
    parser,
    on_row,
) -> tuple[int, bool]:
    """Keep one WebSocket connection open for the bounded capture window.

    Returns (parsed_row_count, connection_started). If the connection closes
    early, the caller can fail over to another transport without restarting the
    entire capture window.
    """
    deadline = time.monotonic() + float(timeout_seconds)
    parsed_count = 0
    connection_started = False
    try:
        async with websockets.connect(
            url,
            ping_interval=20,
            ping_timeout=10,
            open_timeout=10,
            close_timeout=5,
            max_size=2_000_000,
        ) as ws:
            connection_started = True
            last_accepted_row_at = time.monotonic()
            while time.monotonic() < deadline:
                remaining = max(0.25, deadline - time.monotonic())
                if time.monotonic() - last_accepted_row_at >= 150.0:
                    # The socket may still be emitting open-kline updates while
                    # never delivering a valid closed bar. Treat that transport
                    # as unhealthy and fail over instead of waiting for 45 minutes.
                    break
                try:
                    # Bound both transport silence and accepted closed-bar silence.
                    raw = await asyncio.wait_for(
                        ws.recv(),
                        timeout=min(remaining, 45.0),
                    )
                except (asyncio.TimeoutError, websockets.exceptions.ConnectionClosed):
                    break
                received_ms = int(time.time() * 1000)
                try:
                    msg = json.loads(raw)
                except (TypeError, ValueError):
                    continue
                parsed = parser(msg, received_ms)
                if parsed is None:
                    continue
                parsed_count += 1
                last_accepted_row_at = time.monotonic()
                await on_row(parsed)
    except Exception as exc:
        # Keep transport failures observable in Actions logs while preserving
        # fail-closed semantics: parser/reconnect policy still decides acceptance.
        print(
            f"Binance WS transport error url={url} type={type(exc).__name__} "
            f"detail={str(exc)[:240]!r}",
            flush=True,
        )
        return parsed_count, connection_started
    return parsed_count, connection_started


async def capture_closed_klines_stream(
    timeout_seconds: float = 2700.0,
    checkpoint_seconds: float = 60.0,
    initial_rows: list[dict[str, Any]] | None = None,
    on_checkpoint=None,
) -> list[dict[str, Any]]:
    """Capture closed 1m bars on one continuous connection with failover.

    The primary stream is kept open for the entire bounded window. A legacy
    endpoint is only attempted if the current connection cannot continue.
    Checkpoints are emitted without closing the WebSocket so the cache publisher
    can persist fresh data while the same connection continues receiving bars.
    """
    merged = {
        int(row["open_time_ms"]): row
        for row in (initial_rows or [])
        if isinstance(row, dict)
    }
    started_at = time.monotonic()
    last_checkpoint = started_at

    async def on_row(row: dict[str, Any]) -> None:
        nonlocal last_checkpoint
        merged[int(row["open_time_ms"])] = row
        if on_checkpoint is not None and (
            time.monotonic() - last_checkpoint >= float(checkpoint_seconds)
        ):
            rows = [
                merged[k]
                for k in sorted(merged)[-MAX_CACHE_ROWS:]
            ]
            result = on_checkpoint(rows)
            if asyncio.iscoroutine(result):
                await result
            last_checkpoint = time.monotonic()

    deadline = started_at + float(timeout_seconds)
    urls = (KLINE_URL, *KLINE_FALLBACK_URLS)
    url_index = 0
    reconnect_backoff_seconds = 0.5
    while time.monotonic() < deadline:
        remaining = deadline - time.monotonic()
        url = urls[url_index % len(urls)]
        count, started = await _stream_url(
            url,
            remaining,
            parse_kline_message,
            on_row,
        )
        if time.monotonic() >= deadline:
            break
        # A dropped socket must not end the capture window. Reconnect the
        # primary transport first, then its fallback, until the bounded window
        # expires. A stable connection still consumes the full remaining budget.
        if started and count > 0:
            url_index = 0
            reconnect_backoff_seconds = 0.5
        else:
            url_index += 1
            reconnect_backoff_seconds = min(5.0, reconnect_backoff_seconds * 2.0)
        sleep_for = min(
            reconnect_backoff_seconds,
            max(0.0, deadline - time.monotonic()),
        )
        if sleep_for > 0:
            await asyncio.sleep(sleep_for)

    rows = [merged[k] for k in sorted(merged)[-MAX_CACHE_ROWS:]]
    if on_checkpoint is not None:
        result = on_checkpoint(rows)
        if asyncio.iscoroutine(result):
            await result
    return rows


async def capture_closed_klines(timeout_seconds: float = 62.0) -> list[dict[str, Any]]:
    rows, _ = await _collect_with_fallback((KLINE_URL, *KLINE_FALLBACK_URLS), timeout_seconds, parse_kline_message)
    return rows


async def capture_depth_snapshot_stream(
    timeout_seconds: float = 2700.0,
    checkpoint_seconds: float = 60.0,
    initial_snapshot: dict[str, Any] | None = None,
    on_checkpoint=None,
) -> dict[str, Any] | None:
    latest=initial_snapshot
    last_checkpoint=time.monotonic()

    async def on_row(row: dict[str, Any]) -> None:
        nonlocal latest,last_checkpoint
        latest=row
        if on_checkpoint is not None and time.monotonic()-last_checkpoint >= float(checkpoint_seconds):
            result=on_checkpoint(row)
            if asyncio.iscoroutine(result):
                await result
            last_checkpoint=time.monotonic()

    deadline=time.monotonic()+float(timeout_seconds)
    urls=(DEPTH_URL,*DEPTH_FALLBACK_URLS)
    url_index=0
    reconnect_backoff_seconds=1.0
    while time.monotonic()<deadline:
        remaining=deadline-time.monotonic()
        url=urls[url_index % len(urls)]
        count,started=await _stream_url(url,remaining,parse_depth_message,on_row)
        if time.monotonic()>=deadline:
            break
        # Keep retrying the same bounded capture window after transport drops.
        # Live prediction still applies strict depth freshness and fail-closed
        # validation; this only makes the cache collector recoverable.
        if started and count>0:
            url_index=0
            reconnect_backoff_seconds=1.0
        else:
            url_index+=1
            reconnect_backoff_seconds=min(10.0,reconnect_backoff_seconds*2.0)
        sleep_for=min(
            reconnect_backoff_seconds,
            max(0.0,deadline-time.monotonic()),
        )
        if sleep_for>0:
            await asyncio.sleep(sleep_for)
    if latest is not None and on_checkpoint is not None:
        result=on_checkpoint(latest)
        if asyncio.iscoroutine(result):
            await result
    return latest


async def capture_depth_snapshot(timeout_seconds: float = 6.0) -> dict[str, Any] | None:
    # Use the same bounded reconnect/failover logic as the long-lived depth
    # collector so an early socket drop does not consume the entire live recovery
    # window on a single dead transport.
    return await capture_depth_snapshot_stream(timeout_seconds=timeout_seconds)


async def capture_mark_price(timeout_seconds: float = 6.0) -> dict[str, Any] | None:
    rows, _ = await _collect_with_fallback((MARK_URL, *MARK_FALLBACK_URLS), timeout_seconds, parse_mark_price_message)
    return rows[-1] if rows else None


def write_cache(
    rows: list[dict[str, Any]],
    path: Path = DEFAULT_CACHE,
    recovery_epoch_ms: int | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if recovery_epoch_ms is None:
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
            recovery_epoch_ms = int(existing.get("recovery_epoch_ms", 0) or 0)
        except (FileNotFoundError, OSError, ValueError, TypeError, AttributeError):
            recovery_epoch_ms = 0
    payload = {
        "schema_version": SCHEMA_VERSION,
        "source": "Binance USD-M Futures WebSocket",
        "stream": "btcusdt@kline_1m",
        "rows": rows[-MAX_CACHE_ROWS:],
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    if int(recovery_epoch_ms or 0) > 0:
        payload["recovery_epoch_ms"] = int(recovery_epoch_ms)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temp.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    os.replace(temp, path)


async def collect_forever_window(
    timeout_seconds: float,
    path: Path,
    checkpoint_seconds: float = 60.0,
) -> dict[str, Any]:
    existing = load_cache(path)
    recovery_epoch_ms = 0
    try:
        existing_obj = json.loads(path.read_text(encoding="utf-8"))
        recovery_epoch_ms = int(existing_obj.get("recovery_epoch_ms", 0) or 0)
    except (FileNotFoundError, OSError, ValueError, TypeError, AttributeError):
        recovery_epoch_ms = 0

    async def checkpoint(rows: list[dict[str, Any]]) -> None:
        write_cache(rows, path, recovery_epoch_ms=recovery_epoch_ms)

    merged = await capture_closed_klines_stream(
        timeout_seconds,
        checkpoint_seconds=checkpoint_seconds,
        initial_rows=existing,
        on_checkpoint=checkpoint,
    )
    write_cache(merged, path, recovery_epoch_ms=recovery_epoch_ms)
    suffix = contiguous_suffix(merged, minimum=40)
    latest = merged[-1] if merged else None
    return {
        "schema_version": SCHEMA_VERSION,
        "captured_rows": max(0, len(merged) - len(existing)),
        "cache_rows": len(merged),
        "contiguous_rows": len(suffix),
        "latest_open_time_ms": latest["open_time_ms"] if latest else None,
        "latest_event_time_ms": latest["event_time_ms"] if latest else None,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    }


async def collect_depth_forever(
    timeout_seconds: float,
    path: Path,
    checkpoint_seconds: float = 60.0,
) -> dict[str, Any]:
    initial = load_depth_cache(path, max_age_ms=24 * 60 * 60 * 1000)

    async def checkpoint(snapshot: dict[str, Any]) -> None:
        write_depth_cache(snapshot, path)

    latest = await capture_depth_snapshot_stream(
        timeout_seconds,
        checkpoint_seconds=checkpoint_seconds,
        initial_snapshot=initial,
        on_checkpoint=checkpoint,
    )
    if latest is not None:
        write_depth_cache(latest, path)
    return {
        "schema_version": SCHEMA_VERSION,
        "mode": "depth",
        "ok": latest is not None,
        "event_time_ms": latest["event_time_ms"] if latest else None,
        "retrieved_at_ms": latest["retrieved_at_ms"] if latest else None,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("kline", "depth"), default="kline")
    parser.add_argument("--capture-seconds", type=float, default=2700.0)
    parser.add_argument("--checkpoint-seconds", type=float, default=60.0)
    parser.add_argument("--output", type=Path, default=DEFAULT_CACHE)
    args = parser.parse_args()
    if args.mode == "depth":
        result = asyncio.run(
            collect_depth_forever(
                args.capture_seconds,
                args.output,
                checkpoint_seconds=args.checkpoint_seconds,
            )
        )
    else:
        result = asyncio.run(
            collect_forever_window(
                args.capture_seconds,
                args.output,
                checkpoint_seconds=args.checkpoint_seconds,
            )
        )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
