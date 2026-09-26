"""Research-only BTC event normalization and situation aggregation.

This layer is intentionally separate from the production predictor. It turns
validated market observations into Opta-like, provenance-carrying events and
builds prediction-time situation snapshots without weakening PIT rules.

Strict mode is fail-closed:
- event_time_ms, available_at_ms, and retrieved_at_ms are mandatory;
- available_at_ms must not be after retrieved_at_ms;
- events after prediction_time_ms are excluded;
- events whose availability is after prediction_time_ms are rejected;
- malformed provenance raises instead of silently becoming zero.
"""
from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Iterable

SCHEMA_VERSION = 1
DEFAULT_WINDOWS_MS = (60_000, 300_000)
BTC_SYMBOLS = {"BTCUSDT", "BTC-USD", "BTCUSD", "XBTUSD"}


def _finite(value: Any) -> float:
    out = float(value)
    if not math.isfinite(out):
        raise ValueError("non_finite_numeric_value")
    return out


def _positive_int(value: Any, name: str) -> int:
    try:
        out = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid_{name}") from exc
    if out <= 0:
        raise ValueError(f"invalid_{name}")
    return out


def _canonical_hash(payload: Any) -> str:
    raw = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def make_event(
    *,
    source: str,
    venue: str,
    event_type: str,
    event_time_ms: int,
    available_at_ms: int,
    retrieved_at_ms: int,
    payload: dict[str, Any],
    event_id: str | None = None,
    symbol: str = "BTCUSDT",
) -> dict[str, Any]:
    """Build a normalized PIT-carrying event.

    available_at_ms is the timestamp at which this exact observation became
    usable by the research system. For a live WebSocket observation this is
    normally the local receipt time. Historical data must provide an explicit
    availability boundary; this function never guesses one.
    """
    if not isinstance(payload, dict):
        raise ValueError("event_payload_must_be_object")
    if not source or not venue or not event_type:
        raise ValueError("event_identity_missing")

    event_time = _positive_int(event_time_ms, "event_time_ms")
    available = _positive_int(available_at_ms, "available_at_ms")
    retrieved = _positive_int(retrieved_at_ms, "retrieved_at_ms")

    if event_time > retrieved:
        raise ValueError("event_time_after_retrieval")
    if available > retrieved:
        raise ValueError("available_at_after_retrieval")

    clean_symbol = str(symbol).upper().strip()
    if clean_symbol not in BTC_SYMBOLS and clean_symbol != "BTCUSDT_PERP":
        raise ValueError("unsupported_symbol")

    normalized_payload = json.loads(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    )
    stable_id = str(event_id) if event_id else _canonical_hash({
        "source": source,
        "venue": venue,
        "event_type": event_type,
        "event_time_ms": event_time,
        "payload": normalized_payload,
    })

    return {
        "schema_version": SCHEMA_VERSION,
        "event_id": stable_id,
        "source": str(source),
        "venue": str(venue),
        "symbol": clean_symbol,
        "event_type": str(event_type),
        "event_time_ms": event_time,
        "available_at_ms": available,
        "retrieved_at_ms": retrieved,
        "payload_sha256": _canonical_hash(normalized_payload),
        "payload": normalized_payload,
    }


def event_from_binance_kline(
    row: dict[str, Any],
    *,
    available_at_ms: int | None = None,
) -> dict[str, Any]:
    """Convert an already validated Binance WS 1m cache row into an event."""
    if not isinstance(row, dict):
        raise ValueError("binance_kline_row_must_be_object")
    required = {
        "open_time_ms",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "taker_buy_base",
        "event_time_ms",
        "retrieved_at_ms",
    }
    missing = sorted(required - set(row))
    if missing:
        raise ValueError(f"binance_kline_missing_fields:{','.join(missing)}")

    open_price = _finite(row["open"])
    high = _finite(row["high"])
    low = _finite(row["low"])
    close = _finite(row["close"])
    volume = _finite(row["volume"])
    taker_buy = _finite(row["taker_buy_base"])

    if min(open_price, high, low, close) <= 0 or volume < 0:
        raise ValueError("binance_kline_invalid_values")
    if low > min(open_price, close) or high < max(open_price, close):
        raise ValueError("binance_kline_ohlc_inconsistent")
    if taker_buy < 0 or taker_buy > volume + 1e-9:
        raise ValueError("binance_kline_taker_buy_out_of_range")

    taker_sell = max(0.0, volume - taker_buy)
    imbalance = 0.0 if volume <= 0 else (taker_buy - taker_sell) / volume
    retrieved = _positive_int(row["retrieved_at_ms"], "retrieved_at_ms")
    available = retrieved if available_at_ms is None else _positive_int(
        available_at_ms, "available_at_ms"
    )

    return make_event(
        source="binance_ws",
        venue="binance_usdm_futures",
        symbol="BTCUSDT_PERP",
        event_type="kline_1m",
        event_time_ms=_positive_int(row["event_time_ms"], "event_time_ms"),
        available_at_ms=available,
        retrieved_at_ms=retrieved,
        event_id=f"binance-kline-{_positive_int(row['open_time_ms'], 'open_time_ms')}",
        payload={
            "open_time_ms": _positive_int(row["open_time_ms"], "open_time_ms"),
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "taker_buy_base": taker_buy,
            "taker_sell_base": taker_sell,
            "taker_imbalance": imbalance,
        },
    )


def event_from_binance_depth(
    snapshot: dict[str, Any],
    *,
    available_at_ms: int | None = None,
) -> dict[str, Any]:
    """Convert a validated Binance partial-depth snapshot into one event."""
    if not isinstance(snapshot, dict):
        raise ValueError("binance_depth_snapshot_must_be_object")
    bids = snapshot.get("bids")
    asks = snapshot.get("asks")
    if not isinstance(bids, list) or not isinstance(asks, list) or not bids or not asks:
        raise ValueError("binance_depth_missing_sides")

    def clean_side(levels: list[Any]) -> list[tuple[float, float]]:
        out: list[tuple[float, float]] = []
        for level in levels[:20]:
            if not isinstance(level, (list, tuple)) or len(level) < 2:
                raise ValueError("binance_depth_level_invalid")
            price = _finite(level[0])
            qty = _finite(level[1])
            if price <= 0 or qty < 0:
                raise ValueError("binance_depth_level_out_of_range")
            out.append((price, qty))
        if not out:
            raise ValueError("binance_depth_side_empty")
        return out

    bid_levels = clean_side(bids)
    ask_levels = clean_side(asks)
    best_bid = bid_levels[0][0]
    best_ask = ask_levels[0][0]
    if best_bid > best_ask:
        raise ValueError("binance_depth_crossed_book")

    bid_qty_5 = sum(q for _, q in bid_levels[:5])
    ask_qty_5 = sum(q for _, q in ask_levels[:5])
    depth_total = bid_qty_5 + ask_qty_5
    depth_imbalance = (
        0.0 if depth_total <= 0 else (bid_qty_5 - ask_qty_5) / depth_total
    )
    mid = (best_bid + best_ask) / 2.0
    spread_bps = 0.0 if mid <= 0 else (best_ask - best_bid) / mid * 10_000.0

    retrieved = _positive_int(snapshot.get("retrieved_at_ms"), "retrieved_at_ms")
    available = retrieved if available_at_ms is None else _positive_int(
        available_at_ms, "available_at_ms"
    )
    event_time = _positive_int(snapshot.get("event_time_ms"), "event_time_ms")
    update_id = _positive_int(snapshot.get("last_update_id", 1), "last_update_id")

    return make_event(
        source="binance_ws",
        venue="binance_usdm_futures",
        symbol="BTCUSDT_PERP",
        event_type="depth20",
        event_time_ms=event_time,
        available_at_ms=available,
        retrieved_at_ms=retrieved,
        event_id=f"binance-depth-{event_time}-{update_id}",
        payload={
            "best_bid": best_bid,
            "best_ask": best_ask,
            "bid_qty_5": bid_qty_5,
            "ask_qty_5": ask_qty_5,
            "depth_imbalance_5": depth_imbalance,
            "spread_bps": spread_bps,
            "levels_bid": len(bid_levels),
            "levels_ask": len(ask_levels),
            "last_update_id": update_id,
        },
    )


def _validate_event_provenance(event: dict[str, Any]) -> None:
    if not isinstance(event, dict):
        raise ValueError("event_must_be_object")
    for key in ("event_time_ms", "available_at_ms", "retrieved_at_ms", "event_id"):
        if key not in event:
            raise ValueError(f"event_provenance_missing:{key}")
    event_time = _positive_int(event["event_time_ms"], "event_time_ms")
    available = _positive_int(event["available_at_ms"], "available_at_ms")
    retrieved = _positive_int(event["retrieved_at_ms"], "retrieved_at_ms")
    if event_time > retrieved:
        raise ValueError("event_time_after_retrieval")
    if available > retrieved:
        raise ValueError("available_at_after_retrieval")


def _window_events(
    events: Iterable[dict[str, Any]],
    *,
    prediction_time_ms: int,
    window_ms: int,
    strict_pit: bool,
) -> list[dict[str, Any]]:
    cutoff = _positive_int(prediction_time_ms, "prediction_time_ms")
    width = _positive_int(window_ms, "window_ms")
    start = cutoff - width
    selected: list[dict[str, Any]] = []

    for event in events:
        try:
            _validate_event_provenance(event)
            event_time = int(event["event_time_ms"])
            available = int(event["available_at_ms"])
        except ValueError:
            if strict_pit:
                raise
            continue

        if event_time > cutoff:
            continue
        if available > cutoff:
            if strict_pit and event_time >= start:
                raise ValueError("event_available_after_prediction_cutoff")
            continue
        if event_time >= start:
            selected.append(event)

    selected.sort(key=lambda item: (int(item["event_time_ms"]), str(item["event_id"])))
    return selected


def _mean(values: list[float]) -> float | None:
    return None if not values else sum(values) / len(values)


def aggregate_event_window(
    events: Iterable[dict[str, Any]],
    *,
    prediction_time_ms: int,
    window_ms: int,
    strict_pit: bool = True,
) -> dict[str, Any]:
    """Aggregate one causal event window into research-only situation features."""
    selected = _window_events(
        events,
        prediction_time_ms=prediction_time_ms,
        window_ms=window_ms,
        strict_pit=strict_pit,
    )
    cutoff = int(prediction_time_ms)

    volumes: list[float] = []
    buy_volumes: list[float] = []
    depth_imbalances: list[float] = []
    spreads: list[float] = []
    availability_lags: list[float] = []
    source_set: set[str] = set()
    venue_set: set[str] = set()
    event_types: dict[str, int] = {}
    latest_event = None
    latest_available = None

    for event in selected:
        payload = event.get("payload")
        if not isinstance(payload, dict):
            if strict_pit:
                raise ValueError("event_payload_missing")
            continue
        source_set.add(str(event.get("source", "")))
        venue_set.add(str(event.get("venue", "")))
        kind = str(event.get("event_type", "unknown"))
        event_types[kind] = event_types.get(kind, 0) + 1

        if "volume" in payload:
            volume = _finite(payload["volume"])
            taker_buy = _finite(payload.get("taker_buy_base", 0.0))
            if volume < 0 or taker_buy < 0 or taker_buy > volume + 1e-9:
                raise ValueError("aggregated_kline_volume_invalid")
            volumes.append(volume)
            buy_volumes.append(taker_buy)

        if "depth_imbalance_5" in payload:
            depth_imbalances.append(_finite(payload["depth_imbalance_5"]))
        if "spread_bps" in payload:
            spreads.append(_finite(payload["spread_bps"]))

        event_time = int(event["event_time_ms"])
        available = int(event["available_at_ms"])
        availability_lags.append(float(available - event_time))
        latest_event = event_time if latest_event is None else max(latest_event, event_time)
        latest_available = available if latest_available is None else max(latest_available, available)

    total_volume = sum(volumes)
    buy_volume = sum(buy_volumes)
    sell_volume = max(0.0, total_volume - buy_volume)
    trade_imbalance = (
        None if total_volume <= 0 else (buy_volume - sell_volume) / total_volume
    )

    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "prediction_time_ms": cutoff,
        "window_ms": int(window_ms),
        "event_count": len(selected),
        "event_types": event_types,
        "source_count": len(source_set),
        "venue_count": len(venue_set),
        "volume": total_volume,
        "buy_volume": buy_volume,
        "sell_volume": sell_volume,
        "trade_imbalance": trade_imbalance,
        "depth_imbalance_5_mean": _mean(depth_imbalances),
        "spread_bps_mean": _mean(spreads),
        "availability_lag_ms_mean": _mean(availability_lags),
        "latest_event_age_ms": None if latest_event is None else cutoff - latest_event,
        "latest_available_age_ms": None if latest_available is None else cutoff - latest_available,
        "event_ids": [str(event["event_id"]) for event in selected],
        "pit_verified": strict_pit,
    }
    return result


def build_situation_card(
    events: Iterable[dict[str, Any]],
    *,
    prediction_time_ms: int,
    windows_ms: tuple[int, ...] = DEFAULT_WINDOWS_MS,
    strict_pit: bool = True,
) -> dict[str, Any]:
    """Build an Opta-like multi-window BTC situation card.

    The output carries the exact event IDs used so every derived feature can be
    traced back to immutable observations. It is research-only and does not
    alter production model inputs.
    """
    unique_windows = tuple(dict.fromkeys(int(x) for x in windows_ms))
    if not unique_windows or any(x <= 0 for x in unique_windows):
        raise ValueError("invalid_situation_windows")
    materialized = list(events)

    windows = {}
    for width in unique_windows:
        windows[str(width)] = aggregate_event_window(
            materialized,
            prediction_time_ms=prediction_time_ms,
            window_ms=width,
            strict_pit=strict_pit,
        )

    used_ids: list[str] = []
    seen: set[str] = set()
    for snapshot in windows.values():
        for event_id in snapshot["event_ids"]:
            if event_id not in seen:
                seen.add(event_id)
                used_ids.append(event_id)

    return {
        "schema_version": SCHEMA_VERSION,
        "prediction_time_ms": int(prediction_time_ms),
        "strict_pit": bool(strict_pit),
        "windows": windows,
        "event_ids": used_ids,
        "event_count": len(used_ids),
    }
