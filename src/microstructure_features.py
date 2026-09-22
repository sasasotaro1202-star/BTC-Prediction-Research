"""Deterministic, research-only microstructure feature derivations.

All inputs must be already-closed observations available at prediction time.
This module performs no network access, no imputation, and no future lookup.
"""
from __future__ import annotations

import math
from typing import Any, Iterable


def _finite(value: Any) -> float | None:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _ohlcv(rows: Iterable[Iterable[Any]]) -> tuple[list[float], list[float], list[float], list[float], list[float]]:
    opens: list[float] = []
    highs: list[float] = []
    lows: list[float] = []
    closes: list[float] = []
    volumes: list[float] = []
    for row in rows:
        try:
            o, h, l, c, v = map(_finite, (row[1], row[2], row[3], row[4], row[5]))
        except (IndexError, TypeError):
            return [], [], [], [], []
        if None in (o, h, l, c, v) or min(o, h, l, c) <= 0 or v < 0:
            return [], [], [], [], []
        if h < max(o, c) or l > min(o, c):
            return [], [], [], [], []
        opens.append(float(o))
        highs.append(float(h))
        lows.append(float(l))
        closes.append(float(c))
        volumes.append(float(v))
    return opens, highs, lows, closes, volumes


def vwap_distance(rows: list[Iterable[Any]], window: int) -> float | None:
    if len(rows) < window or window <= 0:
        return None
    _, _, _, closes, volumes = _ohlcv(rows[-window:])
    if len(closes) != window:
        return None
    total_volume = sum(volumes)
    if total_volume <= 0:
        return None
    vwap = sum(c * v for c, v in zip(closes, volumes)) / total_volume
    if not math.isfinite(vwap) or vwap <= 0:
        return None
    value = closes[-1] / vwap - 1.0
    return value if math.isfinite(value) else None


def volume_burst(rows: list[Iterable[Any]], recent: int, baseline: int) -> float | None:
    if recent <= 0 or baseline <= 0 or len(rows) < recent + baseline:
        return None
    *_, volumes = _ohlcv(rows[-(recent + baseline):])
    if len(volumes) != recent + baseline:
        return None
    base = sum(volumes[:baseline]) / baseline
    cur = sum(volumes[baseline:]) / recent
    if base <= 0 or not math.isfinite(base) or not math.isfinite(cur):
        return None
    value = cur / base
    return value if math.isfinite(value) else None


def range_compression(rows: list[Iterable[Any]], recent: int, baseline: int) -> float | None:
    if recent <= 0 or baseline <= 0 or len(rows) < recent + baseline:
        return None
    _, highs, lows, closes, _ = _ohlcv(rows[-(recent + baseline):])
    if len(closes) != recent + baseline:
        return None
    ranges = [
        (h - l) / c
        for h, l, c in zip(highs, lows, closes)
    ]
    base = sum(ranges[:baseline]) / baseline
    cur = sum(ranges[baseline:]) / recent
    if base <= 0 or not math.isfinite(base) or not math.isfinite(cur):
        return None
    value = cur / base
    return value if math.isfinite(value) else None


def _taker_window(rows: list[dict[str, Any]], window: int) -> float | None:
    if window <= 0 or len(rows) < window:
        return None
    ordered = sorted(rows, key=lambda r: int(r["open_time_ms"]))
    start = len(ordered) - window
    for i in range(start + 1, len(ordered)):
        if int(ordered[i]["open_time_ms"]) - int(ordered[i - 1]["open_time_ms"]) != 60_000:
            return None
    total = 0.0
    buy = 0.0
    for row in ordered[start:]:
        volume = _finite(row.get("volume"))
        taker_buy = _finite(row.get("taker_buy_base"))
        if volume is None or taker_buy is None or volume <= 0 or taker_buy < 0 or taker_buy > volume + 1e-9:
            return None
        total += volume
        buy += taker_buy
    if total <= 0:
        return None
    value = 2.0 * buy / total - 1.0
    return value if math.isfinite(value) else None


def derive_market_flow_features(
    rows: list[Iterable[Any]],
    *,
    ws_rows: list[dict[str, Any]] | None = None,
) -> dict[str, float | None]:
    """Return complete-case-friendly derived features; missing inputs remain None."""
    result = {
        "vwap_distance_5m": vwap_distance(rows, 5),
        "vwap_distance_15m": vwap_distance(rows, 15),
        "vwap_distance_30m": vwap_distance(rows, 30),
        "volume_burst_5m": volume_burst(rows, 5, 15),
        "volume_burst_15m": volume_burst(rows, 15, 30),
        "range_compression_5m": range_compression(rows, 5, 15),
        "range_compression_15m": range_compression(rows, 15, 30),
        "taker_imbalance_5m": _taker_window(ws_rows or [], 5),
        "taker_imbalance_15m": _taker_window(ws_rows or [], 15),
    }
    t5 = result["taker_imbalance_5m"]
    t15 = result["taker_imbalance_15m"]
    result["taker_imbalance_delta_5m_15m"] = (
        t5 - t15 if t5 is not None and t15 is not None else None
    )
    return result
