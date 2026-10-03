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


ORDERBOOK_V2 = (
    "depth_imbalance_5",
    "depth_imbalance_10",
    "depth_imbalance_20",
    "weighted_depth_imbalance_20",
    "spread_bps",
    "microprice_gap",
    "depth_concentration_imbalance",
    "depth_notional_imbalance_20",
)

# V3 adds order-book *shape* features that are not simple duplicates of the
# existing aggregate imbalance measures. They use only the same validated
# 20-level Binance depth snapshot and remain research-only.
ORDERBOOK_V3 = (
    "depth_imbalance_gradient_5_20",
    "near_far_imbalance_5_20",
    "bid_depth_slope_bps_20",
    "ask_depth_slope_bps_20",
    "depth_slope_asymmetry_20",
    "depth_liquidity_concentration_20",
)


def _book_levels(snapshot: dict[str, Any]) -> tuple[list[tuple[float, float]], list[tuple[float, float]]] | None:
    if not isinstance(snapshot, dict):
        return None
    bids = snapshot.get("bids")
    asks = snapshot.get("asks")
    if not isinstance(bids, list) or not isinstance(asks, list) or len(bids) < 20 or len(asks) < 20:
        return None
    out_bids: list[tuple[float, float]] = []
    out_asks: list[tuple[float, float]] = []
    try:
        for raw in bids[:20]:
            price = _finite(raw[0])
            qty = _finite(raw[1])
            if price is None or qty is None or price <= 0 or qty < 0:
                return None
            out_bids.append((price, qty))
        for raw in asks[:20]:
            price = _finite(raw[0])
            qty = _finite(raw[1])
            if price is None or qty is None or price <= 0 or qty < 0:
                return None
            out_asks.append((price, qty))
    except (TypeError, ValueError, IndexError):
        return None
    if out_bids[0][0] > out_asks[0][0]:
        return None
    return out_bids, out_asks


def _imbalance(bids: list[tuple[float, float]], asks: list[tuple[float, float]], levels: int) -> float | None:
    b = sum(q for _, q in bids[:levels])
    a = sum(q for _, q in asks[:levels])
    denom = b + a
    if denom <= 0 or not math.isfinite(denom):
        return None
    value = (b - a) / denom
    return value if math.isfinite(value) else None


def derive_orderbook_features(snapshot: dict[str, Any] | None) -> dict[str, float] | None:
    """Derive causal order-book features from one validated Binance depth snapshot.

    The snapshot is required to contain 20 valid bid/ask levels. No interpolation,
    imputation, future lookup, or network access is performed here.
    """
    parsed = _book_levels(snapshot or {})
    if parsed is None:
        return None
    bids, asks = parsed
    bid_px, bid_qty = bids[0]
    ask_px, ask_qty = asks[0]
    if ask_px < bid_px or bid_qty + ask_qty <= 0:
        return None
    mid = (bid_px + ask_px) / 2.0
    if mid <= 0 or not math.isfinite(mid):
        return None

    spread_bps = (ask_px - bid_px) / mid * 10_000.0
    microprice = (ask_px * bid_qty + bid_px * ask_qty) / (bid_qty + ask_qty)
    microprice_gap = microprice / mid - 1.0

    weighted_bid = sum(q / (idx + 1.0) for idx, (_, q) in enumerate(bids))
    weighted_ask = sum(q / (idx + 1.0) for idx, (_, q) in enumerate(asks))
    weighted_denom = weighted_bid + weighted_ask
    weighted_imbalance = (
        (weighted_bid - weighted_ask) / weighted_denom
        if weighted_denom > 0 else None
    )

    bid20 = sum(q for _, q in bids)
    ask20 = sum(q for _, q in asks)
    if bid20 <= 0 or ask20 <= 0:
        concentration = 0.0
    else:
        concentration = (sum(q for _, q in bids[:5]) / bid20) - (sum(q for _, q in asks[:5]) / ask20)

    imbalance5 = _imbalance(bids, asks, 5)
    imbalance20 = _imbalance(bids, asks, 20)
    far_bid = sum(q for _, q in bids[5:20])
    far_ask = sum(q for _, q in asks[5:20])
    far_denom = far_bid + far_ask
    far_imbalance = ((far_bid - far_ask) / far_denom) if far_denom > 0 else None

    # Quantity-weighted price distance from the mid, expressed in bps.
    # This measures depth shape/liquidity geometry rather than direction alone.
    bid_slope = (
        sum(q * ((mid - p) / mid) * 10_000.0 for p, q in bids) / bid20
        if bid20 > 0 else None
    )
    ask_slope = (
        sum(q * ((p - mid) / mid) * 10_000.0 for p, q in asks) / ask20
        if ask20 > 0 else None
    )
    slope_denom = (
        (bid_slope + ask_slope)
        if bid_slope is not None and ask_slope is not None else None
    )
    slope_asymmetry = (
        (ask_slope - bid_slope) / slope_denom
        if slope_denom is not None and slope_denom > 0 else None
    )
    total_depth = bid20 + ask20
    near_depth = sum(q for _, q in bids[:5]) + sum(q for _, q in asks[:5])
    liquidity_concentration = near_depth / total_depth if total_depth > 0 else None

    bid_notional = sum(p * q for p, q in bids)
    ask_notional = sum(p * q for p, q in asks)
    notional_denom = bid_notional + ask_notional
    notional_imbalance = (
        (bid_notional - ask_notional) / notional_denom
        if notional_denom > 0 else None
    )

    values = {
        "depth_imbalance_5": _imbalance(bids, asks, 5),
        "depth_imbalance_10": _imbalance(bids, asks, 10),
        "depth_imbalance_20": _imbalance(bids, asks, 20),
        "weighted_depth_imbalance_20": weighted_imbalance,
        "spread_bps": spread_bps,
        "microprice_gap": microprice_gap,
        "depth_concentration_imbalance": concentration,
        "depth_notional_imbalance_20": notional_imbalance,
        "depth_imbalance_gradient_5_20": (
            (imbalance5 - imbalance20) / 15.0
            if imbalance5 is not None and imbalance20 is not None else None
        ),
        "near_far_imbalance_5_20": (
            imbalance5 - far_imbalance
            if imbalance5 is not None and far_imbalance is not None else None
        ),
        "bid_depth_slope_bps_20": bid_slope,
        "ask_depth_slope_bps_20": ask_slope,
        "depth_slope_asymmetry_20": slope_asymmetry,
        "depth_liquidity_concentration_20": liquidity_concentration,
    }
    if any(value is None or not math.isfinite(float(value)) for value in values.values()):
        return None
    return {key: float(value) for key, value in values.items()}
