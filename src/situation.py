
"""Deterministic live BTC market-situation summarizer.

This module is descriptive/routing metadata only: it never changes prediction
probabilities. It uses only information already available at prediction time.
"""
from __future__ import annotations

import math
from typing import Mapping


CLASSES = ("DOWN", "FLAT", "UP")


def _finite(value, default=0.0):
    try:
        v = float(value)
        return v if math.isfinite(v) else float(default)
    except (TypeError, ValueError):
        return float(default)


def _entropy(probs: Mapping[str, float]) -> float:
    vals = [_finite(probs.get(k), 0.0) for k in CLASSES]
    total = sum(max(0.0, v) for v in vals)
    if total <= 0.0:
        return 1.0
    vals = [max(1e-9, v / total) for v in vals]
    h = -sum(v * math.log(v) for v in vals) / math.log(3.0)
    return max(0.0, min(1.0, h))


def _margin(probs: Mapping[str, float]) -> float:
    vals = sorted((_finite(probs.get(k), 0.0) for k in CLASSES), reverse=True)
    return max(0.0, min(1.0, vals[0] - vals[1]))


def summarize_situation(features: Mapping[str, float], market: Mapping[str, float], p5: Mapping[str, float], p10: Mapping[str, float], *, data_quality: Mapping[str, object] | None = None) -> dict:
    """Return stable situation metadata without mutating model probabilities."""
    volatility = max(_finite(features.get("volatility_10m"), 0.0), 1e-8)
    trend = _finite(features.get("trend_alignment"), 0.0)
    trend_strength = abs(trend) / volatility
    if trend_strength >= 2.0:
        trend_state = "TREND_UP" if trend > 0 else "TREND_DOWN"
    else:
        trend_state = "RANGE"

    vol5 = max(_finite(features.get("volatility_5m"), 0.0), 1e-8)
    expansion_ratio = volatility / vol5
    if expansion_ratio >= 1.20:
        volatility_state = "EXPANDING"
    elif expansion_ratio <= 0.83:
        volatility_state = "COMPRESSING"
    else:
        volatility_state = "STABLE"

    d5 = max(p5, key=lambda k: _finite(p5.get(k), 0.0))
    d10 = max(p10, key=lambda k: _finite(p10.get(k), 0.0))
    horizon_alignment = "AGREE" if d5 == d10 else "CONFLICT"
    e5 = _entropy(p5)
    e10 = _entropy(p10)
    entropy = (e5 + e10) / 2.0
    margin = (_margin(p5) + _margin(p10)) / 2.0
    signal_quality = (
        "HIGH" if entropy <= 0.78 and margin >= 0.15
        else "MEDIUM" if entropy <= 0.88 and margin >= 0.05
        else "LOW"
    )

    book = _finite(market.get("book_imbalance"), 0.0)
    taker = _finite(market.get("taker_imbalance"), 0.0)
    funding = _finite(market.get("funding_binance"), 0.0)
    gap = _finite(market.get("cross_exchange_gap"), 0.0) if market.get("cross_exchange_gap") is not None else 0.0
    flow = (
        "BUY_PRESSURE" if book >= 0.20 or taker >= 0.35
        else "SELL_PRESSURE" if book <= -0.20 or taker <= -0.35
        else "BALANCED"
    )
    divergence = "HIGH" if abs(gap) >= 0.0005 else "NORMAL"

    quality_errors = 0
    if isinstance(data_quality, Mapping):
        for key in ("binance_futures", "binance_depth", "binance_taker", "binance_premium"):
            value = data_quality.get(key)
            if isinstance(value, str) and value.startswith("error:"):
                quality_errors += 1
    data_state = "DEGRADED" if quality_errors >= 2 else "PARTIAL" if quality_errors == 1 else "HEALTHY"

    # Situation quality is allowed to degrade when the two horizons conflict
    # or the underlying live inputs are incomplete. This changes descriptive
    # situation metadata only; prediction probabilities remain untouched.
    quality_rank = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
    if horizon_alignment == "CONFLICT" and signal_quality == "HIGH":
        signal_quality = "MEDIUM"
    if data_state == "PARTIAL" and quality_rank[signal_quality] > quality_rank["MEDIUM"]:
        signal_quality = "MEDIUM"
    elif data_state == "DEGRADED":
        # Preserve the established MEDIUM classification for conflicting
        # horizons when data is degraded; only a high-confidence, horizon-
        # aligned signal is forced to LOW. This is descriptive metadata only.
        if signal_quality == "HIGH":
            signal_quality = "MEDIUM" if horizon_alignment == "CONFLICT" else "LOW"

    state = "|".join((trend_state, volatility_state, horizon_alignment, signal_quality))
    return {
        "market_state": state,
        "trend_state": trend_state,
        "volatility_state": volatility_state,
        "volatility_expansion_ratio": expansion_ratio,
        "trend_strength": trend_strength,
        "horizon_alignment": horizon_alignment,
        "direction_5m": d5,
        "direction_10m": d10,
        "entropy_5m": e5,
        "entropy_10m": e10,
        "normalized_entropy": entropy,
        "probability_margin": margin,
        "signal_quality": signal_quality,
        "orderflow_state": flow,
        "book_imbalance": book,
        "taker_imbalance": taker,
        "funding_binance": funding,
        "cross_exchange_divergence": divergence,
        "data_state": data_state,
    }
