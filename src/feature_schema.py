"""Canonical BTC production feature schema.

Keep feature names and order in one place so training, research, runtime
prediction, and artifact validation cannot silently drift apart.
"""
from __future__ import annotations

FEATURES = (
    "ret_1m",
    "ret_3m",
    "ret_5m",
    "ret_10m",
    "acceleration",
    "volatility_5m",
    "volatility_10m",
    "range_position_10m",
    "body_1m",
    "upper_wick_1m",
    "lower_wick_1m",
    "volume_ratio",
    "volume_trend",
    "ema_gap_5m",
    "ema_gap_10m",
)

FEATURE_COUNT = len(FEATURES)
