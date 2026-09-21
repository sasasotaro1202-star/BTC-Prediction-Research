"""Leakage-safe interaction features for BTC short-horizon research.

This module only combines already-observed features. It deliberately avoids
automatic Cartesian expansion: interactions are a small, predeclared set of
economically interpretable combinations that can be ablated and validated
chronologically before production promotion.
"""
from __future__ import annotations

from math import isfinite


def _mul(a: float, b: float) -> float:
    x = float(a) * float(b)
    return x if isfinite(x) else 0.0


def interaction_features(f: dict) -> dict:
    """Return a conservative, predeclared set of second-order interactions.

    All inputs are contemporaneous/lagged features already available at the
    prediction cutoff. No target, future return, or post-cutoff information is
    accessed here.
    """
    required = (
        "ret_1m", "ret_5m", "ret_15m", "ret_30m",
        "volatility_5m", "volatility_10m", "volume_ratio",
        "volume_trend", "ema_gap_5m", "ema_gap_10m",
    )
    missing = [k for k in required if k not in f]
    if missing:
        raise KeyError("missing_interaction_inputs:" + ",".join(missing))

    return {
        # Momentum conditioned on the current volatility regime.
        "ix_ret1_vol10": _mul(f["ret_1m"], f["volatility_10m"]),
        "ix_ret5_vol10": _mul(f["ret_5m"], f["volatility_10m"]),
        "ix_trend_vol": _mul(
            0.50 * f["ret_5m"] + 0.30 * f["ret_15m"] + 0.20 * f["ret_30m"],
            f["volatility_10m"],
        ),
        # Trend agreement across short and higher time scales.
        "ix_ema_agreement": _mul(f["ema_gap_5m"], f["ema_gap_10m"]),
        "ix_momentum_agreement": _mul(f["ret_5m"], f["ret_15m"]),
        "ix_momentum_extension": _mul(f["ret_15m"], f["ret_30m"]),
        # Activity-conditioned price behaviour.
        "ix_ret1_volume": _mul(f["ret_1m"], f["volume_ratio"]),
        "ix_ret5_volume": _mul(f["ret_5m"], f["volume_ratio"]),
        "ix_volume_vol": _mul(f["volume_ratio"] - 1.0, f["volatility_10m"]),
        "ix_volume_trend_vol": _mul(
            f["volume_trend"] - 1.0, f["volatility_5m"]
        ),
    }


def add_interactions(f: dict) -> dict:
    """Copy a feature mapping and append the interaction candidates."""
    out = dict(f)
    out.update(interaction_features(f))
    return out
