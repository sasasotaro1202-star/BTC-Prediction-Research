"""Canonical BTC forecast horizon registry.

5m/10m remain Production Champion horizons. Additional horizons are
research-only until they pass local PIT/OOS/robustness/holdout gates.
"""
from __future__ import annotations

from datetime import datetime, timezone

PRIMARY_HORIZONS = ("5m", "10m")
EXTENDED_RESEARCH_HORIZONS = ("15m", "30m", "1h", "3h", "6h", "12h", "24h")
ALL_HORIZONS = PRIMARY_HORIZONS + EXTENDED_RESEARCH_HORIZONS

HORIZON_MINUTES = {
    "5m": 5,
    "10m": 10,
    "15m": 15,
    "30m": 30,
    "1h": 60,
    "3h": 180,
    "6h": 360,
    "12h": 720,
    "24h": 1440,
}

TARGET_DEFINITION_VERSION = "direction_v1_neutral_bps"
EXTENDED_FORECAST_METHOD = "research_structural_time_scaled_v1"


def horizon_minutes(horizon: str) -> int:
    try:
        return int(HORIZON_MINUTES[horizon])
    except KeyError as exc:
        raise ValueError(f"unsupported_horizon:{horizon}") from exc


def target_at(cutoff: datetime, horizon: str) -> datetime:
    """Return the next UTC grid boundary for the requested horizon."""
    minutes = horizon_minutes(horizon)
    if cutoff.tzinfo is None:
        raise ValueError("cutoff_timezone_required")
    cutoff = cutoff.astimezone(timezone.utc)
    seconds = int(cutoff.timestamp())
    step = minutes * 60
    return datetime.fromtimestamp(((seconds // step) + 1) * step, timezone.utc)
