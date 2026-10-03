"""Chart-ready canonical multi-horizon forecast curve for BTC.

The curve is a presentation/observability layer only. It does not change the
5m/10m production decision path and never uses future outcomes.
"""
from __future__ import annotations

import math
from datetime import datetime
from typing import Mapping

try:
    from horizon_registry import ALL_HORIZONS, HORIZON_MINUTES
except ModuleNotFoundError:
    from src.horizon_registry import ALL_HORIZONS, HORIZON_MINUTES

CLASSES = ("DOWN", "FLAT", "UP")
PRODUCTION_HORIZONS = frozenset(("5m", "10m"))


def _normalize(values: Mapping[str, float]) -> dict[str, float]:
    raw = []
    for cls in CLASSES:
        try:
            value = float(values[cls])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid_probability:{cls}") from exc
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"invalid_probability:{cls}")
        raw.append(value)
    total = sum(raw)
    if not math.isfinite(total) or total <= 0:
        raise ValueError("invalid_probability_sum")
    normalized = [value / total for value in raw]
    return {cls: float(normalized[i]) for i, cls in enumerate(CLASSES)}


def _point(
    horizon: str,
    probabilities: Mapping[str, float],
    target_at: datetime,
    *,
    research_only: bool,
    calibration_status: str,
    previous: Mapping[str, float] | None,
) -> dict:
    p = _normalize(probabilities)
    top = max(CLASSES, key=p.get)
    delta_prev = (
        {cls: float(p[cls] - previous[cls]) for cls in CLASSES}
        if previous is not None
        else None
    )
    return {
        "horizon": horizon,
        "minutes": int(HORIZON_MINUTES[horizon]),
        "target_at": target_at.isoformat(),
        "state": "RESEARCH_ONLY" if research_only else (
            "PRIMARY_5M" if horizon == "5m" else "PRIMARY_10M"
        ),
        "research_only": bool(research_only),
        "calibration_status": calibration_status,
        "top_prediction": top,
        "probabilities": p,
        "delta_from_previous": delta_prev,
    }


def build_horizon_curve(
    prediction_cutoff: datetime,
    base_price: float,
    p5: Mapping[str, float],
    p10: Mapping[str, float],
    extended_forecasts: Mapping[str, Mapping],
    target5: datetime,
    target10: datetime,
) -> dict:
    """Build the complete ordered 5m -> 24h forecast term structure."""
    if prediction_cutoff.tzinfo is None:
        raise ValueError("prediction_cutoff_timezone_required")
    try:
        price = float(base_price)
    except (TypeError, ValueError) as exc:
        raise ValueError("base_price_invalid") from exc
    if not math.isfinite(price) or price <= 0:
        raise ValueError("base_price_invalid")

    required_extended = tuple(ALL_HORIZONS[2:])
    missing = [h for h in required_extended if h not in extended_forecasts]
    if missing:
        raise ValueError(f"missing_extended_horizons:{','.join(missing)}")

    points = []
    previous = None
    for horizon in ALL_HORIZONS:
        if horizon == "5m":
            probabilities, target_at = p5, target5
            research_only = False
            calibration_status = "PRODUCTION_BUNDLE_CALIBRATION"
        elif horizon == "10m":
            probabilities, target_at = p10, target10
            research_only = False
            calibration_status = "PRODUCTION_BUNDLE_CALIBRATION"
        else:
            item = extended_forecasts[horizon]
            probabilities = item.get("probabilities")
            target_at = item.get("target_at")
            if not isinstance(target_at, datetime):
                raise ValueError(f"invalid_target_timestamp:{horizon}")
            research_only = True
            calibration_status = str(
                item.get("calibration_status", "UNVALIDATED_UNCALIBRATED")
            )

        point = _point(
            horizon,
            probabilities,
            target_at,
            research_only=research_only,
            calibration_status=calibration_status,
            previous=previous,
        )
        point["delta_from_5m"] = {
            cls: float(point["probabilities"][cls] - points[0]["probabilities"][cls])
            for cls in CLASSES
        } if points else {cls: 0.0 for cls in CLASSES}
        points.append(point)
        previous = point["probabilities"]

    return {
        "schema_version": 1,
        "generated_at_utc": prediction_cutoff.isoformat(),
        "prediction_cutoff": prediction_cutoff.isoformat(),
        "base_price": price,
        "primary_horizon": "5m",
        "primary_horizons": ["5m", "10m"],
        "horizon_order": list(ALL_HORIZONS),
        "points": points,
        "purpose": "chart_ready_multi_horizon_forecast_term_structure",
        "future_outcomes_used": False,
    }
