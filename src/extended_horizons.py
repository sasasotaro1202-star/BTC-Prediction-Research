"""Research-only forecasts for additional BTC horizons."""
from __future__ import annotations

import math
from typing import Mapping

import numpy as np

try:
    from horizon_registry import (
        EXTENDED_FORECAST_METHOD,
        EXTENDED_RESEARCH_HORIZONS,
        TARGET_DEFINITION_VERSION,
        target_at,
    )
except ModuleNotFoundError:
    from src.horizon_registry import (
        EXTENDED_FORECAST_METHOD,
        EXTENDED_RESEARCH_HORIZONS,
        TARGET_DEFINITION_VERSION,
        target_at,
    )

CLASSES = ("DOWN", "FLAT", "UP")


def _normalize(values) -> np.ndarray:
    p = np.asarray(values, dtype=float)
    if p.shape != (3,) or not np.isfinite(p).all() or np.any(p < 0) or float(p.sum()) <= 0:
        raise ValueError("extended_probability_input_invalid")
    p = np.clip(p, 1e-6, None)
    return p / p.sum()


def _class_probability_vector(values: Mapping[str, float] | object) -> np.ndarray:
    """Return DOWN/FLAT/UP probabilities from mappings or class-ordered arrays."""
    if isinstance(values, Mapping):
        raw = [values["DOWN"], values["FLAT"], values["UP"]]
    else:
        raw = np.asarray(values, dtype=float).reshape(-1).tolist()
    if len(raw) != 3:
        raise ValueError("class_probability_vector_invalid_shape")
    return np.asarray(raw, dtype=float)


def _time_scaled_probability(short: Mapping[str, float] | object, structural: Mapping[str, float], horizon: str) -> dict[str, float]:
    minutes = {
        "15m": 15, "30m": 30, "1h": 60, "3h": 180,
        "6h": 360, "12h": 720, "24h": 1440,
    }[horizon]
    scale = min(1.0, math.sqrt(10.0 / minutes))
    anchor_weight = 0.50 + 0.50 * scale
    anchor = _class_probability_vector(short)
    struct = _class_probability_vector(structural)
    raw = _normalize(anchor_weight * anchor + (1.0 - anchor_weight) * struct)
    out = _normalize((1.0 / 3.0) + scale * (raw - 1.0 / 3.0))
    return {c: float(out[i]) for i, c in enumerate(CLASSES)}


def forecast_extended_horizons(
    cutoff,
    features: Mapping[str, float],
    market: Mapping[str, float],
    p5: Mapping[str, float],
    p10: Mapping[str, float],
    structural_fn,
):
    """Generate explicit unvalidated research baselines without model promotion."""
    structural = structural_fn(features, market)
    short = _normalize([
        (float(p5["DOWN"]) + float(p10["DOWN"])) / 2.0,
        (float(p5["FLAT"]) + float(p10["FLAT"])) / 2.0,
        (float(p5["UP"]) + float(p10["UP"])) / 2.0,
    ])
    return {
        horizon: {
            "horizon": horizon,
            "target_at": target_at(cutoff, horizon),
            "target_definition_version": TARGET_DEFINITION_VERSION,
            "forecast_method": EXTENDED_FORECAST_METHOD,
            "research_only": True,
            "calibration_status": "UNVALIDATED_UNCALIBRATED",
            "probabilities": _time_scaled_probability(short, structural, horizon),
        }
        for horizon in EXTENDED_RESEARCH_HORIZONS
    }
