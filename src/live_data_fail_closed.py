"""Fail-closed validation for live prediction data quality.

The predictor may expose a degraded policy for exceptional conditions, but a
normal directional prediction must not be accepted when a production input
source failed and was replaced by a default/zero/fallback value.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from db import DB

# These sources directly affect the current production prediction/blend path.
# A missing source must therefore invalidate the newly generated directional row.
CRITICAL = (
    "binance_futures",
    "binance_spot",
    "bybit_futures",
    "binance_depth",
    "bybit_depth",
    "binance_taker",
    "bybit_funding",
)


def validate_latest() -> dict:
    with sqlite3.connect(DB) as con:
        row = con.execute(
            "SELECT prediction_id, model_version, scenario_json "
            "FROM predictions ORDER BY prediction_id DESC LIMIT 1"
        ).fetchone()
    if not row:
        raise SystemExit("live_data_fail_closed: no prediction exists")

    prediction_id, model_version, scenario_raw = row
    try:
        scenario = json.loads(scenario_raw or "{}")
    except Exception as exc:
        raise SystemExit(f"live_data_fail_closed: invalid scenario_json: {exc}") from exc

    if model_version == "DEGRADED_NO_FRESH_DATA":
        return {"ok": True, "prediction_id": prediction_id, "mode": "degraded"}

    quality = scenario.get("data_quality")
    if not isinstance(quality, dict):
        raise SystemExit("live_data_fail_closed: missing data_quality metadata")

    failures = []
    for key in CRITICAL:
        value = quality.get(key)
        if value in (None, "", "error", "unavailable", "missing"):
            failures.append(f"{key}={value!r}")
        elif isinstance(value, str) and value.startswith("error"):
            failures.append(f"{key}={value}")

    if quality.get("bybit_series_available") is not True:
        failures.append("bybit_series_available=False")

    if failures:
        raise SystemExit(
            "live_data_fail_closed: refusing directional prediction with missing/failed "
            "production inputs: " + ", ".join(failures)
        )

    return {"ok": True, "prediction_id": prediction_id, "mode": "normal"}


if __name__ == "__main__":
    print(json.dumps(validate_latest(), ensure_ascii=False))
