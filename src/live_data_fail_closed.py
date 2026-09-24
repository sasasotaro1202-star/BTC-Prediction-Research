"""Fail-closed validation for live prediction data quality.

The predictor may use an explicitly trained, source-specific fallback model when
the primary venue is unavailable. Such a fallback is accepted only when the
prediction records an explicit production_mode/policy and provenance for that
fallback source. It is never silently treated as a normal Binance prediction.
"""
from __future__ import annotations

import json
import os
import sqlite3

from db import DB

CRITICAL = (
    "binance_futures",
    "binance_depth",
    "binance_taker",
    "binance_premium",
)

FALLBACK_MODES = {
    "bybit_fallback": "bybit",
    "coinbase_fallback": "coinbase",
}


def _validate_explicit_fallback(prediction_id: int, scenario: dict, model_version: str) -> dict | None:
    mode = scenario.get("production_mode")
    policy = scenario.get("policy")
    source = FALLBACK_MODES.get(mode)
    if source is None:
        return None

    accepted_policy = f"{source}_fallback_model+fallback_oos_calibration"
    legacy_policy = f"{source}_fallback_model_only_uncalibrated"
    if policy not in {accepted_policy, legacy_policy}:
        raise SystemExit(
            f"live_data_fail_closed: fallback policy mismatch for {mode}: {policy!r}"
        )

    provenance = scenario.get("provenance")
    sources = provenance.get("sources") if isinstance(provenance, dict) else None
    source_key = f"{source}_futures"
    source_meta = sources.get(source_key) if isinstance(sources, dict) else None
    if not isinstance(source_meta, dict) or source_meta.get("status") != "ok":
        raise SystemExit(
            f"live_data_fail_closed: explicit fallback provenance missing/invalid: {source_key}"
        )

    if not isinstance(model_version, str) or f"{source}_fallback" not in model_version:
        raise SystemExit(
            f"live_data_fail_closed: fallback model binding missing for {mode}"
        )

    return {"ok": True, "prediction_id": prediction_id, "mode": mode}


def validate_latest() -> dict:
    if os.environ.get("BTC_LIVE_DEFERRED") == "1":
        return {"ok": True, "mode": "deferred"}

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

    fallback_result = _validate_explicit_fallback(prediction_id, scenario, model_version)
    if fallback_result is not None:
        return fallback_result

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

    if failures:
        raise SystemExit(
            "live_data_fail_closed: refusing directional prediction with missing/failed "
            "production inputs: " + ", ".join(failures)
        )

    return {"ok": True, "prediction_id": prediction_id, "mode": "normal"}


if __name__ == "__main__":
    print(json.dumps(validate_latest(), ensure_ascii=False))
