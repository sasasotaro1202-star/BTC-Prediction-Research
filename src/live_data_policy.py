"""Fail-closed policy for live BTC prediction inputs.

A production directional prediction is permitted only when every input that
materially affects the current fusion path was actually retrieved successfully.
Inputs that are fetched only for diagnostics/monitoring remain optional so an
unrelated venue/API outage cannot disable an otherwise valid prediction.
"""
from __future__ import annotations

# These sources are actually consumed by the current production fusion path.
# Do not make an unused diagnostic input a production blocker.
CRITICAL_STATUS_KEYS = (
    "binance_futures",
    "bybit_futures",
    "binance_depth",
    "bybit_depth",
    "binance_taker",
    "binance_premium",
)


def validate_live_inputs(status: dict, *, fut_rows: int, spot_rows: int, bybit_rows: int) -> None:
    """Raise instead of predicting when production-critical inputs are incomplete."""
    if not isinstance(status, dict):
        raise ValueError("live_data_status_must_be_dict")

    if fut_rows < 40:
        raise ValueError("binance_futures_contiguous_history_insufficient")
    if bybit_rows < 40:
        raise ValueError("bybit_futures_contiguous_history_insufficient")

    failures = []
    for key in CRITICAL_STATUS_KEYS:
        value = status.get(key)
        if not isinstance(value, str) or value != "ok":
            failures.append(f"{key}={value!r}")

    if status.get("price_feature_fallback") != "none":
        failures.append(f"price_feature_fallback={status.get('price_feature_fallback')!r}")

    if failures:
        raise ValueError("live_prediction_inputs_incomplete:" + ",".join(failures))


def is_valid_status(status: dict, *, fut_rows: int, spot_rows: int, bybit_rows: int) -> bool:
    try:
        validate_live_inputs(status, fut_rows=fut_rows, spot_rows=spot_rows, bybit_rows=bybit_rows)
    except ValueError:
        return False
    return True
