"""Fail-closed policy for live BTC prediction inputs.

A production directional prediction is permitted only when every input that
materially affects the current fusion path was actually retrieved successfully.
No zero/default/fallback substitution is accepted by this policy.
"""
from __future__ import annotations

CRITICAL_STATUS_KEYS = (
    "binance_futures",
    "binance_spot",
    "bybit_futures",
    "binance_depth",
    "bybit_depth",
    "binance_taker",
    "bybit_funding",
)


def validate_live_inputs(status: dict, *, fut_rows: int, spot_rows: int, bybit_rows: int) -> None:
    """Raise instead of predicting when required live inputs are incomplete."""
    if not isinstance(status, dict):
        raise ValueError("live_data_status_must_be_dict")

    if fut_rows < 40:
        raise ValueError("binance_futures_contiguous_history_insufficient")
    if spot_rows < 40:
        raise ValueError("binance_spot_contiguous_history_insufficient")
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
