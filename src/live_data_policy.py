"""Fail-closed policy for live BTC prediction inputs.

Primary Binance price history and required structural inputs remain hard
requirements. Cross-venue Bybit signals are optional because a single venue
outage must not disable an otherwise valid production prediction; missing
optional signals are explicitly recorded and never replaced with guesses.
"""
from __future__ import annotations

CRITICAL_STATUS_KEYS = (
    "binance_futures",
    "binance_depth",
    "binance_taker",
    "binance_premium",
)


def validate_live_inputs(status: dict, *, fut_rows: int, spot_rows: int, bybit_rows: int, allow_bybit_fallback: bool = False, allow_coinbase_fallback: bool = False) -> None:
    """Raise instead of predicting when production-critical inputs are incomplete."""
    if not isinstance(status, dict):
        raise ValueError("live_data_status_must_be_dict")
    fallback = status.get("price_feature_fallback") == "bybit"
    coinbase_fallback = status.get("price_feature_fallback") == "coinbase"
    if fallback or coinbase_fallback:
        if fallback and not allow_bybit_fallback:
            raise ValueError("live_prediction_inputs_incomplete:price_feature_fallback=bybit,fallback_not_allowed")
        if coinbase_fallback and not allow_coinbase_fallback:
            raise ValueError("live_prediction_inputs_incomplete:price_feature_fallback=coinbase,fallback_not_allowed")
        if (fallback and bybit_rows < 40) or (coinbase_fallback and fut_rows < 40):
            raise ValueError("live_prediction_inputs_incomplete:fallback_history_insufficient")
        return

    if fut_rows < 40:
        raise ValueError("binance_futures_contiguous_history_insufficient")

    failures = []
    for key in CRITICAL_STATUS_KEYS:
        value = status.get(key)
        if not isinstance(value, str) or value != "ok":
            failures.append(f"{key}={value!r}")

    if status.get("price_feature_fallback") != "none":
        failures.append(f"price_feature_fallback={status.get('price_feature_fallback')!r}")
    if failures:
        raise ValueError("live_prediction_inputs_incomplete:" + ",".join(failures))

    # Bybit is secondary cross-venue information. An outage is permitted, but
    # only as an explicitly missing signal. The predictor must omit its terms,
    # never synthesize them or use stale data.
    bybit_status = status.get("bybit_futures")
    if bybit_rows > 0 and bybit_status not in {"ok", "ok_current_only"}:
        raise ValueError(f"bybit_futures_status_inconsistent:{bybit_status!r}")
    if bybit_rows == 0 and bybit_status not in {
        "ok", "ok_current_only", "non_contiguous_or_insufficient",
        "error:missing_current_price",
    }:
        raise ValueError(f"bybit_futures_status_invalid:{bybit_status!r}")


def is_valid_status(status: dict, *, fut_rows: int, spot_rows: int, bybit_rows: int, allow_bybit_fallback: bool = False, allow_coinbase_fallback: bool = False) -> bool:
    try:
        validate_live_inputs(status, fut_rows=fut_rows, spot_rows=spot_rows, bybit_rows=bybit_rows, allow_bybit_fallback=allow_bybit_fallback, allow_coinbase_fallback=allow_coinbase_fallback)
    except ValueError:
        return False
    return True
