"""Research-only PIT-safe exogenous feature joining for BTC OOS studies.

No production feature schema is changed here. The join is deliberately causal:
an event can contribute only when available_at <= prediction_time, and only
within an explicit lookback window. Invalid provenance fails closed.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Iterable, Mapping

from exogenous_information import InformationEvent, aggregate_event_features, events_from_records


def events_for_prediction(
    events: Iterable[InformationEvent],
    prediction_time: datetime,
    lookback: timedelta = timedelta(hours=1),
) -> list[InformationEvent]:
    if prediction_time.tzinfo is None or prediction_time.utcoffset() is None:
        raise ValueError("prediction_time must be timezone-aware")
    if lookback.total_seconds() < 0:
        raise ValueError("lookback must be non-negative")
    parsed = list(events)
    # Validate every record, not only events that happen to fall in the window.
    for event in parsed:
        event.validate()
    lower = prediction_time - lookback
    return [
        event for event in parsed
        if lower <= event.available_at <= prediction_time
    ]


def aggregate_records_for_prediction(
    records: Iterable[Mapping[str, object]],
    prediction_time: datetime,
    lookback: timedelta = timedelta(hours=1),
) -> dict[str, float | int]:
    events = events_from_records(records)
    window = events_for_prediction(events, prediction_time, lookback)
    return aggregate_event_features(window, prediction_time)
