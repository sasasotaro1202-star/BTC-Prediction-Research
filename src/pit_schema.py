"""Strict point-in-time provenance validation for BTC research inputs.

This module is intentionally source-neutral. It does not fetch data and it does
not infer missing timestamps. Missing provenance is a validation failure for
strict OOS research.
"""
from __future__ import annotations

from datetime import datetime, timezone

REQUIRED = ("event_time", "available_at", "retrieved_at", "prediction_cutoff")
OPTIONAL = ("publication_time", "revision_time")


def parse_utc(value: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("timestamp must be a non-empty string")
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        raise ValueError("timestamp must include an explicit timezone")
    return dt.astimezone(timezone.utc)


def validate_provenance(record: dict) -> dict:
    """Validate a single PIT provenance envelope without filling missing data."""
    if not isinstance(record, dict):
        raise ValueError("provenance record must be a dict")

    parsed = {}
    for key in REQUIRED:
        if key not in record:
            raise ValueError(f"missing required PIT field: {key}")
        parsed[key] = parse_utc(record[key])

    for key in OPTIONAL:
        value = record.get(key)
        parsed[key] = None if value in (None, "") else parse_utc(value)

    event = parsed["event_time"]
    available = parsed["available_at"]
    retrieved = parsed["retrieved_at"]
    cutoff = parsed["prediction_cutoff"]
    publication = parsed["publication_time"]
    revision = parsed["revision_time"]

    if event > available:
        raise ValueError("event_time_after_available_at")
    if available > retrieved:
        raise ValueError("available_at_after_retrieved_at")
    if available > cutoff:
        raise ValueError("available_at_after_prediction_cutoff")
    if publication is not None and publication > available:
        raise ValueError("publication_time_after_available_at")
    if revision is not None and publication is not None and revision < publication:
        raise ValueError("revision_time_before_publication_time")

    return {key: (value.isoformat() if value is not None else None) for key, value in parsed.items()}
