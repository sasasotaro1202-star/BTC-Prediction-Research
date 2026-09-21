"""Pure helpers for research-only exogenous source adapters."""
from __future__ import annotations
from datetime import datetime, timezone

def _parse_seen_time(value: str) -> datetime | None:
    try:
        return datetime.strptime(value, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
    except (TypeError, ValueError):
        return None
