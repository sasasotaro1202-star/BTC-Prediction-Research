"""Research-only exogenous BTC information contracts.

This module does not change production prediction inputs. It defines the
minimum provenance/PIT contract for future macro, policy, regulatory, and news
features so they cannot be silently introduced with look-ahead leakage.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable, Mapping


@dataclass(frozen=True)
class InformationEvent:
    source: str
    event_id: str
    published_at: datetime
    available_at: datetime
    event_type: str
    importance: float = 0.0
    sentiment: float | None = None
    surprise: float | None = None

    def validate(self) -> None:
        if not self.source or not self.event_id or not self.event_type:
            raise ValueError("information event identity is incomplete")
        for value in (self.published_at, self.available_at):
            if value.tzinfo is None or value.utcoffset() is None:
                raise ValueError("information timestamps must be timezone-aware")
        if self.available_at < self.published_at:
            raise ValueError("available_at cannot precede published_at")
        if not 0.0 <= float(self.importance) <= 1.0:
            raise ValueError("importance must be in [0, 1]")
        if self.sentiment is not None and not -1.0 <= float(self.sentiment) <= 1.0:
            raise ValueError("sentiment must be in [-1, 1]")


def pit_safe_events(
    events: Iterable[InformationEvent],
    prediction_time: datetime,
) -> list[InformationEvent]:
    """Return only information provably available by prediction_time.

    Unknown/malformed provenance fails closed rather than being imputed.
    """
    if prediction_time.tzinfo is None or prediction_time.utcoffset() is None:
        raise ValueError("prediction_time must be timezone-aware")
    safe: list[InformationEvent] = []
    for event in events:
        event.validate()
        if event.available_at <= prediction_time:
            safe.append(event)
    return safe


def aggregate_event_features(
    events: Iterable[InformationEvent],
    prediction_time: datetime,
) -> dict[str, float | int]:
    """Create bounded research diagnostics without touching production models."""
    safe = pit_safe_events(events, prediction_time)
    if not safe:
        return {
            "news_event_count": 0,
            "news_weighted_sentiment": 0.0,
            "policy_event_count": 0,
            "macro_event_count": 0,
            "information_importance": 0.0,
        }
    weights = [max(0.0, min(1.0, float(e.importance))) for e in safe]
    weight_sum = sum(weights)
    sentiment = [
        float(e.sentiment) * w
        for e, w in zip(safe, weights)
        if e.sentiment is not None
    ]
    return {
        "news_event_count": len(safe),
        "news_weighted_sentiment": (
            sum(sentiment) / weight_sum if weight_sum > 0 and sentiment else 0.0
        ),
        "policy_event_count": sum(
            e.event_type in {"policy", "regulation"} for e in safe
        ),
        "macro_event_count": sum(e.event_type == "macro" for e in safe),
        "information_importance": (
            sum(weights) / len(weights) if weights else 0.0
        ),
    }


def source_metadata(event: InformationEvent) -> Mapping[str, str]:
    """Return immutable provenance fields for future research artifacts."""
    return {
        "source": event.source,
        "event_id": event.event_id,
        "published_at": event.published_at.astimezone(timezone.utc).isoformat(),
        "available_at": event.available_at.astimezone(timezone.utc).isoformat(),
        "event_type": event.event_type,
    }
