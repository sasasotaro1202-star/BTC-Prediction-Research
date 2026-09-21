import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from datetime import datetime, timedelta, timezone

import pytest

from exogenous_information import (
    InformationEvent,
    aggregate_event_features,
    pit_safe_events,
)


T0 = datetime(2026, 9, 21, 12, 0, tzinfo=timezone.utc)


def event(event_id, minutes, kind="news", sentiment=0.5, importance=1.0):
    published = T0 + timedelta(minutes=minutes)
    return InformationEvent(
        source="test",
        event_id=event_id,
        published_at=published,
        available_at=published,
        event_type=kind,
        sentiment=sentiment,
        importance=importance,
    )


def test_pit_excludes_future_information():
    events = [event("old", -5), event("future", 1)]
    safe = pit_safe_events(events, T0)
    assert [x.event_id for x in safe] == ["old"]


def test_delayed_availability_is_bound_to_available_at():
    delayed = InformationEvent(
        source="test",
        event_id="delayed",
        published_at=T0 - timedelta(minutes=10),
        available_at=T0 + timedelta(minutes=2),
        event_type="policy",
        importance=1.0,
    )
    assert pit_safe_events([delayed], T0) == []


def test_aggregate_is_bounded_and_separates_policy_and_macro():
    events = [
        event("news", -5, "news", sentiment=1.0, importance=1.0),
        event("policy", -4, "regulation", sentiment=-1.0, importance=0.5),
        event("macro", -3, "macro", sentiment=None, importance=0.25),
    ]
    features = aggregate_event_features(events, T0)
    assert features["news_event_count"] == 3
    assert features["policy_event_count"] == 1
    assert features["macro_event_count"] == 1
    assert -1.0 <= features["news_weighted_sentiment"] <= 1.0
    assert 0.0 <= features["information_importance"] <= 1.0


def test_malformed_timestamp_fails_closed():
    bad = InformationEvent(
        source="test",
        event_id="bad",
        published_at=T0.replace(tzinfo=None),
        available_at=T0,
        event_type="news",
    )
    with pytest.raises(ValueError):
        pit_safe_events([bad], T0)
