from datetime import datetime, timedelta, timezone, timezone

import pytest

from exogenous_feature_join import aggregate_records_for_prediction, events_for_prediction
from exogenous_information import InformationEvent


T0 = datetime(2026, 9, 21, 12, 0, tzinfo=timezone.utc)


def _event(event_id, published_at, available_at=None, kind="news"):
    return InformationEvent(
        source="test",
        event_id=event_id,
        published_at=published_at,
        available_at=available_at or published_at,
        event_type=kind,
        importance=1.0,
        sentiment=0.5,
    )


def test_join_excludes_future_and_stale_events():
    events = [
        _event("recent", T0 - timedelta(minutes=30)),
        _event("future", T0 + timedelta(minutes=1)),
        _event("stale", T0 - timedelta(hours=2)),
    ]
    selected = events_for_prediction(events, T0, timedelta(hours=1))
    assert [x.event_id for x in selected] == ["recent"]


def test_join_uses_available_time_not_publication_time():
    delayed = _event(
        "delayed",
        T0 - timedelta(minutes=30),
        T0 + timedelta(minutes=5),
        "policy",
    )
    assert events_for_prediction([delayed], T0) == []


def test_join_fails_closed_on_bad_event_even_if_it_would_be_outside_window():
    bad = _event("bad", T0 - timedelta(hours=2))
    bad = InformationEvent(
        source=bad.source,
        event_id=bad.event_id,
        published_at=bad.published_at.replace(tzinfo=None),
        available_at=bad.available_at,
        event_type=bad.event_type,
    )
    with pytest.raises(ValueError):
        events_for_prediction([bad], T0)


def test_record_join_is_deterministic_and_bounded():
    records = [{
        "source": "test",
        "event_id": "r1",
        "published_at": "2026-09-21T11:30:00+00:00",
        "available_at": "2026-09-21T11:30:00+00:00",
        "event_type": "news",
        "importance": 1.0,
        "sentiment": 0.5,
    }]
    f = aggregate_records_for_prediction(records, T0)
    assert f["news_event_count"] == 1
    assert -1.0 <= f["news_weighted_sentiment"] <= 1.0
