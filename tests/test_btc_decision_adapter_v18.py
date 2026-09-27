from src.btc_decision_object_v18 import PredictionDecisionObject
from src.btc_decision_adapter_v18 import build_decision_from_situation_card
import pytest


def _events():
    return [
        {
            "event_id": "a",
            "source": "hyperliquid",
            "event_type": "trade",
            "event_time_ms": 900,
            "available_at_ms": 950,
            "retrieved_at_ms": 950,
            "payload_sha256": "ha",
            "payload": {"px": "1", "sz": "1"},
        },
        {
            "event_id": "b",
            "source": "bitget",
            "event_type": "l2_book",
            "event_time_ms": 800,
            "available_at_ms": 925,
            "retrieved_at_ms": 930,
            "payload_sha256": "hb",
            "payload": {"bids": [["1", "1"]], "asks": [["2", "1"]]},
        },
    ]


def _card():
    return {
        "prediction_cutoff_ms": 1_000,
        "used_event_ids": ["a", "b"],
        "freshness_ms": 100,
    }


def test_adapter_preserves_probability_and_builds_verified_pit():
    obj = build_decision_from_situation_card(
        cutoff_ms=1_000,
        probability={"up": 0.4, "flat": 0.2, "down": 0.4},
        situation_card=_card(),
        events=_events(),
    )
    assert isinstance(obj, PredictionDecisionObject)
    assert obj.probability == {"up": 0.4, "flat": 0.2, "down": 0.4}
    assert obj.pit_status == "verified"
    assert len(obj.provenance) == 2
    assert max(p["available_at_ms"] for p in obj.provenance) == 950
    obj.require_verified_pit()


def test_missing_event_provenance_fails_closed():
    with pytest.raises(ValueError, match="situation_card_missing_event_provenance"):
        build_decision_from_situation_card(
            cutoff_ms=1_000,
            probability={"up": 0.4, "flat": 0.2, "down": 0.4},
            situation_card=_card(),
            events=_events()[:1],
        )


def test_future_shadow_information_is_rejected():
    events = _events()
    events[0] = {**events[0], "available_at_ms": 1_001}
    with pytest.raises(ValueError, match="future_shadow_information"):
        build_decision_from_situation_card(
            cutoff_ms=1_000,
            probability={"up": 0.4, "flat": 0.2, "down": 0.4},
            situation_card={"prediction_cutoff_ms": 1_000, "used_event_ids": ["a"]},
            events=events,
        )


def test_empty_card_is_deferred_not_verified():
    obj = build_decision_from_situation_card(
        cutoff_ms=1_000,
        probability={"up": 0.4, "flat": 0.2, "down": 0.4},
        situation_card={"prediction_cutoff_ms": 1_000, "used_event_ids": []},
        events=[],
    )
    assert obj.pit_status == "deferred"
    with pytest.raises(ValueError, match="pit_not_verified"):
        obj.require_verified_pit()


def test_cutoff_mismatch_is_rejected():
    with pytest.raises(ValueError, match="situation_card_cutoff_mismatch"):
        build_decision_from_situation_card(
            cutoff_ms=1_000,
            probability={"up": 0.4, "flat": 0.2, "down": 0.4},
            situation_card={"prediction_cutoff_ms": 999, "used_event_ids": []},
            events=[],
        )


def test_duplicate_used_ids_are_rejected():
    with pytest.raises(ValueError, match="duplicate_situation_event_ids"):
        build_decision_from_situation_card(
            cutoff_ms=1_000,
            probability={"up": 0.4, "flat": 0.2, "down": 0.4},
            situation_card={"prediction_cutoff_ms": 1_000, "used_event_ids": ["a", "a"]},
            events=_events(),
        )