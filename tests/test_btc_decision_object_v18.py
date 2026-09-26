from dataclasses import replace

import pytest

from src.btc_decision_object_v18 import PredictionDecisionObject, build_research_decision


def _base(**overrides):
    payload = dict(
        prediction_time_ms=1_000,
        target="direction",
        horizon="5m",
        granularity="1m",
        probability={"up": 0.4, "flat": 0.2, "down": 0.4},
        model="SoftVotingEnsemble",
        strategy="baseline",
        pit_status="verified",
        provenance=({"source": "binance", "available_at_ms": 999},),
    )
    payload.update(overrides)
    return build_research_decision(**payload)


def test_full_v18_fields_round_trip_and_stable_id():
    obj = _base(
        predictability=0.7,
        confidence=0.6,
        uncertainty=0.3,
        current_regime="range",
        future_regime="high_vol",
        disagreement=0.2,
        error_correlation=0.1,
        future_failure=0.15,
        time_to_failure_ms=30_000,
        ood=False,
        novelty=0.2,
        tail_risk=0.25,
        forecast_lifetime_ms=60_000,
        freshness_ms=500,
        revision_risk=0.2,
        update_need=0.3,
        next_update_time_ms=2_000,
        information_value=0.4,
        action="maintain",
        scenarios=({"name": "shock", "probability": 0.1},),
        branches=({"name": "A", "probability": 0.9},),
    )
    restored = PredictionDecisionObject.from_dict(obj.to_dict())
    assert restored.prediction_id == obj.prediction_id
    assert restored.highest_probability_class == "up"
    assert restored.confidence_gap == 0.0


def test_future_provenance_is_rejected():
    with pytest.raises(ValueError, match="future_information_in_provenance"):
        _base(provenance=({"source": "bad", "available_at_ms": 1_001},))


def test_verified_pit_requires_provenance():
    with pytest.raises(ValueError, match="verified_pit_requires_provenance"):
        _base(provenance=())


def test_probability_mismatch_is_rejected():
    obj = _base()
    with pytest.raises(ValueError, match="probability_distribution_mismatch"):
        replace(obj, probability={"up": 0.5, "down": 0.5})


def test_next_update_before_prediction_is_rejected():
    with pytest.raises(ValueError, match="next_update_before_prediction"):
        _base(next_update_time_ms=999)


def test_pit_gate_is_fail_closed():
    obj = _base(pit_status="deferred")
    with pytest.raises(ValueError, match="pit_not_verified"):
        obj.require_verified_pit()
