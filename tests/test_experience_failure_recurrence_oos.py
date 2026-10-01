import json

import numpy as np

from src.experience_case_adaptive_controller_oos import CLASSES
from src.experience_failure_recurrence_oos import (
    MIN_CASE_SUPPORT,
    _eligible_prior,
    _recurrence_features,
    _fit_prequential,
)


def _row(
    *,
    experience_id=1,
    created="2026-10-01T00:00:00+00:00",
    settled="2026-10-01T00:05:00+00:00",
    horizon="5m",
    regime="TREND",
    direction="UP",
    mode="binance_primary",
    p=(0.10, 0.10, 0.80),
    correct=1,
):
    return {
        "experience_id": experience_id,
        "horizon": horizon,
        "created_at_utc": created,
        "settled_at_utc": settled,
        "regime": regime,
        "predicted_direction": direction,
        "production_mode": mode,
        "warning_flags": "[]",
        "data_quality_flags": "[]",
        "probability_json": json.dumps(dict(zip(CLASSES, p))),
        "correct": correct,
    }


def test_recurrence_features_detect_recent_same_case_failure_streak():
    prior = [
        _row(
            experience_id=i,
            created=f"2026-10-01T00:{i:02d}:00+00:00",
            settled=f"2026-10-01T00:{i + 1:02d}:00+00:00",
            correct=0 if i >= 6 else 1,
        )
        for i in range(1, 11)
    ]
    current = _row(
        experience_id=99,
        created="2026-10-01T01:00:00+00:00",
        settled="2026-10-01T01:05:00+00:00",
    )
    feat = _recurrence_features(prior, current)
    assert feat["case_support"] == 10
    assert feat["recent_error_rate"] > 0.4
    assert feat["error_streak"] == 5
    assert 0.0 < feat["failure_recency_decay"] <= 1.0
    assert feat["case_has_failure"] == 1.0


def test_recurrence_features_ignore_future_experiences():
    past = [
        _row(
            experience_id=i,
            created=f"2026-10-01T00:{i:02d}:00+00:00",
            settled=f"2026-10-01T00:{i + 1:02d}:00+00:00",
            correct=1,
        )
        for i in range(1, 10)
    ]
    future = [
        _row(
            experience_id=100 + i,
            created=f"2026-10-01T02:{i:02d}:00+00:00",
            settled=f"2026-10-01T02:{i + 1:02d}:00+00:00",
            correct=0,
        )
        for i in range(1, 10)
    ]
    current = _row(
        experience_id=999,
        created="2026-10-01T01:00:00+00:00",
        settled="2026-10-01T01:05:00+00:00",
    )
    before = _recurrence_features(past, current)
    after = _recurrence_features(past + future, current)
    assert before == after
    assert len(_eligible_prior(past + future, current)) == len(past)


def test_prequential_recurrence_model_is_fail_closed_on_insufficient_data():
    prior = [
        _row(
            experience_id=i,
            created=f"2026-10-01T00:{i:02d}:00+00:00",
            settled=f"2026-10-01T00:{i + 1:02d}:00+00:00",
            correct=i % 2,
        )
        for i in range(1, 20)
    ]
    current = _row(
        experience_id=999,
        created="2026-10-01T01:00:00+00:00",
        settled="2026-10-01T01:05:00+00:00",
    )
    p, fitted = _fit_prequential(prior, current)
    assert fitted is False
    assert 0.0 <= p <= 1.0


def test_support_floor_is_positive_and_stable():
    assert MIN_CASE_SUPPORT >= 1
    current = _row(
        experience_id=999,
        created="2026-10-01T01:00:00+00:00",
        settled="2026-10-01T01:05:00+00:00",
    )
    empty = _recurrence_features([], current)
    assert empty["case_support"] == 0.0
    assert np.isfinite(list(empty.values())).all()
