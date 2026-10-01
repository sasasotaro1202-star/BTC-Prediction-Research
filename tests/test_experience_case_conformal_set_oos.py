import json

import numpy as np

from src.experience_case_adaptive_controller_oos import CLASSES
from src.experience_case_conformal_set_oos import (
    MIN_CALIBRATION_SUPPORT,
    _calibration_pool,
    _conformal_quantile,
    _eligible_prior,
    _aps_score,
    prediction_set,
)


def _row(
    *,
    experience_id=1,
    horizon="5m",
    created="2026-10-01T00:00:00+00:00",
    settled="2026-10-01T00:05:00+00:00",
    regime="TREND",
    direction="UP",
    mode="binance_primary",
    p=(0.10, 0.10, 0.80),
    actual="UP",
):
    return {
        "experience_id": experience_id,
        "horizon": horizon,
        "created_at_utc": created,
        "settled_at_utc": settled,
        "regime": regime,
        "predicted_direction": direction,
        "production_mode": mode,
        "predicted_direction": direction,
        "warning_flags": "[]",
        "data_quality_flags": "[]",
        "probability_json": json.dumps(dict(zip(CLASSES, p))),
        "actual_direction": actual,
        "correct": int(direction == actual),
    }


def test_prediction_set_always_contains_at_least_one_class():
    p = np.asarray([0.05, 0.10, 0.85], dtype=float)
    out = prediction_set(p, 0.70)
    assert len(out) >= 1
    assert out[0] == "UP"


def test_high_threshold_can_expand_to_all_classes():
    p = np.asarray([0.34, 0.33, 0.33], dtype=float)
    out = prediction_set(p, 0.99)
    assert out == list(CLASSES)


def test_conformal_quantile_is_bounded_and_deterministic():
    scores = [0.2, 0.4, 0.7, 0.9]
    a = _conformal_quantile(scores, 0.20)
    b = _conformal_quantile(scores, 0.20)
    assert 0.0 <= a <= 1.0
    assert a == b


def test_eligible_prior_excludes_future_rows():
    past = [
        _row(
            experience_id=i,
            created=f"2026-10-01T00:{i:02d}:00+00:00",
            settled=f"2026-10-01T00:{i + 1:02d}:00+00:00",
            actual="UP",
        )
        for i in range(1, 20)
    ]
    future = [
        _row(
            experience_id=100 + i,
            created=f"2026-10-01T02:{i:02d}:00+00:00",
            settled=f"2026-10-01T02:{i + 1:02d}:00+00:00",
            actual="DOWN",
        )
        for i in range(1, 20)
    ]
    current = _row(
        experience_id=999,
        created="2026-10-01T01:00:00+00:00",
        settled="2026-10-01T01:05:00+00:00",
    )
    eligible = _eligible_prior(past + future, current)
    assert len(eligible) == len(past)


def test_calibration_pool_uses_hierarchical_backoff():
    rows = [
        _row(experience_id=i, actual="UP", direction="UP")
        for i in range(MIN_CALIBRATION_SUPPORT)
    ]
    current = _row(experience_id=999, mode="binance_primary")
    pool, level = _calibration_pool(rows, current)
    assert len(pool) >= MIN_CALIBRATION_SUPPORT
    assert level in {"exact_case", "without_production_mode", "horizon_regime_direction", "horizon_regime", "horizon", "global"}


def test_aps_score_is_between_zero_and_one():
    row = _row(actual="DOWN", p=(0.70, 0.20, 0.10))
    score = _aps_score(row)
    assert 0.0 <= score <= 1.0
