import json
import numpy as np

from src.experience_case_adaptive_controller_oos import CLASSES
from src.experience_predictability_router_oos import (
    CANDIDATES,
    MIN_TRAIN,
    VALIDATION_SIZE,
    _binary_logloss,
    _choose_source,
    _eligible_prior,
    _risk,
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
        "warning_flags": "[]",
        "data_quality_flags": "[]",
        "probability_json": json.dumps(dict(zip(CLASSES, p))),
        "correct": correct,
        "actual_direction": actual,
    }


def test_router_candidates_are_explicit():
    assert CANDIDATES == ("global", "case_memory", "meta")


def test_router_eligible_prior_excludes_future_rows():
    past = [
        _row(
            experience_id=i,
            created=f"2026-10-01T00:{i:02d}:00+00:00",
            settled=f"2026-10-01T00:{i+1:02d}:00+00:00",
        )
        for i in range(1, 10)
    ]
    future = [
        _row(
            experience_id=100+i,
            created=f"2026-10-01T02:{i:02d}:00+00:00",
            settled=f"2026-10-01T02:{i+1:02d}:00+00:00",
            correct=0,
            actual="DOWN",
            direction="UP",
        )
        for i in range(1, 10)
    ]
    current = _row(
        experience_id=999,
        created="2026-10-01T01:00:00+00:00",
        settled="2026-10-01T01:05:00+00:00",
    )
    eligible = _eligible_prior(past + future, current)
    assert len(eligible) == len(past)


def test_router_fails_closed_to_global_when_validation_history_is_short():
    rows = [_row(experience_id=i) for i in range(MIN_TRAIN)]
    current = _row(experience_id=999, created="2026-10-01T01:00:00+00:00")
    source, scores, n = _choose_source(rows, current)
    assert source == "global"
    assert n == 0
    assert all(np.isnan(v) for v in scores.values())


def test_risk_sources_are_finite_and_bounded():
    rows = [_row(experience_id=i, correct=i % 2, actual="UP" if i % 2 else "DOWN") for i in range(1, 12)]
    current = _row(experience_id=999, created="2026-10-01T01:00:00+00:00")
    for candidate in CANDIDATES:
        value = _risk(candidate, rows, current)
        assert 0.0 <= value <= 1.0
        assert np.isfinite(value)


def test_binary_logloss_is_finite():
    y = np.asarray([0, 1, 1], dtype=int)
    p = np.asarray([0.2, 0.8, 0.6], dtype=float)
    value = _binary_logloss(y, p)
    assert np.isfinite(value)
    assert value > 0


def test_router_constants_keep_causal_validation_window():
    assert MIN_TRAIN >= 100
    assert VALIDATION_SIZE >= 40
