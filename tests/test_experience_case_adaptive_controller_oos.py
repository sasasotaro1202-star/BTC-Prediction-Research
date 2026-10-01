import json
from datetime import datetime, timezone, timedelta

import numpy as np

from src.experience_case_adaptive_controller_oos import (
    CLASSES,
    _adjust_probabilities,
    _baseline_error,
    _case_key,
    _information_state,
    _hierarchical_prior,
)


def _row(
    *,
    experience_id=1,
    horizon="5m",
    created="2026-10-01T00:00:00+00:00",
    settled="2026-10-01T00:05:00+00:00",
    regime="TREND",
    direction="UP",
    p=(0.10, 0.10, 0.80),
    correct=1,
    mode="binance_primary",
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


def test_case_key_is_horizon_specific():
    a = _row(horizon="5m")
    b = _row(horizon="10m")
    assert _case_key(a)[0] == "5m"
    assert _case_key(b)[0] == "10m"
    assert _case_key(a) != _case_key(b)


def test_case_key_separates_production_mode():
    primary = _row(mode="binance_primary")
    shadow = _row(mode="shadow_candidate")
    assert _case_key(primary)[-1] == "binance_primary"
    assert _case_key(shadow)[-1] == "shadow_candidate"
    assert _case_key(primary) != _case_key(shadow)


def test_hierarchical_memory_uses_matching_case_history():
    matching = [_row(experience_id=i, correct=0) for i in range(1, 31)]
    other = [
        _row(
            experience_id=100 + i,
            regime="RANGE",
            direction="DOWN",
            p=(0.80, 0.10, 0.10),
            correct=1,
        )
        for i in range(1, 31)
    ]
    target = _row(
        experience_id=999,
        created="2026-10-01T00:10:00+00:00",
        settled="2026-10-01T00:15:00+00:00",
    )
    estimate = _hierarchical_prior(matching + other, target)
    assert estimate > _baseline_error(matching + other)


def test_high_risk_shrinks_or_abstains_but_never_sharpens():
    base = np.asarray([0.05, 0.10, 0.85], dtype=float)
    adjusted, action, shrink = _adjust_probabilities(base, 0.75, 0.50)
    assert action == "SHRINK"
    assert 0.0 < shrink <= 0.35
    assert adjusted[2] < base[2]
    assert np.isclose(adjusted.sum(), 1.0)


def test_extreme_risk_abstains_without_mutating_probabilities():
    base = np.asarray([0.20, 0.30, 0.50], dtype=float)
    adjusted, action, shrink = _adjust_probabilities(base, 0.90, 0.50)
    assert action == "ABSTAIN"
    assert shrink == 1.0
    np.testing.assert_allclose(adjusted, base / base.sum())


def test_case_memory_does_not_depend_on_future_rows():
    t0 = datetime(2026, 10, 1, tzinfo=timezone.utc)
    past = [
        _row(
            experience_id=i,
            created=(t0 + timedelta(minutes=i)).isoformat(),
            settled=(t0 + timedelta(minutes=i + 5)).isoformat(),
            correct=1,
        )
        for i in range(1, 20)
    ]
    future = [
        _row(
            experience_id=100 + i,
            created=(t0 + timedelta(minutes=100 + i)).isoformat(),
            settled=(t0 + timedelta(minutes=100 + i + 5)).isoformat(),
            correct=0,
        )
        for i in range(1, 20)
    ]
    target = _row(
        experience_id=999,
        created=(t0 + timedelta(minutes=50)).isoformat(),
        settled=(t0 + timedelta(minutes=55)).isoformat(),
    )
    before = _hierarchical_prior(past, target)
    with_future = _hierarchical_prior(past + future, target)
    assert before == with_future

def test_information_state_separates_clean_warning_and_degraded():
    clean = _row()
    warned = _row(
        warning_flags='["order-book imbalance"]',
    )
    degraded = _row(
        warning_flags='["partial market-data coverage; confidence reduced"]',
    )
    assert _information_state(clean) == "CLEAN"
    assert _information_state(warned) == "WARNED"
    assert _information_state(degraded) == "DEGRADED"
    assert len(_case_key(clean)) == 6
    assert _case_key(clean) != _case_key(warned)
    assert _case_key(warned) != _case_key(degraded)
