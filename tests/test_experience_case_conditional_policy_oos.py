import json

import numpy as np

from src.experience_case_adaptive_controller_oos import CLASSES, _baseline_error
from src.experience_case_conditional_policy_oos import (
    CASE_RISK_THRESHOLD,
    MIN_CASE_SUPPORT,
    _case_correction,
    _case_outcome_prior,
    _eligible_prior,
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
    actual="DOWN",
    correct=0,
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
        "actual_direction": actual,
        "correct": correct,
    }


def test_case_outcome_prior_uses_direction_and_production_mode():
    matching = [
        _row(experience_id=i, actual="DOWN", correct=0, direction="UP", mode="binance_primary")
        for i in range(1, 31)
    ]
    other_mode = [
        _row(
            experience_id=100 + i,
            actual="UP",
            correct=1,
            direction="UP",
            mode="other_mode",
        )
        for i in range(1, 31)
    ]
    current = _row(
        experience_id=999,
        created="2026-10-01T01:00:00+00:00",
        settled="2026-10-01T01:05:00+00:00",
        actual="UP",
    )
    prior = matching + other_mode
    posterior, support, source = _case_outcome_prior(prior, current)
    assert support >= MIN_CASE_SUPPORT
    assert source.split("/")[0] == "5m"
    assert posterior[CLASSES.index("DOWN")] > posterior[CLASSES.index("UP")]


def test_future_rows_are_excluded_from_case_outcome_prior():
    past = [
        _row(
            experience_id=i,
            created=f"2026-10-01T00:{i:02d}:00+00:00",
            settled=f"2026-10-01T00:{i + 1:02d}:00+00:00",
            actual="UP",
            correct=1,
        )
        for i in range(1, 20)
    ]
    future = [
        _row(
            experience_id=100 + i,
            created=f"2026-10-01T02:{i:02d}:00+00:00",
            settled=f"2026-10-01T02:{i + 1:02d}:00+00:00",
            actual="DOWN",
            correct=0,
        )
        for i in range(1, 20)
    ]
    current = _row(
        experience_id=999,
        created="2026-10-01T01:00:00+00:00",
        settled="2026-10-01T01:05:00+00:00",
        actual="UP",
    )
    eligible = _eligible_prior(past + future, current)
    assert len(eligible) == len(past)
    before, _, _ = _case_outcome_prior(past, current)
    after, _, _ = _case_outcome_prior(past + future, current)
    np.testing.assert_allclose(before, after)


def test_case_correction_is_conservative_and_requires_support():
    base = np.asarray([0.05, 0.10, 0.85], dtype=float)
    case_prior = np.asarray([0.75, 0.15, 0.10], dtype=float)

    untouched, blend, action = _case_correction(
        base,
        case_prior,
        support=MIN_CASE_SUPPORT - 1,
        error_risk=0.90,
        baseline_error=0.40,
    )
    np.testing.assert_allclose(untouched, base)
    assert blend == 0.0
    assert action == "KEEP"

    corrected, blend, action = _case_correction(
        base,
        case_prior,
        support=MIN_CASE_SUPPORT * 5,
        error_risk=max(0.90, CASE_RISK_THRESHOLD + 0.1),
        baseline_error=0.40,
    )
    assert action == "CASE_CORRECTION"
    assert 0.0 < blend <= 0.35
    assert corrected[CLASSES.index("UP")] < base[CLASSES.index("UP")]
    assert corrected[CLASSES.index("DOWN")] > base[CLASSES.index("DOWN")]
    assert np.isclose(corrected.sum(), 1.0)


def test_case_prior_is_shrunk_toward_global_when_support_is_small():
    matching = [
        _row(experience_id=i, actual="DOWN", direction="UP", mode="binance_primary")
        for i in range(1, MIN_CASE_SUPPORT)
    ]
    global_history = [
        _row(
            experience_id=100 + i,
            horizon="10m",
            actual="UP",
            direction="DOWN",
            mode="other_mode",
            regime="RANGE",
            p=(0.80, 0.10, 0.10),
            correct=1,
        )
        for i in range(1, 40)
    ]
    current = _row(
        experience_id=999,
        created="2026-10-01T02:00:00+00:00",
        settled="2026-10-01T02:05:00+00:00",
    )
    posterior, support, _ = _case_outcome_prior(matching + global_history, current)
    assert support < MIN_CASE_SUPPORT
    global_error = _baseline_error(matching + global_history)
    assert 0.0 <= global_error <= 1.0
    assert np.isclose(posterior.sum(), 1.0)
