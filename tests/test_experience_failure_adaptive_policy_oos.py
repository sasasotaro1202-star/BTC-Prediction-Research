import json
from datetime import datetime, timezone

import numpy as np

from src.experience_case_adaptive_controller_oos import CLASSES
from src.experience_failure_adaptive_policy_oos import (
    ABSTAIN_THRESHOLD,
    MAX_SHRINK,
    MIN_COVERAGE,
    MIN_VALIDATION_SUPPORT,
    POLICY_GRID,
    RiskRecord,
    _apply_policy,
    _choose_policy,
    _eligible_prior,
    _metrics,
)


def _row(
    *,
    experience_id=1,
    created="2026-10-01T00:00:00+00:00",
    settled="2026-10-01T00:05:00+00:00",
    correct=1,
    actual="UP",
    direction="UP",
    regime="TREND",
    mode="binance_primary",
    p=(0.10, 0.10, 0.80),
):
    return {
        "experience_id": experience_id,
        "horizon": "5m",
        "created_at_utc": created,
        "settled_at_utc": settled,
        "regime": regime,
        "predicted_direction": direction,
        "production_mode": mode,
        "probability_json": json.dumps(dict(zip(CLASSES, p))),
        "correct": correct,
        "actual_direction": actual,
    }


def _record(i, *, risk=0.8, correct=True):
    return RiskRecord(
        experience_id=i,
        horizon="5m",
        case_key=("5m", "TREND", "UP", "0.70+", "binance_primary"),
        created_at_utc=datetime.fromtimestamp(i * 60, tz=timezone.utc),
        settled_at_utc=datetime.fromtimestamp(i * 60 + 300, tz=timezone.utc),
        y_index=2 if correct else 0,
        base_probability=(0.10, 0.10, 0.80),
        baseline_error=0.35,
        error_risk=risk,
    )


def test_eligible_prior_requires_created_and_settled_before_prediction():
    current = _row(
        experience_id=999,
        created="2026-10-01T01:00:00+00:00",
        settled="2026-10-01T01:05:00+00:00",
    )
    usable = _row(
        experience_id=1,
        created="2026-10-01T00:40:00+00:00",
        settled="2026-10-01T00:45:00+00:00",
    )
    unsettled = _row(
        experience_id=2,
        created="2026-10-01T00:40:00+00:00",
        settled="2026-10-01T01:10:00+00:00",
    )
    future_created = _row(
        experience_id=3,
        created="2026-10-01T01:10:00+00:00",
        settled="2026-10-01T01:15:00+00:00",
    )
    eligible = _eligible_prior([usable, unsettled, future_created], current)
    assert [r["experience_id"] for r in eligible] == [1]


def test_apply_policy_respects_threshold_and_shrink_cap():
    base = np.asarray([0.10, 0.10, 0.80], dtype=float)
    adjusted, action, shrink = _apply_policy(base, 0.95, 0.35, 0.80, 0.25)
    assert action == "ABSTAIN"
    assert shrink == 0.0
    assert np.allclose(adjusted, base)

    adjusted, action, shrink = _apply_policy(base, 0.70, 0.35, 0.80, 0.25)
    assert action == "SHRINK"
    assert 0.0 < shrink <= 0.25
    assert np.isclose(adjusted.sum(), 1.0)


def test_policy_grid_stays_within_existing_guardrails():
    assert len(POLICY_GRID) == 15
    assert all(threshold >= 0.70 for threshold, _ in POLICY_GRID)
    assert all(0.0 < shrink <= MAX_SHRINK for _, shrink in POLICY_GRID)


def test_policy_selection_fails_closed_when_support_is_short():
    policy, detail = _choose_policy([_record(i) for i in range(MIN_VALIDATION_SUPPORT - 1)])
    assert policy == (ABSTAIN_THRESHOLD, MAX_SHRINK)
    assert detail["source"] == "fixed_fallback"


def test_policy_selection_uses_matured_failure_history():
    validation = []
    for i in range(1, 81):
        high_risk = i <= 40
        validation.append(_record(i, risk=0.85 if high_risk else 0.45, correct=False if high_risk else True))
    policy, detail = _choose_policy(validation)
    assert detail["source"] == "matured_failure_history"
    assert detail["support"] == 80
    assert policy in POLICY_GRID
    assert detail["validation"]["coverage"] >= MIN_COVERAGE


def test_metrics_contract_is_finite():
    records = [_record(i, risk=0.4 + (i % 3) * 0.1, correct=(i % 2 == 0)) for i in range(1, 41)]
    metrics = _metrics(records, (ABSTAIN_THRESHOLD, MAX_SHRINK))
    assert metrics["n"] == 40
    assert 0.0 <= metrics["coverage"] <= 1.0
    assert np.isfinite(metrics["accuracy"])
    assert np.isfinite(metrics["logloss"])
    assert np.isfinite(metrics["brier"])
