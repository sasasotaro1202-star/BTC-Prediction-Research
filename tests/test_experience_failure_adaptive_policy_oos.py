import json
from datetime import datetime, timezone

import numpy as np

from src.experience_case_adaptive_controller_oos import (
    CLASSES,
    _case_key,
)
from src.experience_failure_adaptive_policy_oos import (
    ABSTAIN_THRESHOLD,
    MAX_SHRINK,
    MIN_COVERAGE,
    MIN_CASE_VALIDATION_SUPPORT,
    MIN_VALIDATION_SUPPORT,
    POLICY_GRID,
    RISK_REFRESH,
    RiskRecord,
    _apply_policy,
    _choose_case_or_global_policy,
    _choose_policy,
    _eligible_prior,
    _eligible_risk_records,
    _policy_stable_against_fixed,
    _fit_risk_model,
    _metrics,
    _oracle_policy,
    _predict_risk_model,
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
        "warning_flags": "",
        "data_quality_flags": "",
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
        case_key=("5m", "TREND", "UP", "0.70+", "binance_primary", "CLEAN"),
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
        validation.append(_record(i, risk=0.65 if high_risk else 0.45, correct=False if high_risk else True))
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


def test_policy_selection_filters_unsettled_risk_records():
    current = _record(100, risk=0.6, correct=False)
    current = RiskRecord(
        experience_id=current.experience_id,
        horizon=current.horizon,
        case_key=current.case_key,
        created_at_utc=datetime.fromtimestamp(100 * 60, tz=timezone.utc),
        settled_at_utc=datetime.fromtimestamp(100 * 60 + 300, tz=timezone.utc),
        y_index=current.y_index,
        base_probability=current.base_probability,
        baseline_error=current.baseline_error,
        error_risk=current.error_risk,
    )
    usable = _record(1)
    late_settlement = RiskRecord(
        experience_id=2,
        horizon="5m",
        case_key=usable.case_key,
        created_at_utc=datetime.fromtimestamp(2 * 60, tz=timezone.utc),
        settled_at_utc=current.created_at_utc,
        y_index=usable.y_index,
        base_probability=usable.base_probability,
        baseline_error=usable.baseline_error,
        error_risk=usable.error_risk,
    )
    assert [r.experience_id for r in _eligible_risk_records([usable, late_settlement], current)] == [1]


def test_policy_case_key_uses_shared_six_axis_identity():
    row = _row()
    assert len(_case_key(row)) == 6
    assert _case_key(row)[4] == "binance_primary"
    assert _case_key(row)[5] == "CLEAN"


def test_bounded_risk_model_refresh_is_finite():
    rows = [
        _row(
            experience_id=i,
            created=f"2026-09-{25 + i // 30:02d}T{(i % 24):02d}:{i % 60:02d}:00+00:00",
            settled=f"2026-09-{25 + i // 30:02d}T{(i % 24):02d}:{(i % 60 + 1):02d}:00+00:00",
            correct=i % 2,
            actual="UP" if i % 2 else "DOWN",
            direction="UP" if i % 2 else "DOWN",
        )
        for i in range(1, 121)
    ]
    model, default = _fit_risk_model(rows)
    assert RISK_REFRESH >= 5
    assert 0.0 <= default <= 1.0
    if model is not None:
        values = _predict_risk_model(model, rows[-5:], default)
        assert values.shape == (5,)
        assert np.isfinite(values).all()
        assert np.all((values >= 0.0) & (values <= 1.0))


def test_case_policy_uses_matured_history_when_case_support_is_sufficient():
    current = _record(999, risk=0.6)
    case_history = [
        _record(i, risk=0.85 if i <= 15 else 0.45, correct=(i > 15))
        for i in range(1, MIN_VALIDATION_SUPPORT + 1)
    ]
    other_history = [
        RiskRecord(
            experience_id=1000 + i,
            horizon="5m",
            case_key=("5m", "RANGE", "DOWN", "0.40-0.50", "binance_primary", "CLEAN"),
            created_at_utc=datetime.fromtimestamp((1000 + i) * 60, tz=timezone.utc),
            settled_at_utc=datetime.fromtimestamp((1000 + i) * 60 + 300, tz=timezone.utc),
            y_index=0,
            base_probability=(0.80, 0.10, 0.10),
            baseline_error=0.35,
            error_risk=0.55,
        )
        for i in range(1, MIN_VALIDATION_SUPPORT + 1)
    ]
    policy, detail = _choose_case_or_global_policy(case_history + other_history, current)
    assert policy in POLICY_GRID
    assert detail["source"] == "case_matured_failure_history"
    assert detail["case_support"] == MIN_CASE_VALIDATION_SUPPORT


def test_policy_stability_gate_rejects_time_localized_degradation():
    records = [
        _record(i, risk=0.75, correct=(i <= 40))
        for i in range(1, 81)
    ]
    candidate = (0.90, 0.15)
    assert _policy_stable_against_fixed(records, candidate) is False


def test_fixed_policy_passes_its_own_stability_gate():
    records = [
        _record(i, risk=0.75, correct=(i % 2 == 0))
        for i in range(1, 81)
    ]
    assert _policy_stable_against_fixed(records, (ABSTAIN_THRESHOLD, MAX_SHRINK)) is True


def test_oracle_policy_is_in_policy_grid_and_is_evaluation_only():
    records = [
        _record(i, risk=0.75, correct=(i % 3 != 0))
        for i in range(1, 81)
    ]
    policy, metrics = _oracle_policy(records)
    assert policy in POLICY_GRID
    assert metrics["n"] == 80
    assert np.isfinite(metrics["logloss"])
    assert np.isfinite(metrics["brier"])
