import json
import numpy as np

from src.experience_case_adaptive_controller_oos import CLASSES
from src.experience_predictability_router_oos import (
    CANDIDATES,
    MIN_TRAIN,
    VALIDATION_SIZE,
    MIN_CONSECUTIVE_SELECTIONS,
    _binary_logloss,
    _choose_source,
    _candidate_stable_gain,
    _eligible_prior,
    _meta_predictions,
    _risk,
    _select_source_from_scores,
    _apply_consecutive_selection_gate,
    evaluate_horizon,
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
    assert MIN_CONSECUTIVE_SELECTIONS >= 2

def test_meta_predictions_fail_closed_on_short_training_history():
    rows = [_row(experience_id=i, correct=i % 2, actual="UP" if i % 2 else "DOWN") for i in range(1, 12)]
    current = _row(experience_id=999, created="2026-10-01T01:00:00+00:00")
    values = _meta_predictions(rows, [current])
    assert values.shape == (1,)
    assert 0.0 <= values[0] <= 1.0

def test_router_evaluate_horizon_emits_case_metrics():
    rows = []
    for i in range(1, 241):
        minute = i % 60
        hour = i // 60
        created = f"2026-09-{25 + (hour // 24):02d}T{hour % 24:02d}:{minute:02d}:00+00:00"
        settled = f"2026-09-{25 + (hour // 24):02d}T{(hour % 24) + (1 if minute >= 55 else 0):02d}:{(minute + 5) % 60:02d}:00+00:00"
        correct = i % 2
        rows.append(
            _row(
                experience_id=i,
                created=created,
                settled=settled,
                correct=correct,
                actual="UP" if correct else "DOWN",
            )
        )
    result = evaluate_horizon(rows, "5m")
    assert result["status"] == "OK"
    assert result["pit_violation_count"] == 0
    assert "case_group_metrics" in result
    assert "chronological_blocks" in result
    assert result["chronological_blocks"]
    assert all(block["n"] >= 20 for block in result["chronological_blocks"])


def test_router_requires_material_relative_gain_before_switching():
    scores = {"global": 0.70, "case_memory": 0.697, "meta": 0.699}
    assert _select_source_from_scores(scores) == "global"


def test_router_allows_material_relative_gain():
    scores = {"global": 0.70, "case_memory": 0.68, "meta": 0.69}
    assert _select_source_from_scores(scores) == "case_memory"


def test_router_stable_gain_rejects_gain_only_in_one_validation_half():
    labels = np.asarray([1, 0, 1, 0, 1, 0, 1, 0], dtype=int)
    global_risk = np.asarray([0.2, 0.8, 0.2, 0.8, 0.2, 0.8, 0.2, 0.8], dtype=float)
    candidate_risk = np.asarray([0.1, 0.9, 0.1, 0.9, 0.4, 0.6, 0.4, 0.6], dtype=float)
    assert _candidate_stable_gain(
        labels,
        {"global": global_risk, "case_memory": candidate_risk, "meta": candidate_risk},
        "case_memory",
    ) is False


def test_router_requires_persistent_non_global_source_selection():
    assert _apply_consecutive_selection_gate("case_memory", None, 0) == ("global", "case_memory", 1)
    assert _apply_consecutive_selection_gate("case_memory", "case_memory", 1) == ("case_memory", "case_memory", 2)
    assert _apply_consecutive_selection_gate("global", "case_memory", 2) == ("global", None, 0)
    assert _apply_consecutive_selection_gate("meta", "case_memory", 1) == ("global", "meta", 1)
