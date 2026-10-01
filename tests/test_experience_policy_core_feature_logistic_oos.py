from datetime import datetime, timedelta, timezone

from src import experience_policy_core_feature_logistic_oos as mod


def _row(i: int, correct: int):
    ts = datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(minutes=i)
    return {
        "experience_id": i + 1,
        "prediction_id": i + 1,
        "horizon": "5m",
        "created_at_utc": ts.isoformat(),
        "regime": "TREND" if i % 2 else "RANGE",
        "predicted_direction": "UP" if i % 3 else "DOWN",
        "correct": int(correct),
        "confidence": 0.80 if i % 5 == 0 else 0.45,
        "entropy": 0.40 if i % 5 == 0 else 0.90,
        "margin": 0.35 if i % 5 == 0 else 0.10,
        "settled_at_utc": (ts + timedelta(seconds=1)).isoformat(),
    }


def test_core_features_exclude_source_runtime_and_outcome_fields():
    row = _row(1, 0)
    features = mod._core_features(row)
    assert set(features) == {
        "confidence",
        "entropy",
        "margin",
        "hour_sin",
        "hour_cos",
        "regime",
        "predicted_direction",
    }


def test_core_feature_oos_is_research_only():
    rows = [_row(i, int(i % 3 != 0)) for i in range(115)]
    result = mod.evaluate_rows(rows, min_train_rows=100, test_block_rows=1)
    assert result["status"] == "OK"
    assert result["prequential_test_rows"] == 15
    assert result["model"]["feature_profile"] == "core_prediction_time"
    assert result["model"]["class_weight"] is None
    assert result["model"]["trained_only_on_prior_settled_experiences"] is True
    assert result["model"]["same_settlement_timestamp_isolation"] is True
    assert result["research_only"] is True
    assert result["production_changed"] is False
    assert result["promotion_evidence_eligible"] is False


def test_core_feature_oos_is_deterministic():
    rows = [_row(i, int(i % 5 != 0)) for i in range(115)]
    a = mod.evaluate_rows(rows, min_train_rows=100, max_report_cases=10)
    b = mod.evaluate_rows(rows, min_train_rows=100, max_report_cases=10)
    assert a == b
