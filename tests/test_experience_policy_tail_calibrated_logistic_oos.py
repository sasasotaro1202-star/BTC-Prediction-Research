from datetime import datetime, timedelta, timezone

from src import experience_policy_tail_calibrated_logistic_oos as mod


def _row(i: int, correct: int):
    ts = datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(minutes=i)
    return {
        "experience_id": i + 1,
        "prediction_id": i + 1,
        "horizon": "5m",
        "created_at_utc": ts.isoformat(),
        "target_at_utc": (ts + timedelta(minutes=5)).isoformat(),
        "model_version": "test.v1",
        "production_mode": "binance_primary",
        "regime": "TREND" if i % 2 else "RANGE",
        "predicted_direction": "UP" if i % 3 else "DOWN",
        "correct": int(correct),
        "confidence": 0.80 if i % 5 == 0 else 0.45,
        "entropy": 0.40 if i % 5 == 0 else 0.90,
        "margin": 0.35 if i % 5 == 0 else 0.10,
        "warning_flags": "[]",
        "data_quality_flags": "[]",
        "settled_at_utc": (ts + timedelta(seconds=1)).isoformat(),
    }


def test_tail_calibrated_oos_is_research_only():
    rows = [_row(i, int(i % 3 != 0)) for i in range(115)]
    result = mod.evaluate_rows(
        rows,
        min_train_rows=100,
        test_block_rows=1,
        calibration_tail_fraction=0.20,
        min_calibration_rows=20,
    )
    assert result["status"] == "OK"
    assert result["prequential_test_rows"] == 15
    assert result["model"]["probability_calibration"] == "temporal_tail_sigmoid"
    assert result["model"]["class_weight"] is None
    assert result["model"]["trained_only_on_prior_settled_experiences"] is True
    assert result["model"]["calibration_applied_count"] > 0
    assert result["research_only"] is True
    assert result["production_changed"] is False
    assert result["promotion_evidence_eligible"] is False


def test_tail_calibration_split_respects_settlement_boundary(monkeypatch):
    rows = [_row(i, 1) for i in range(8)]
    shared_ts = rows[2]["settled_at_utc"]
    rows[3]["settled_at_utc"] = shared_ts
    seen = []

    def fake_fit(train_rows, test_rows, **kwargs):
        seen.append(
            ([r["experience_id"] for r in train_rows], [r["experience_id"] for r in test_rows])
        )
        return [0.5] * len(test_rows), True, True

    monkeypatch.setattr(mod, "_fit_tail_calibrated", fake_fit)
    out = mod.evaluate_rows(rows, min_train_rows=2, test_block_rows=1, min_calibration_rows=1)
    assert out["status"] == "OK"
    assert any(test_ids == [3, 4] and train_ids == [1, 2] for train_ids, test_ids in seen)
    assert any(test_ids == [5] and train_ids[:4] == [1, 2, 3, 4] for train_ids, test_ids in seen)


def test_tail_calibrated_oos_is_deterministic():
    rows = [_row(i, int(i % 5 != 0)) for i in range(115)]
    a = mod.evaluate_rows(rows, min_train_rows=100, max_report_cases=10)
    b = mod.evaluate_rows(rows, min_train_rows=100, max_report_cases=10)
    assert a == b
