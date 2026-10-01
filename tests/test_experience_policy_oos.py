import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

from src import experience_policy_oos as mod


def _row(i: int, correct: int, warning: bool = False) -> dict:
    base = datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(minutes=5 * i)
    return {
        "experience_id": i + 1,
        "prediction_id": i + 1,
        "horizon": "5m",
        "created_at_utc": base.isoformat(),
        "target_at_utc": (base + timedelta(minutes=5)).isoformat(),
        "model_version": "test.v1",
        "production_mode": "binance_primary",
        "regime": "TREND" if i % 2 else "RANGE",
        "predicted_direction": "UP" if i % 3 else "DOWN",
        "correct": int(correct),
        "confidence": 0.80 if warning else 0.45,
        "entropy": 0.40 if warning else 0.90,
        "margin": 0.35 if warning else 0.10,
        "warning_flags": json.dumps(["high_vol"] if warning else []),
        "data_quality_flags": json.dumps([]),
        "source": "binance_futures",
        "base_price": 100_000.0,
        "actual_price": 100_010.0,
        "probability_json": json.dumps({"DOWN": 0.10, "FLAT": 0.10, "UP": 0.80}),
        "feature_hash": "0" * 64,
        "settled_at_utc": (base + timedelta(seconds=1)).isoformat(),
    }


def test_features_exclude_realized_outcome_fields():
    row = _row(1, 0, warning=True)
    features = mod._features(row)
    for forbidden in (
        "correct",
        "actual_direction",
        "actual_price",
        "settled_at_utc",
        "source",
    ):
        assert forbidden not in features


def test_hierarchical_memory_backoff_uses_smoothed_global_prior():
    rows = [_row(i, int(i % 2 == 0), warning=(i % 3 == 0)) for i in range(20)]
    probabilities = mod._hierarchical_memory_predict(rows[:10], [rows[10]])
    assert len(probabilities) == 1
    assert 0.0 < probabilities[0] < 1.0


def test_prequential_returns_deferred_below_minimum():
    rows = [_row(i, i % 2) for i in range(50)]
    result = mod.prequential_evaluate(rows, min_train_rows=100)
    assert result["status"] == "DEFERRED"


def test_prequential_uses_only_prior_settled_experiences():
    rows = []
    for i in range(180):
        # Warning cases become harder, but this relationship is only visible
        # after their outcomes settle.
        warning = (i % 5) == 0
        correct = 0 if warning else int(i % 2 == 0)
        rows.append(_row(i, correct, warning=warning))

    result = mod.prequential_evaluate(
        rows,
        min_train_rows=100,
        thresholds=(0.55, 0.70),
        block_size=1,
        model_c=0.5,
        max_report_cases=20,
    )
    assert result["status"] == "OK"
    assert result["prequential_test_rows"] == 80
    assert result["model"]["trained_only_on_prior_settled_experiences"] is True
    assert result["model"]["model_fit_count"] > 0
    assert result["meta_error_probability"]["n"] == 80
    assert result["baseline_error_probability"]["n"] == 80
    assert result["hierarchical_experience_memory"]["n"] == 80
    assert "delta_logloss_memory_minus_baseline" in result
    assert "delta_brier_memory_minus_baseline" in result
    assert set(result["threshold_policy_candidates"]) == {"0.55", "0.70"}
    assert len(result["high_risk_cases_latest"]) <= 20


def test_prequential_is_deterministic():
    rows = [_row(i, int(i % 3 != 0), warning=(i % 7 == 0)) for i in range(150)]
    a = mod.prequential_evaluate(rows, min_train_rows=100, max_report_cases=10)
    b = mod.prequential_evaluate(rows, min_train_rows=100, max_report_cases=10)
    assert a == b


def test_build_is_research_only_and_does_not_mutate_production(tmp_path):
    db = tmp_path / "predictions.db"
    out = tmp_path / "experience_policy_oos.json"
    con = sqlite3.connect(db)
    try:
        con.execute(
            """CREATE TABLE experience_ledger (
                experience_id INTEGER PRIMARY KEY,
                prediction_id INTEGER NOT NULL,
                horizon TEXT NOT NULL,
                created_at_utc TEXT NOT NULL,
                target_at_utc TEXT NOT NULL,
                model_version TEXT NOT NULL,
                production_mode TEXT NOT NULL,
                regime TEXT,
                predicted_direction TEXT NOT NULL,
                actual_direction TEXT NOT NULL,
                correct INTEGER NOT NULL,
                confidence REAL,
                entropy REAL,
                margin REAL,
                warning_flags TEXT NOT NULL,
                data_quality_flags TEXT NOT NULL,
                source TEXT,
                base_price REAL,
                actual_price REAL,
                probability_json TEXT NOT NULL,
                feature_hash TEXT,
                settled_at_utc TEXT NOT NULL
            )"""
        )
        rows = [_row(i, int(i % 4 != 0), warning=(i % 6 == 0)) for i in range(130)]
        for row in rows:
            con.execute(
                """INSERT INTO experience_ledger VALUES (
                :experience_id,:prediction_id,:horizon,:created_at_utc,:target_at_utc,
                :model_version,:production_mode,:regime,:predicted_direction,'UP',
                :correct,:confidence,:entropy,:margin,:warning_flags,:data_quality_flags,
                :source,:base_price,:actual_price,:probability_json,:feature_hash,:settled_at_utc
                )""",
                row,
            )
        con.commit()
    finally:
        con.close()

    config = tmp_path / "config.json"
    config.write_text(
        json.dumps({
            "schema_version": 1,
            "min_train_rows": 100,
            "thresholds": [0.60],
            "max_report_cases": 5,
            "model_c": 0.5,
        }),
        encoding="utf-8",
    )
    report = mod.build(db, config_path=config, output_path=out)
    assert report["research_only"] is True
    assert report["production_changed"] is False
    assert report["promotion_evidence_eligible"] is False
    assert set(report["horizons"]) == {"5m", "10m"}
    obj = json.loads(out.read_text(encoding="utf-8"))
    assert obj["horizons"]["5m"]["status"] == "OK"
    assert obj["horizons"]["10m"]["status"] == "DEFERRED"
