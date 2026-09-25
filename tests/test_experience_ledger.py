import json
import sqlite3
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import experience_ledger
from src.experience_ledger import _entropy, _prob_tuple, _stats


class RowLike(dict):
    pass


def test_entropy_uniform_is_one():
    assert abs(_entropy([1 / 3, 1 / 3, 1 / 3]) - 1.0) < 1e-12


def test_probability_tuple_is_normalized():
    row = {
        "p_down_5m": 2.0,
        "p_flat_5m": 3.0,
        "p_up_5m": 5.0,
    }
    p = _prob_tuple(row, "5m")
    assert p == [0.2, 0.3, 0.5]


def test_stats_counts_accuracy_by_prediction_and_actual():
    rows = [
        {"predicted_direction": "UP", "actual_direction": "UP", "correct": 1},
        {"predicted_direction": "UP", "actual_direction": "DOWN", "correct": 0},
        {"predicted_direction": "DOWN", "actual_direction": "DOWN", "correct": 1},
    ]
    out = _stats(rows)
    assert out["n"] == 3
    assert abs(out["accuracy"] - 2 / 3) < 1e-12
    assert out["by_predicted"]["UP"]["n"] == 2
    assert abs(out["by_predicted"]["UP"]["accuracy"] - 0.5) < 1e-12
    assert out["by_actual"]["DOWN"]["n"] == 2
    assert abs(out["by_actual"]["DOWN"]["hit_rate"] - 0.5) < 1e-12


def test_db_schema_contains_experience_ledger(tmp_path):
    db = tmp_path / "predictions.db"
    con = sqlite3.connect(db)
    try:
        con.execute(
            """CREATE TABLE experience_ledger (
              experience_id INTEGER PRIMARY KEY AUTOINCREMENT,
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
              settled_at_utc TEXT NOT NULL,
              UNIQUE(prediction_id, horizon)
            )"""
        )
        con.commit()
        columns = {r[1] for r in con.execute("PRAGMA table_info(experience_ledger)")}
        assert {"prediction_id", "horizon", "actual_direction", "correct", "confidence"} <= columns
    finally:
        con.close()


def _schema(con):
    con.executescript(
        """
        CREATE TABLE predictions (
          prediction_id INTEGER PRIMARY KEY,
          created_at_utc TEXT NOT NULL,
          target_5m TEXT NOT NULL,
          target_10m TEXT NOT NULL,
          base_price REAL NOT NULL,
          p_up_5m REAL NOT NULL, p_down_5m REAL NOT NULL, p_flat_5m REAL NOT NULL,
          p_up_10m REAL NOT NULL, p_down_10m REAL NOT NULL, p_flat_10m REAL NOT NULL,
          model_version TEXT NOT NULL,
          feature_json TEXT NOT NULL,
          scenario_json TEXT NOT NULL,
          actual_price_5m REAL, actual_price_10m REAL,
          actual_direction_5m TEXT, actual_direction_10m TEXT,
          correct_5m INTEGER, correct_10m INTEGER,
          settled_5m_at_utc TEXT, settled_10m_at_utc TEXT,
          settlement_source_5m TEXT, settlement_source_10m TEXT
        );
        CREATE TABLE experience_ledger (
          experience_id INTEGER PRIMARY KEY AUTOINCREMENT,
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
          settled_at_utc TEXT NOT NULL,
          UNIQUE(prediction_id, horizon)
        );
        """
    )


def _prediction(prediction_id, actual5="UP", actual10=None, p5=(0.10, 0.10, 0.80)):
    scenario = {
        "production_mode": "binance_primary",
        "regime": "trend_up",
        "warnings": ["high_vol"] if prediction_id == 1 else [],
        "data_quality": {"price_feature_fallback": "none"},
        "features": {"ret5": 0.001 * prediction_id, "rv10": 0.002},
    }
    return (
        prediction_id,
        "2026-09-25T18:00:00+00:00",
        "2026-09-25T18:05:00+00:00",
        "2026-09-25T18:10:00+00:00",
        100_000.0,
        p5[2], p5[0], p5[1],
        0.20, 0.40, 0.40,
        "bootstrap_rf.v1",
        "{}",
        json.dumps(scenario),
        100_010.0 if actual5 else None,
        100_000.0 if actual10 else None,
        actual5,
        actual10,
        int(actual5 == "UP") if actual5 else None,
        int(actual10 == "FLAT") if actual10 else None,
        "2026-09-25T18:05:01+00:00" if actual5 else None,
        "2026-09-25T18:10:01+00:00" if actual10 else None,
        "binance_futures" if actual5 else None,
        "binance_futures" if actual10 else None,
    )


def test_build_is_idempotent_and_records_case_context():
    with tempfile.TemporaryDirectory() as td:
        db = Path(td) / "predictions.db"
        summary = Path(td) / "experience_summary.json"
        con = sqlite3.connect(db)
        _schema(con)
        con.execute(
            "INSERT INTO predictions VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            _prediction(1),
        )
        con.execute(
            "INSERT INTO predictions VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            _prediction(2, actual5="DOWN", p5=(0.70, 0.20, 0.10)),
        )
        con.commit()
        con.close()

        with (
            patch.object(experience_ledger, "DB", db),
            patch.object(experience_ledger, "SUMMARY", summary),
            patch.object(experience_ledger, "init_db", lambda: None),
        ):
            first = experience_ledger.build()
            second = experience_ledger.build()

        assert first["inserted_since_last_run"] == 2
        assert second["inserted_since_last_run"] == 0
        assert first["total_experiences"] == second["total_experiences"] == 2
        assert first["parse_errors"] == second["parse_errors"] == 0

        obj = json.loads(summary.read_text(encoding="utf-8"))
        assert obj["research_only"] is True
        assert obj["production_changed"] is False

        con = sqlite3.connect(db)
        con.row_factory = sqlite3.Row
        rows = con.execute(
            "SELECT predicted_direction,actual_direction,correct,regime,warning_flags,feature_hash FROM experience_ledger ORDER BY prediction_id"
        ).fetchall()
        con.close()
        assert [(r["predicted_direction"], r["actual_direction"], r["correct"]) for r in rows] == [
            ("UP", "UP", 1),
            ("DOWN", "DOWN", 1),
        ]
        assert rows[0]["regime"] == "trend_up"
        assert rows[0]["warning_flags"] == '["high_vol"]'
        assert len(rows[0]["feature_hash"]) == 64


def test_unsettled_rows_are_excluded_and_invalid_probabilities_fail_closed():
    with tempfile.TemporaryDirectory() as td:
        db = Path(td) / "predictions.db"
        summary = Path(td) / "experience_summary.json"
        con = sqlite3.connect(db)
        _schema(con)
        con.execute(
            "INSERT INTO predictions VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            _prediction(1, actual5=None, actual10=None),
        )
        bad = list(_prediction(2, actual5="UP"))
        bad[5:8] = [0.0, 0.0, 0.0]
        con.execute(
            "INSERT INTO predictions VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            bad,
        )
        con.commit()
        con.close()

        with (
            patch.object(experience_ledger, "DB", db),
            patch.object(experience_ledger, "SUMMARY", summary),
            patch.object(experience_ledger, "init_db", lambda: None),
            pytest.raises(SystemExit, match="experience ledger parse errors: 1"),
        ):
            experience_ledger.build()

        report = json.loads(summary.read_text(encoding="utf-8"))
        assert report["total_experiences"] == 0
        assert report["parse_errors"] == 1
        assert report["horizons"]["5m"]["total"]["n"] == 0
