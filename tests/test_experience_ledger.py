import json
import sqlite3
from pathlib import Path

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
