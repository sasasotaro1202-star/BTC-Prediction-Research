import json
import sqlite3
from datetime import datetime, timedelta, timezone

from src.performance_change import _strict_pit_scores


def _scenario(created: datetime, valid: bool = True) -> str:
    created_s = created.isoformat()
    cutoff = (created - timedelta(seconds=1)).isoformat()
    retrieved = (created - timedelta(seconds=10)).isoformat()
    available = (created - timedelta(seconds=30)).isoformat()
    source = {
        "status": "ok",
        "available_at": available,
        "retrieved_at": retrieved,
        "prediction_cutoff": cutoff,
        "event_time": available,
    }
    sources = {
        "binance_futures": source,
        "binance_depth": source,
        "binance_taker": source,
        "binance_premium": source,
    }
    obj = {
        "production_mode": "binance_primary",
        "decision_time_utc": created_s,
        "provenance": {
            "available_at": available,
            "retrieved_at": retrieved,
            "prediction_cutoff": cutoff,
            "sources": sources,
        },
    }
    if not valid:
        obj["provenance"].pop("sources")
    return json.dumps(obj)


def _make_db(path):
    with sqlite3.connect(path) as con:
        con.execute(
            """
            CREATE TABLE predictions (
                created_at_utc TEXT,
                target_5m TEXT,
                actual_direction_5m TEXT,
                p_up_5m REAL,
                p_down_5m REAL,
                p_flat_5m REAL,
                target_10m TEXT,
                actual_direction_10m TEXT,
                p_up_10m REAL,
                p_down_10m REAL,
                p_flat_10m REAL,
                model_version TEXT,
                scenario_json TEXT
            )
            """
        )


def test_strict_pit_scores_exclude_non_pit_rows(tmp_path):
    db = tmp_path / "predictions.db"
    _make_db(db)
    created = datetime(2026, 1, 1, tzinfo=timezone.utc)
    target = (created + timedelta(minutes=5)).isoformat()

    with sqlite3.connect(db) as con:
        # Eligible strict-PIT row: certainty is valid and should count.
        con.execute(
            """
            INSERT INTO predictions VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                created.isoformat(), target, "UP",
                0.6, 0.2, 0.2,
                target, "UP", 0.6, 0.2, 0.2,
                "5m:v1|10m:v1", _scenario(created),
            ),
        )
        # Missing source provenance must fail closed.
        invalid_created = created + timedelta(minutes=1)
        con.execute(
            """
            INSERT INTO predictions VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                invalid_created.isoformat(),
                (invalid_created + timedelta(minutes=5)).isoformat(),
                "UP",
                0.9, 0.05, 0.05,
                (invalid_created + timedelta(minutes=10)).isoformat(),
                "UP", 0.9, 0.05, 0.05,
                "5m:v1|10m:v1", _scenario(invalid_created, valid=False),
            ),
        )
        # Fallback venue rows are valid research data but must not alter
        # the production Champion performance baseline.
        fallback_created = created + timedelta(minutes=3)
        fallback_scenario = _scenario(fallback_created).replace(
            '"production_mode": "binance_primary"',
            '"production_mode": "coinbase_fallback"',
        )
        con.execute(
            """
            INSERT INTO predictions VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                fallback_created.isoformat(),
                (fallback_created + timedelta(minutes=5)).isoformat(),
                "DOWN",
                0.05, 0.9, 0.05,
                (fallback_created + timedelta(minutes=10)).isoformat(),
                "DOWN", 0.05, 0.9, 0.05,
                "5m:coinbase_fallback.rf.v1|10m:coinbase_fallback.rf.v1",
                fallback_scenario,
            ),
        )

        # Prediction at the target timestamp must fail chronological PIT.
        late_created = created + timedelta(minutes=2)
        con.execute(
            """
            INSERT INTO predictions VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                late_created.isoformat(),
                late_created.isoformat(),
                "UP",
                0.99, 0.005, 0.005,
                (late_created + timedelta(minutes=10)).isoformat(),
                "UP", 0.99, 0.005, 0.005,
                "5m:v1|10m:v1", _scenario(late_created),
            ),
        )

    scores = _strict_pit_scores("5m", db)

    assert scores["n"] == 1
    assert scores["accuracy"] == 1.0
    assert scores["mode_counts"] == {"binance_primary": 1}
    assert 0.0 <= scores["logloss"] < 1.0
    assert 0.0 <= scores["brier"] < 1.0
    assert 0.0 <= scores["ece"] <= 1.0


def test_strict_pit_scores_are_binance_primary_only(tmp_path):
    db = tmp_path / "predictions.db"
    _make_db(db)
    created = datetime(2026, 1, 1, tzinfo=timezone.utc)
    with sqlite3.connect(db) as con:
        for i, mode in enumerate(("binance_primary", "coinbase_fallback"), start=1):
            at = created + timedelta(minutes=i)
            scenario = _scenario(at).replace(
                '"production_mode": "binance_primary"',
                f'"production_mode": "{mode}"',
            )
            con.execute(
                "INSERT INTO predictions VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    at.isoformat(),
                    (at + timedelta(minutes=5)).isoformat(),
                    "UP",
                    0.6, 0.2, 0.2,
                    (at + timedelta(minutes=10)).isoformat(),
                    "UP", 0.6, 0.2, 0.2,
                    f"5m:{mode}.v1|10m:{mode}.v1",
                    scenario,
                ),
            )
    scores = _strict_pit_scores("5m", db)
    assert scores["n"] == 1
    assert scores["mode_counts"] == {"binance_primary": 1}
