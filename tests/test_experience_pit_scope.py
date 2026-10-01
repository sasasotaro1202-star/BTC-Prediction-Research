import json

from src.experience_pit_scope import _strict_prediction_ok


def _row(*, valid=True, target10_after=True):
    source = {
        "status": "ok",
        "event_time": "2026-10-01T00:50:00+00:00",
        "available_at": "2026-10-01T00:55:00+00:00",
        "retrieved_at": "2026-10-01T00:59:00+00:00",
        "prediction_cutoff": "2026-10-01T01:00:00+00:00",
    }
    provenance = {
        "event_time": "2026-10-01T00:50:00+00:00",
        "available_at": "2026-10-01T00:55:00+00:00",
        "retrieved_at": "2026-10-01T00:59:00+00:00",
        "prediction_cutoff": "2026-10-01T01:00:00+00:00",
        "sources": {"binance_futures": source},
    }
    if not valid:
        provenance = {}
    return {
        "actual_direction_5m": "UP",
        "actual_direction_10m": "UP",
        "target_5m": "2026-10-01T01:05:00+00:00",
        "target_10m": (
            "2026-10-01T01:10:00+00:00"
            if target10_after else "2026-10-01T01:04:00+00:00"
        ),
        "created_at_utc": "2026-10-01T01:00:00+00:00",
        "scenario_json": json.dumps({"provenance": provenance}),
    }


def test_strict_primary_accepts_valid_binance_provenance():
    ok, reason = _strict_prediction_ok(_row(), "5m")
    assert ok is True
    assert reason == "ok"


def test_strict_primary_rejects_missing_provenance():
    ok, reason = _strict_prediction_ok(_row(valid=False), "5m")
    assert ok is False
    assert reason == "missing_top_level_provenance"


def test_strict_primary_rejects_non_chronological_10m_target():
    ok, reason = _strict_prediction_ok(_row(target10_after=False), "10m")
    assert ok is False
    assert reason == "10m_target_not_after_5m_target"
