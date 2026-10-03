import json

from src.experience_ledger import _experience_from_row


def test_extended_horizon_experience_is_recorded_after_settlement():
    row = {
        "prediction_id": 101,
        "created_at_utc": "2026-10-03T04:00:00+00:00",
        "target_15m": "2026-10-03T04:15:00+00:00",
        "model_version": "5m:v1|10m:v1",
        "p_down_15m": 0.2,
        "p_flat_15m": 0.3,
        "p_up_15m": 0.5,
        "actual_direction_15m": "UP",
        "settled_15m_at_utc": "2026-10-03T04:15:01+00:00",
        "settlement_source_15m": "binance",
        "actual_price_15m": 100500.0,
        "base_price": 100000.0,
        "scenario_json": json.dumps({
            "production_mode": "binance_primary",
            "regime": "TREND",
            "warnings": [],
            "data_quality": {"binance_futures": "ok"},
            "features": {"ret_1m": 0.001},
        }),
    }

    experience = _experience_from_row(row, "15m")

    assert experience is not None
    assert experience[0] == 101
    assert experience[1] == "15m"
    assert experience[3] == "2026-10-03T04:15:00+00:00"
    assert experience[7] == "UP"
    assert experience[8] == "UP"
    assert experience[9] == 1
    assert experience[15] == "binance"
