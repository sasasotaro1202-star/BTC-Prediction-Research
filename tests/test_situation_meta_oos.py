import numpy as np
from datetime import datetime, timedelta, timezone

from src.situation_meta_oos import (
    build_meta_vector,
    causal_train_rows,
    meta_feature_names,
    primary_live_scenario,
)

def _row(created, target):
    return {
        "created": created,
        "target": target,
        "production": [0.2, 0.3, 0.5],
        "microstructure": {"book_imbalance": 0.2, "taker_imbalance_5m": 0.1},
        "situation": {
            "normalized_entropy": 0.7,
            "probability_margin": 0.2,
            "trend_state": "TREND_UP",
            "volatility_state": "EXPANDING",
            "orderflow_state": "BUY_PRESSURE",
            "signal_quality": "HIGH",
            "horizon_alignment": "AGREE",
        },
    }

def test_meta_vector_contract():
    vector = build_meta_vector(_row("2026-09-25T00:00:00+00:00", "2026-09-25T00:05:00+00:00"))
    assert vector.ndim == 1
    assert len(vector) == len(meta_feature_names())
    assert np.isfinite(vector).all()
    assert vector[:3].tolist() == [0.2, 0.3, 0.5]

def test_missingness_is_explicit():
    row = _row("2026-09-25T00:00:00+00:00", "2026-09-25T00:05:00+00:00")
    row["microstructure"] = {}
    vector = build_meta_vector(row)
    names = meta_feature_names()
    assert vector[names.index("missing=book_imbalance")] == 1.0

def test_causal_training_embargo():
    start = datetime(2026, 9, 25, tzinfo=timezone.utc)
    rows = [
        _row((start - timedelta(minutes=80)).isoformat(), (start - timedelta(minutes=65)).isoformat()),
        _row((start - timedelta(minutes=70)).isoformat(), (start - timedelta(minutes=30)).isoformat()),
        _row((start - timedelta(minutes=70)).isoformat(), (start + timedelta(minutes=5)).isoformat()),
    ]
    assert len(causal_train_rows(rows, start.isoformat(), "5m")) == 1

def test_primary_live_scenario_is_fail_closed():
    assert primary_live_scenario({"production_mode": "binance_primary"}) is True
    assert primary_live_scenario({"production_mode": "bybit_fallback"}) is False
    assert primary_live_scenario({}) is False
