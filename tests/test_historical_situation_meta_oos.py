import numpy as np

from src.historical_situation_meta_oos import _causal_train, _vector


def _row(created, target):
    return {
        "created": created,
        "target": target,
        "y": "UP",
        "y10": "DOWN",
        "p5": {"DOWN": 0.2, "FLAT": 0.2, "UP": 0.6},
        "p10": {"DOWN": 0.4, "FLAT": 0.2, "UP": 0.4},
        "situation": {
            "trend_strength": 2.2,
            "volatility_expansion_ratio": 1.3,
            "normalized_entropy": 0.7,
            "probability_margin": 0.2,
            "trend_state": "TREND_UP",
            "volatility_state": "EXPANDING",
            "horizon_alignment": "AGREE",
            "signal_quality": "HIGH",
            "direction_5m": "UP",
            "direction_10m": "UP",
        },
    }


def test_historical_situation_vector_is_finite():
    x = _vector(_row("2026-09-25T00:00:00+00:00", "2026-09-25T00:05:00+00:00"))
    assert x.ndim == 1
    assert len(x) > 20
    assert np.isfinite(x).all()
    assert x[-1] == 1.0


def test_historical_situation_embargo_is_strict():
    start = "2026-09-25T02:00:00+00:00"
    rows = [
        _row("2026-09-25T00:00:00+00:00", "2026-09-25T00:05:00+00:00"),
        _row("2026-09-25T00:30:00+00:00", "2026-09-25T00:35:00+00:00"),
        _row("2026-09-25T01:00:00+00:00", "2026-09-25T01:05:00+00:00"),
    ]
    assert len(_causal_train(rows, start, "5m")) == 1
