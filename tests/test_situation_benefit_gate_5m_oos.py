import numpy as np

from src.situation_benefit_gate_5m_oos import (
    GATE_THRESHOLD,
    _apply,
    _features,
    _fit_gate,
)


def _rows(n):
    out = []
    for i in range(n):
        out.append(
            {
                "p5": {"DOWN": 0.7, "FLAT": 0.2, "UP": 0.1},
                "p10": {"DOWN": 0.6, "FLAT": 0.2, "UP": 0.2},
                "situation": {
                    "trend_strength": 1.0,
                    "volatility_expansion_ratio": 1.1,
                    "normalized_entropy": 0.8,
                    "probability_margin": 0.5,
                    "ret_5m": 0.0,
                    "ret_15m": 0.0,
                    "ret_30m": 0.0,
                    "volatility_5m": 0.01,
                    "volatility_10m": 0.01,
                    "range_position_10m": 0.5,
                    "range_position_30m": 0.5,
                    "ema_gap_5m": 0.0,
                    "ema_gap_10m": 0.0,
                    "trend_state": "RANGE",
                    "volatility_state": "STABLE",
                    "horizon_alignment": "AGREE",
                    "signal_quality": "MEDIUM",
                    "direction_5m": "DOWN",
                    "direction_10m": "DOWN",
                },
            }
        )
    return out


def test_features_are_finite():
    rows = _rows(4)
    base = np.tile([0.7, 0.2, 0.1], (4, 1))
    cand = np.tile([0.2, 0.2, 0.6], (4, 1))
    x = _features(rows, base, cand)
    assert x.ndim == 2
    assert x.shape[0] == 4
    assert np.isfinite(x).all()


def test_gate_is_fail_closed_without_training():
    rows = _rows(1)
    base = np.asarray([[0.7, 0.2, 0.1]], dtype=float)
    cand = np.asarray([[0.2, 0.2, 0.6]], dtype=float)
    out, use, prob = _apply(None, rows, base, cand)
    assert np.allclose(out, base)
    assert not bool(use[0])
    assert prob[0] == 0.0
    assert GATE_THRESHOLD == 0.65


def test_fit_gate_requires_two_classes():
    rows = _rows(2)
    base = np.tile([0.7, 0.2, 0.1], (2, 1))
    cand = np.tile([0.2, 0.2, 0.6], (2, 1))
    x = _features(rows, base, cand)
    assert _fit_gate([x], [0, 0]) is None
