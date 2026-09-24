import numpy as np


def test_mix_preserves_probability_contract():
    from src.dual_memory_oos import _mix
    out = _mix(np.array([[0.8, 0.1, 0.1]]), np.array([[0.1, 0.2, 0.7]]), 0.75)
    assert out.shape == (1, 3)
    assert np.all(np.isfinite(out))
    assert np.allclose(out.sum(axis=1), 1.0)


def test_recent_weight_stays_bounded_without_history():
    from src.dual_memory_oos import _adaptive_recent_weight
    assert _adaptive_recent_weight([]) == 0.5


def test_recent_weight_prefers_lower_loss_expert():
    from src.dual_memory_oos import _adaptive_recent_weight
    history = [
        {"recent_logloss": 0.8, "full_logloss": 1.1, "recent_brier": 0.5, "full_brier": 0.7},
        {"recent_logloss": 0.9, "full_logloss": 1.2, "recent_brier": 0.55, "full_brier": 0.72},
    ]
    assert _adaptive_recent_weight(history) > 0.5
