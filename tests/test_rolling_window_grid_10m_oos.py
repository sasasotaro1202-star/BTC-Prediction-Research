import numpy as np
from src.rolling_window_grid_10m_oos import _metrics, _norm


def test_probability_contract():
    p = _norm(np.asarray([0.2, 0.3, 0.5]))
    assert p.shape == (1, 3)
    assert np.isclose(p.sum(), 1.0)
    assert np.isfinite(p).all()


def test_metrics_contract():
    m = _metrics(["DOWN", "FLAT", "UP"], np.eye(3))
    assert m["accuracy"] == 1.0
    assert m["logloss"] >= 0
    assert m["brier"] >= 0
    assert m["ece"] >= 0


def test_window_grid_source_preserves_block_count_contract():
    from src.rolling_window_grid_10m_oos import _evaluate_blocks
    blocks = [
        {
            "y": ["UP"],
            "frozen": np.asarray([[1.0, 0.0, 0.0]]),
            "recent": np.asarray([[0.0, 0.0, 1.0]]),
        }
    ]
    result = _evaluate_blocks(blocks)
    assert result["baseline"]["n"] == 1
    assert result["candidate"]["n"] == 1
    assert "delta" in result
