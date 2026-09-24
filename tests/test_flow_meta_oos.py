import numpy as np

from src.flow_meta_oos import _causal_train, _norm, _vector


def _row(created, target):
    return {
        "id": 1,
        "created": created,
        "target": target,
        "y": "UP",
        "model_version": "test",
        "base": [0.2, 0.2, 0.6],
        "flow": [0.0] * 47,
        "context": {"entropy": 0.8, "margin": 0.2, "direction": "UP"},
    }


def test_probability_normalization_contract():
    p = _norm([2.0, 3.0, 5.0])
    assert p is not None
    assert np.isclose(p.sum(), 1.0)
    assert np.isfinite(p).all()


def test_causal_train_requires_embargoed_target():
    rows = [
        _row("2026-09-25T00:00:00+00:00", "2026-09-25T00:05:00+00:00"),
        _row("2026-09-25T00:01:00+00:00", "2026-09-25T00:12:00+00:00"),
    ]
    train = _causal_train(rows, "2026-09-25T00:20:00+00:00", "5m")
    assert train == []


def test_vector_is_finite():
    row = _row("2026-09-25T00:00:00+00:00", "2026-09-25T00:05:00+00:00")
    x = _vector(row)
    assert x.ndim == 1
    assert len(x) == 52
    assert np.isfinite(x).all()


def test_deferred_output_contract():
    from src.flow_meta_oos import evaluate
    result = evaluate("5m", [])
    assert result["status"] == "DEFERRED"
    assert result["research_only"] is True
    assert result["production_changed"] is False
    assert result["promotion_allowed"] is False
