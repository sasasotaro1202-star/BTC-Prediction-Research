import math

from context_model_oos import _context_blend_weight


def test_context_blend_is_conservative_and_support_aware():
    uniform = {"a": 0.5, "b": 0.5}
    assert _context_blend_weight(250, uniform) < _context_blend_weight(1000, uniform)
    assert _context_blend_weight(10000, uniform) <= 0.30
    assert _context_blend_weight(250, {"a": 1.0, "b": 0.0}) < _context_blend_weight(250, uniform)
    assert _context_blend_weight(249, uniform) == 0.0


def test_context_blend_has_no_nan_or_negative_values():
    for n in (0, 250, 500, 1000, 5000):
        for weights in ({}, {"a": 1.0}, {"a": 0.8, "b": 0.2}):
            v = _context_blend_weight(n, weights)
            assert math.isfinite(v)
            assert 0.0 <= v <= 0.30
