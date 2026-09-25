import numpy as np
from src.uncertainty_layer_oos import uncertainty_features, _shrink, _metrics


def test_uncertainty_features_are_finite_and_bounded():
    ensemble = np.asarray([[0.70, 0.20, 0.10], [0.34, 0.33, 0.33]], dtype=float)
    models = np.asarray([
        [[0.70, 0.20, 0.10], [0.34, 0.33, 0.33]],
        [[0.65, 0.25, 0.10], [0.33, 0.34, 0.33]],
        [[0.72, 0.18, 0.10], [0.33, 0.33, 0.34]],
        [[0.68, 0.22, 0.10], [0.34, 0.33, 0.33]],
    ], dtype=float)
    f = uncertainty_features(models, ensemble)
    for key in ("confidence", "entropy", "margin", "disagreement", "agreement", "uncertainty"):
        assert np.isfinite(f[key]).all()
        assert np.all((f[key] >= 0.0) & (f[key] <= 1.0))


def test_shrink_is_normalized_and_less_confident():
    p = np.asarray([[0.90, 0.08, 0.02]], dtype=float)
    out = _shrink(p, np.asarray([1.0]), 0.40)
    assert np.isclose(out.sum(), 1.0)
    assert out[0, 0] < p[0, 0]


def test_metrics_contract():
    y = ["UP", "DOWN", "FLAT"]
    p = np.asarray([[0.1,0.2,0.7],[0.7,0.2,0.1],[0.2,0.7,0.1]],dtype=float)
    m = _metrics(y,p)
    assert m["accuracy"] == 1.0


def test_uncertainty_is_higher_for_ambiguous_prediction():
    clear = np.asarray([[0.90, 0.08, 0.02]], dtype=float)
    ambiguous = np.asarray([[0.35, 0.34, 0.31]], dtype=float)
    models = np.asarray([clear, clear * 0.0 + np.asarray([[0.89,0.09,0.02]]),
                         ambiguous, np.asarray([[0.34,0.35,0.31]])], dtype=float)
    f = uncertainty_features(models, np.vstack([clear, ambiguous]))
    assert f["uncertainty"][1] > f["uncertainty"][0]


def test_learned_risk_probability_is_bounded():
    from src.uncertainty_layer_oos import _fit_error_model, _risk_probability
    y = ["UP", "DOWN"] * 200
    ensemble = np.asarray([[0.7, 0.2, 0.1], [0.7, 0.2, 0.1]] * 200, dtype=float)
    models = np.asarray([ensemble, ensemble * 0.98, ensemble * 1.01, ensemble], dtype=float)
    feats = uncertainty_features(models, ensemble)
    model = _fit_error_model(feats, ensemble, y, list(range(400)))
    p = _risk_probability(model, feats, list(range(20)))
    assert np.isfinite(p).all()
    assert np.all((p >= 0.0) & (p <= 1.0))
