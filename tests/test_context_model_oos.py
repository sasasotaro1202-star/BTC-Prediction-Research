from src.context_model_oos import context_of, fit_context_thresholds, strict_improvement


def row(ret10, vol10):
    return {"x": [0,0,0,ret10,0,0,vol10]}


def test_context_thresholds_use_training_only():
    train=[row(0.001,0.001),row(-0.001,0.002),row(0.0,0.001)]
    t=fit_context_thresholds(train)
    assert t["vol_median"] == 0.001


def test_context_classification_is_deterministic():
    t={"vol_median":0.001,"trend_band":0.00015}
    assert context_of(row(0.001,0.002),t) == "high_vol_trend_up"
    assert context_of(row(-0.001,0.0005),t) == "trend_down"
    assert context_of(row(0.0,0.0005),t) == "range"


def test_strict_improvement_gate():
    base={"accuracy":0.50,"logloss":0.90,"brier":0.60}
    assert strict_improvement(base,{"accuracy":0.50,"logloss":0.894,"brier":0.597})
    assert not strict_improvement(base,{"accuracy":0.49,"logloss":0.896,"brier":0.599})
