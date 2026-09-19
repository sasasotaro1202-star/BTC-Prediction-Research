from src.context_model_oos import (
    _select_factory,
    context_of,
    evaluate_routing,
    fit_context_thresholds,
    strict_improvement,
)


def row(ret10, vol10, label="FLAT", marker=0.0):
    return {"x": [marker, 0, 0, ret10, 0, 0, vol10], "y": label}


class MarkerModel:
    classes_ = ("DOWN", "FLAT", "UP")

    def __init__(self, mode):
        self.mode = mode

    def fit(self, X, y):
        return self

    def predict_proba(self, X):
        out = []
        for x in X:
            if self.mode == "marker":
                idx = int(round(float(x[0]))) % 3
            else:
                idx = 1
            p = [0.01, 0.01, 0.01]
            p[idx] = 0.98
            out.append(p)
        return out


def test_context_thresholds_use_training_only():
    train = [row(0.001, 0.001), row(-0.001, 0.002), row(0.0, 0.001)]
    t = fit_context_thresholds(train)
    assert t["vol_median"] == 0.001


def test_context_classification_is_deterministic():
    t = {"vol_median": 0.001, "trend_band": 0.00015}
    assert context_of(row(0.001, 0.002), t) == "high_vol_trend_up"
    assert context_of(row(-0.001, 0.0005), t) == "trend_down"
    assert context_of(row(0.0, 0.0005), t) == "range"


def test_factory_selection_uses_only_training_tail():
    rows = []
    labels = ("DOWN", "FLAT", "UP")
    for i in range(300):
        marker = float(i % 3)
        rows.append(row(0.0, 0.001, labels[i % 3], marker))
    factories = {
        "marker": lambda: MarkerModel("marker"),
        "constant": lambda: MarkerModel("constant"),
    }
    factory, name = _select_factory(rows, factories)
    assert factory is factories["marker"]
    assert name == "marker"


def test_evaluate_routing_is_descriptive_and_well_formed():
    y = ["DOWN", "FLAT", "UP"]
    global_probs = [
        [0.90, 0.05, 0.05],
        [0.05, 0.90, 0.05],
        [0.05, 0.05, 0.90],
    ]
    routed_probs = [
        [0.80, 0.10, 0.10],
        [0.10, 0.80, 0.10],
        [0.10, 0.10, 0.80],
    ]
    out = evaluate_routing(y, routed_probs, global_probs)
    assert set(out) == {"global", "routed"}
    assert out["global"]["n"] == 3
    assert out["routed"]["n"] == 3
    assert out["global"]["accuracy"] == 1.0
    assert out["routed"]["accuracy"] == 1.0
    assert out["routed"]["logloss"] > out["global"]["logloss"]


def test_strict_improvement_gate():
    base = {"accuracy": 0.50, "logloss": 0.90, "brier": 0.60}
    assert strict_improvement(base, {"accuracy": 0.50, "logloss": 0.894, "brier": 0.597})
    assert not strict_improvement(base, {"accuracy": 0.49, "logloss": 0.896, "brier": 0.599})
