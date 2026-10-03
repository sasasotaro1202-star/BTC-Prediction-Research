from pathlib import Path
import numpy as np

from src.rich_production_research import (
    CLASSES,
    LEGACY_FEATURES,
    LEGACY_INDICES,
    RICH_FEATURES,
    PANEL_LOOKBACK_MINUTES,
    _block_bootstrap_ci,
    _metrics,
    _relative_gain,
)


def test_rich_schema_contains_legacy_features_exactly():
    assert len(LEGACY_FEATURES) == 15
    assert len(RICH_FEATURES) == 61
    assert len(LEGACY_INDICES) == 15
    assert [RICH_FEATURES[i] for i in LEGACY_INDICES] == [
        "ret1","ret3","ret5","ret10","accel","rv5","rv10",
        "rangepos10","body","upper","lower","volratio","voltrend",
        "ema_gap_5m","ema_gap_10m",
    ]


def test_metrics_three_class_known_case():
    y = ["DOWN", "FLAT", "UP"]
    p = np.eye(3, dtype=float)
    m = _metrics(y, p)
    assert m["n"] == 3
    assert m["accuracy"] == 1.0
    assert m["brier"] < 1e-12
    assert m["logloss"] < 1e-6


def test_relative_gain_direction():
    assert abs(_relative_gain(0.50, 0.55, True) - 0.10) < 1e-12
    assert abs(_relative_gain(1.00, 0.97, False) - 0.03) < 1e-12


def test_block_bootstrap_is_paired_and_finite():
    y = np.array(["DOWN","FLAT","UP"] * 2000, dtype=object)
    rich = np.tile(np.eye(3), (2000,1))
    base = np.roll(rich, 1, axis=1)
    out = _block_bootstrap_ci(y, rich, base)
    assert out["n_blocks"] >= 5
    assert np.isfinite(out["mean_block_accuracy_delta"])
    assert out["ci95_low"] > 0.0


def test_research_only_marker_is_present():
    text = Path("src/rich_production_research.py").read_text(encoding="utf-8")
    assert 'research_only": True' in text
    assert 'production_changed": False' in text

def test_extended_microstructure_features_are_append_only():
    expected_tail = [
        "rv60","volume_intensity_5m_60m","trade_intensity_5m_60m",
        "flow_30m","flow_60m","flow_toxicity_30m","flow_toxicity_60m",
        "amihud_15m","amihud_30m","range_intensity_10m",
    ]
    assert list(RICH_FEATURES[-18:-8]) == expected_tail
    assert [RICH_FEATURES[i] for i in LEGACY_INDICES] == [
        "ret1","ret3","ret5","ret10","accel","rv5","rv10",
        "rangepos10","body","upper","lower","volratio","voltrend",
        "ema_gap_5m","ema_gap_10m",
    ]


def test_model_factories_include_xgboost_when_dependency_is_available():
    from src import rich_production_research as r
    factories = r._factories()
    if r.XGBClassifier is not None:
        assert "xgboost" in factories
    else:
        assert "xgboost" not in factories


def test_archive_resilient_request_patch_is_wired():
    import historical_research as hr
    assert hr.req_json.__module__ in {"historical_research_runner", "src.historical_research_runner"}


def test_main_passes_horizon_arrays_as_three_arguments(monkeypatch, tmp_path):
    from src import rich_production_research as r

    X5 = np.zeros((4, 53), dtype=float)
    y5 = np.asarray(["DOWN", "FLAT", "UP", "DOWN"], dtype=object)
    t5 = np.arange(4, dtype=np.int64)
    X10 = np.ones((3, 53), dtype=float)
    y10 = np.asarray(["UP", "FLAT", "DOWN"], dtype=object)
    t10 = np.arange(10, 13, dtype=np.int64)

    seen = []

    def fake_build_panel():
        return (X5, y5, t5), (X10, y10, t10), np.arange(4, dtype=np.int64)

    def fake_evaluate(horizon, X, y, t):
        seen.append((horizon, X, y, t))
        return {"status": "OK", "horizon": horizon}

    monkeypatch.setattr(r, "build_panel", fake_build_panel)
    monkeypatch.setattr(r, "evaluate_horizon", fake_evaluate)
    monkeypatch.setattr(r, "OUT", tmp_path / "rich.json")

    assert r.main() == 0
    assert [(h, len(X), len(y), len(t)) for h, X, y, t in seen] == [
        ("5m", 4, 4, 4),
        ("10m", 3, 3, 3),
    ]


def test_xgboost_factory_preserves_canonical_string_labels_when_available():
    from src import rich_production_research as r

    if r.XGBClassifier is None:
        return

    model = r._factories()["xgboost"]()
    X = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0],
            [0.0, 0.1, 0.0],
            [1.0, 1.0, 1.0],
            [1.1, 1.0, 1.0],
            [1.0, 1.1, 1.0],
            [-1.0, -1.0, -1.0],
            [-1.1, -1.0, -1.0],
            [-1.0, -1.1, -1.0],
        ],
        dtype=float,
    )
    y = np.asarray(
        ["FLAT", "FLAT", "FLAT", "UP", "UP", "UP", "DOWN", "DOWN", "DOWN"]
    )
    model.fit(X, y)
    probs = model.predict_proba(X)
    pred = model.predict(X)

    assert model.classes_.tolist() == ["DOWN", "FLAT", "UP"]
    assert probs.shape == (len(y), 3)
    assert np.isfinite(probs).all()
    assert set(pred.tolist()) <= {"DOWN", "FLAT", "UP"}


def test_candlestick_pattern_features_are_causal_and_well_bounded():
    from src import rich_production_research as r
    vals = r._candlestick_pattern_features(
        curr_open=100.0, curr_high=105.0, curr_low=99.0, curr_close=104.0,
        prev_open=104.0, prev_high=105.0, prev_low=98.0, prev_close=100.0,
    )
    assert len(vals) == 8
    assert all(np.isfinite(vals))
    assert all(0.0 <= vals[i] <= 1.0 for i in (0,1,2,7))
    assert vals[3] == 1.0
    assert vals[4] == 0.0
    assert vals[5] == 1.0
    assert vals[6] == 0.0


def test_candlestick_features_are_append_only():
    assert list(RICH_FEATURES[-8:]) == [
        "doji_score","hammer_score","shooting_star_score",
        "bullish_engulfing","bearish_engulfing","inside_bar","outside_bar",
        "marubozu_score",
    ]


def test_panel_lookback_supports_all_60m_features():
    assert PANEL_LOOKBACK_MINUTES >= 60
