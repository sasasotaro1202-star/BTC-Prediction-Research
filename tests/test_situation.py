from src.situation import summarize_situation


def _inputs():
    features = {
        "volatility_10m": 0.001,
        "volatility_5m": 0.001,
        "trend_alignment": 0.0005,
    }
    market = {}
    p5 = {"DOWN": 0.45, "FLAT": 0.10, "UP": 0.45}
    p10 = {"DOWN": 0.20, "FLAT": 0.10, "UP": 0.70}
    return features, market, p5, p10


def test_conflicting_horizons_cap_high_quality():
    features, market, p5, p10 = _inputs()
    result = summarize_situation(features, market, p5, p10, data_quality={})
    assert result["horizon_alignment"] == "CONFLICT"
    assert result["signal_quality"] == "MEDIUM"


def test_partial_data_cannot_report_high_quality():
    features, market, p5, p10 = _inputs()
    p10 = {"DOWN": 0.05, "FLAT": 0.05, "UP": 0.90}
    result = summarize_situation(
        features,
        market,
        p5,
        p10,
        data_quality={"binance_futures": "error:HTTPError"},
    )
    assert result["data_state"] == "PARTIAL"
    assert result["signal_quality"] == "MEDIUM"


def test_degraded_data_forces_low_quality():
    features, market, p5, p10 = _inputs()
    p5 = {"DOWN": 0.05, "FLAT": 0.05, "UP": 0.90}
    p10 = {"DOWN": 0.05, "FLAT": 0.05, "UP": 0.90}
    result = summarize_situation(
        features,
        market,
        p5,
        p10,
        data_quality={
            "binance_futures": "error:HTTPError",
            "binance_depth": "error:HTTPError",
        },
    )
    assert result["data_state"] == "DEGRADED"
    assert result["signal_quality"] == "LOW"


def test_missing_market_signals_are_not_reported_as_neutral():
    features, market, p5, p10 = _inputs()
    result = summarize_situation(features, market, p5, p10, data_quality={})
    assert result["microstructure_flow_available"] is False
    assert result["cross_exchange_divergence"] == "UNKNOWN"
    assert result["unknown_signal_count"] >= 2
    assert result["orderflow_state"] == "UNKNOWN"
