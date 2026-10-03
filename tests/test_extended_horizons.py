import unittest
from datetime import datetime, timezone

from src.extended_horizons import forecast_extended_horizons


class ExtendedHorizonTests(unittest.TestCase):
    def test_longer_horizon_forecasts_shrink_toward_uniform(self):
        cutoff = datetime(2026, 10, 3, 4, 7, tzinfo=timezone.utc)
        features = {
            "ret_1m": 0.001, "ret_3m": 0.001, "ret_5m": 0.002, "ret_10m": 0.003,
            "ret_15m": 0.004, "ret_30m": 0.005, "acceleration": 0.0001,
            "volatility_5m": 0.001, "volatility_10m": 0.002, "range_position_10m": 0.7,
            "range_position_30m": 0.7, "body_1m": 0.001, "upper_wick_1m": 0.0001,
            "lower_wick_1m": 0.0001, "volume_ratio": 1.1, "volume_trend": 1.0,
            "ema_gap_5m": 0.001, "ema_gap_10m": 0.001, "trend_alignment": 0.002,
        }
        short = {"DOWN": 0.05, "FLAT": 0.10, "UP": 0.85}
        out = forecast_extended_horizons(
            cutoff, features, {}, short, short,
            lambda _f, _m: {"DOWN": 0.10, "FLAT": 0.10, "UP": 0.80},
        )
        d15 = abs(out["15m"]["probabilities"]["UP"] - 1 / 3)
        d24 = abs(out["24h"]["probabilities"]["UP"] - 1 / 3)
        self.assertGreater(d15, d24)
        self.assertTrue(all(v["research_only"] for v in out.values()))


if __name__ == "__main__":
    unittest.main()
