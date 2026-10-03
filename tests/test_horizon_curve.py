import unittest
from datetime import datetime, timezone

from src.horizon_curve import build_horizon_curve


class HorizonCurveTests(unittest.TestCase):
    def test_curve_is_complete_and_keeps_5m_first(self):
        cutoff = datetime(2026, 10, 3, 4, 7, tzinfo=timezone.utc)
        short = {"DOWN": 0.10, "FLAT": 0.20, "UP": 0.70}
        extended = {}
        for horizon, minutes in (
            ("15m", 15), ("30m", 30), ("1h", 60), ("3h", 180),
            ("6h", 360), ("12h", 720), ("24h", 1440),
        ):
            extended[horizon] = {
                "target_at": datetime.fromtimestamp(
                    cutoff.timestamp() + minutes * 60, tz=timezone.utc
                ),
                "probabilities": {"DOWN": 0.20, "FLAT": 0.30, "UP": 0.50},
                "calibration_status": "UNVALIDATED_UNCALIBRATED",
            }

        out = build_horizon_curve(
            cutoff,
            100_000.0,
            short,
            {"DOWN": 0.20, "FLAT": 0.30, "UP": 0.50},
            extended,
            datetime.fromtimestamp(cutoff.timestamp() + 5 * 60, tz=timezone.utc),
            datetime.fromtimestamp(cutoff.timestamp() + 10 * 60, tz=timezone.utc),
        )

        self.assertEqual(
            out["horizon_order"],
            ["5m", "10m", "15m", "30m", "1h", "3h", "6h", "12h", "24h"],
        )
        self.assertEqual([p["horizon"] for p in out["points"]], out["horizon_order"])
        self.assertEqual(out["primary_horizon"], "5m")
        self.assertEqual(out["points"][0]["state"], "PRIMARY_5M")
        self.assertFalse(out["points"][0]["research_only"])
        self.assertTrue(all(p["research_only"] for p in out["points"][2:]))
        for point in out["points"]:
            self.assertAlmostEqual(sum(point["probabilities"].values()), 1.0)
            self.assertEqual(
                set(point["delta_from_previous"] or {}),
                {"DOWN", "FLAT", "UP"},
            )


if __name__ == "__main__":
    unittest.main()
