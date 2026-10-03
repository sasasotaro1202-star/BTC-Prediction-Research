import unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
class ForecastTrajectoryContractTests(unittest.TestCase):
    def test_live_cycle_wires_chart_ready_export(self):
        text=(ROOT/".github"/"workflows"/"btc_live_cycle.yml").read_text(encoding="utf-8")
        self.assertIn("python src/forecast_trajectory.py --limit 2880",text)
        self.assertGreaterEqual(text.count("data/historical_research/forecast_trajectory.json"), 5)
        self.assertIn("tests/test_forecast_trajectory.py",text)
        self.assertIn("forecast_trajectory.json",text)
        self.assertIn("btc-forecast-trajectory",text)

    def test_dashboard_and_live_cycle_allow_extended_history_bootstrap(self):
        workflow=(ROOT/".github"/"workflows"/"btc_live_cycle.yml").read_text(encoding="utf-8")
        self.assertIn('if h in {"5m","10m"} and item.get("point_count",0) < 1',workflow)
        self.assertIn("PENDING_FIRST_LIVE_SAMPLE",workflow)

    def test_five_minute_is_primary(self):
        text=(ROOT/"src"/"forecast_trajectory.py").read_text(encoding="utf-8")
        self.assertIn('"primary_5m"',text)
        self.assertIn('"no_missing_as_zero":True',text)
if __name__=="__main__": unittest.main()
