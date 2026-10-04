import unittest
from datetime import datetime, timezone, timedelta

from src.model_compare import _strict_pit_provenance_ok


class TestStrictPITProvenance(unittest.TestCase):
    def _base(self, mode="bybit_fallback"):
        t = datetime(2026, 9, 22, 8, 0, tzinfo=timezone.utc).isoformat()
        return {
            "decision_time_utc": t,
            "provenance": {
                "available_at": t,
                "retrieved_at": t,
                "prediction_cutoff": t,
                "sources": {
                    "bybit_futures": {
                        "event_time": None,
                        "available_at": t,
                        "retrieved_at": t,
                        "prediction_cutoff": t,
                        "status": "ok",
                    }
                },
            },
            "production_mode": mode,
        }, t

    def test_bybit_fallback_requires_source_prediction_cutoff(self):
        scenario, created = self._base()
        scenario["provenance"]["sources"]["bybit_futures"].pop("prediction_cutoff")
        self.assertFalse(_strict_pit_provenance_ok(scenario, created))

    def test_bybit_fallback_passes_when_cutoff_is_recorded(self):
        scenario, created = self._base()
        self.assertTrue(_strict_pit_provenance_ok(scenario, created))

    def test_source_cutoff_after_decision_fails_closed(self):
        scenario, created = self._base()
        later = (datetime.fromisoformat(created) + timedelta(seconds=1)).isoformat()
        scenario["provenance"]["sources"]["bybit_futures"]["prediction_cutoff"] = later
        self.assertFalse(_strict_pit_provenance_ok(scenario, created))

    def test_future_feature_snapshot_cutoff_fails_closed(self):
        scenario, created = self._base()
        later = (datetime.fromisoformat(created) + timedelta(seconds=1)).isoformat()
        scenario["feature_snapshot_cutoff"] = later
        self.assertFalse(_strict_pit_provenance_ok(scenario, created))

    def test_unknown_feature_pit_status_fails_closed(self):
        scenario, created = self._base()
        scenario["feature_pit_status"] = "UNKNOWN"
        self.assertFalse(_strict_pit_provenance_ok(scenario, created))

    def test_feature_level_available_at_after_cutoff_fails_closed(self):
        scenario, created = self._base()
        later = (datetime.fromisoformat(created) + timedelta(seconds=1)).isoformat()
        scenario["feature_provenance"] = {
            "pit_status": "PASS",
            "features": {"ret_1m": {"pit_status": "PASS", "available_at": later}},
        }
        self.assertFalse(_strict_pit_provenance_ok(scenario, created))

    def test_valid_feature_provenance_is_backward_compatible(self):
        scenario, created = self._base()
        scenario["feature_provenance"] = {
            "pit_status": "PASS",
            "feature_snapshot_cutoff": created,
            "feature_max_available_at": created,
            "features": {"ret_1m": {"pit_status": "PASS", "available_at": created}},
        }
        self.assertTrue(_strict_pit_provenance_ok(scenario, created))


if __name__ == "__main__":
    unittest.main()
