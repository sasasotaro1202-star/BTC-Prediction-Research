import json
import unittest

import numpy as np

from src.adaptive_calibration_replay_oos import _fit_temperature, _quality


class TestAdaptiveCalibrationReplay(unittest.TestCase):
    def test_quality_reports_finite_metrics(self):
        y = ["DOWN", "FLAT", "UP"] * 40
        p = np.tile(
            np.asarray([[0.7, 0.2, 0.1], [0.2, 0.6, 0.2], [0.1, 0.2, 0.7]]),
            (40, 1),
        )
        q = _quality(y, p)
        for key in ("accuracy", "logloss", "brier", "ece"):
            self.assertTrue(np.isfinite(float(q[key])))

    def test_temperature_falls_back_safely_on_small_or_single_class_history(self):
        y_small = ["UP"] * 20
        p_small = np.tile(np.asarray([[0.2, 0.2, 0.6]]), (20, 1))
        self.assertEqual(_fit_temperature(p_small, y_small), 1.0)

    def test_temperature_is_finite_and_bounded_on_balanced_history(self):
        y = ["DOWN", "FLAT", "UP"] * 400
        p = np.tile(
            np.asarray([[0.6, 0.3, 0.1], [0.2, 0.6, 0.2], [0.1, 0.2, 0.7]]),
            (400, 1),
        )
        t = _fit_temperature(p, y)
        self.assertTrue(np.isfinite(float(t)))
        self.assertGreaterEqual(t, 0.5)
        self.assertLessEqual(t, 3.0)


    def test_main_records_deferred_when_archive_is_below_minimum(self):
        from src import adaptive_calibration_replay_oos as r
        from pathlib import Path
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as tmp:
            original_out = r.OUT
            original_cutoff = r._post_champion_cutoff_ms
            original_dataset = r._dataset
            try:
                r.OUT = Path(tmp) / "adaptive.json"
                r._post_champion_cutoff_ms = lambda: 1234567890000
                def sparse_dataset():
                    raise RuntimeError("insufficient post-Champion archive rows: 42")
                r._dataset = sparse_dataset

                r.main()
                obj = json.loads(r.OUT.read_text(encoding="utf-8"))
                self.assertEqual(obj["dataset_status"], "DEFERRED")
                self.assertEqual(obj["dataset_rows"], 42)
                self.assertEqual(set(obj["horizons"]), {"5m", "10m"})
                self.assertTrue(
                    all(item["status"] == "DEFERRED" for item in obj["horizons"].values())
                )
                self.assertTrue(
                    all(item["replay_holdout_protected"] is True for item in obj["horizons"].values())
                )
            finally:
                r.OUT = original_out
                r._post_champion_cutoff_ms = original_cutoff
                r._dataset = original_dataset


if __name__ == "__main__":
    unittest.main()
