import unittest
import numpy as np

from src.hierarchical_oos import HierarchicalClassifier, _base_factories, _sort_oos_rows


class HierarchicalOOSTests(unittest.TestCase):
    def test_two_stage_probabilities_sum_to_one_and_respect_structure(self):
        rng = np.random.default_rng(42)
        X = rng.normal(size=(600, 5))
        y = np.array(["FLAT"] * 300 + ["DOWN"] * 150 + ["UP"] * 150)
        model = HierarchicalClassifier(
            _base_factories()["logistic"],
            _base_factories()["logistic"],
        )
        model.fit(X, y)
        p = model.predict_proba(X[:25])
        self.assertEqual(p.shape, (25, 3))
        self.assertTrue(np.isfinite(p).all())
        self.assertTrue(np.allclose(p.sum(axis=1), 1.0, atol=1e-10))
        self.assertTrue((p >= 0).all())

    def test_direction_stage_requires_both_move_classes(self):
        rng = np.random.default_rng(42)
        X = rng.normal(size=(100, 5))
        y = np.array(["FLAT"] * 50 + ["UP"] * 50)
        model = HierarchicalClassifier(
            _base_factories()["logistic"],
            _base_factories()["logistic"],
        )
        with self.assertRaises(ValueError):
            model.fit(X, y)


    def test_mixed_numeric_and_archive_ids_sort_without_type_assumptions(self):
        rows = [
            {"id": "archive:binance_vision:1789314900000:5m", "created": "2026-09-22T00:02:00+00:00"},
            {"id": 96481, "created": "2026-09-22T00:01:00+00:00"},
        ]
        ordered = _sort_oos_rows(rows)
        self.assertEqual(ordered[0]["id"], 96481)
        self.assertEqual(ordered[1]["id"], "archive:binance_vision:1789314900000:5m")


    def test_load_rows_uses_fresh_venue_fallback_when_archive_has_no_post_train_rows(self):
        from unittest.mock import patch
        from src import hierarchical_oos as hierarchical

        fallback = [{
            "id": "fresh:1",
            "created": "2026-09-22T00:01:00+00:00",
            "x": [0.0],
            "y": "UP",
            "production": [0.2, 0.3, 0.5],
        }]
        with patch.object(hierarchical, "load_primary_production_strict_rows", return_value=[]), \
             patch.object(hierarchical, "load_archive_research_rows", return_value=[]), \
             patch.object(hierarchical, "_fresh_post_training_fallback",
                           return_value=(fallback, "bybit_fresh_archive_frozen_champion")):
            rows, source, eligible = hierarchical.load_rows("5m")

        self.assertEqual(rows, fallback)
        self.assertEqual(source, "bybit_fresh_archive_frozen_champion")
        self.assertFalse(eligible)


if __name__ == "__main__":
    unittest.main()
