import unittest
import numpy as np

from src.hierarchical_oos import HierarchicalClassifier, _base_factories


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


if __name__ == "__main__":
    unittest.main()
