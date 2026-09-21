import unittest

import numpy as np

from ensemble_model import SoftVotingEnsemble


class SoftVotingEnsembleTests(unittest.TestCase):
    def test_fit_predict_proba_has_canonical_classes(self):
        rng = np.random.default_rng(42)
        X = rng.normal(size=(180, 15))
        y = np.array(["DOWN", "FLAT", "UP"] * 60)

        model = SoftVotingEnsemble().fit(X, y)
        probs = model.predict_proba(X[:7])

        self.assertEqual(list(model.classes_), ["DOWN", "FLAT", "UP"])
        self.assertEqual(probs.shape, (7, 3))
        self.assertTrue(np.isfinite(probs).all())
        self.assertTrue(np.all(probs >= 0))
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-7)
        np.testing.assert_allclose(model.weights_.sum(), 1.0, atol=1e-12)
        self.assertIn(model.weight_learning_, {"chronological_internal_validation", "constructor_default"})

    def test_requires_three_classes(self):
        X = np.zeros((30, 15), dtype=float)
        y = np.array(["UP"] * 30)
        with self.assertRaises(ValueError):
            SoftVotingEnsemble().fit(X, y)


if __name__ == "__main__":
    unittest.main()
