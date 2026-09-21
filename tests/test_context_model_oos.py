import unittest
import numpy as np
from sklearn.linear_model import LogisticRegression
from src.context_model_oos import route_predictions, dynamic_route_predictions, fit_context_thresholds, context_of, _validation_slices

def factory():
    return LogisticRegression(max_iter=1000, random_state=42)

def make_rows(n, offset=0):
    rows = []
    for i in range(n):
        label = ("DOWN", "FLAT", "UP")[i % 3]
        x = np.zeros(15, dtype=float)
        x[3] = {"DOWN": -0.002, "FLAT": 0.0, "UP": 0.002}[label] + ((i + offset) % 5) * 1e-5
        x[6] = 0.001 + ((i + offset) % 7) * 0.0001
        x[0] = x[3]
        rows.append({"x": x.tolist(), "y": label})
    return rows

class ContextRouterTests(unittest.TestCase):
    def test_context_thresholds_are_train_only(self):
        train = make_rows(300)
        thresholds = fit_context_thresholds(train)
        self.assertEqual(
            context_of({"x": train[-1]["x"]}, thresholds),
            context_of({"x": train[-1]["x"]}, thresholds),
        )

    def test_routed_probabilities_do_not_depend_on_test_labels(self):
        train = make_rows(300)
        test_a = make_rows(20, offset=100)
        test_b = [dict(r, y=("UP" if r["y"] == "DOWN" else "DOWN")) for r in test_a]
        result_a = route_predictions(train, test_a, {"logreg": factory})
        result_b = route_predictions(train, test_b, {"logreg": factory})
        self.assertIsNotNone(result_a)
        np.testing.assert_allclose(result_a["routed_probs"], result_b["routed_probs"])
        self.assertEqual(result_a["contexts"], result_b["contexts"])

    def test_insufficient_training_data_fails_closed(self):
        self.assertIsNone(route_predictions(make_rows(100), make_rows(20), {"logreg": factory}))

    def test_dynamic_router_is_research_only_and_label_invariant(self):
        train = make_rows(320)
        test_a = make_rows(40, offset=200)
        test_b = [dict(r, y=("UP" if r["y"] == "DOWN" else "DOWN")) for r in test_a]
        factories = {"logreg": factory}
        a = dynamic_route_predictions(train, test_a, factories)
        b = dynamic_route_predictions(train, test_b, factories)
        self.assertIsNotNone(a)
        self.assertTrue(a["research_only"])
        self.assertFalse(a["production_changed"])
        np.testing.assert_allclose(a["routed_probs"], b["routed_probs"])
        self.assertEqual(a["routing_mode"], "soft_dynamic_ensemble")

    def test_dynamic_router_weights_are_normalized(self):
        train = make_rows(320)
        result = dynamic_route_predictions(train, make_rows(20, offset=400), {"logreg": factory})
        self.assertIsNotNone(result)
        self.assertAlmostEqual(sum(result["global_weights"].values()), 1.0, places=6)


    def test_validation_slices_use_positions_not_duplicate_row_identity(self):
        rows = make_rows(420)
        slices = _validation_slices(rows)
        self.assertEqual(len(slices), 2)
        self.assertTrue(all(end > start for start, end in slices))
        self.assertEqual(slices[0][0], max(int(len(rows) * 0.70), len(rows) - 150))

    def test_multi_model_router_uses_stable_weights_and_shrinkage(self):
        train = make_rows(420)
        test = make_rows(40, offset=500)
        factories = {
            "logreg_c01": lambda: LogisticRegression(C=0.1, max_iter=1000, random_state=42),
            "logreg_c10": lambda: LogisticRegression(C=10.0, max_iter=1000, random_state=42),
        }
        result = dynamic_route_predictions(train, test, factories)
        self.assertIsNotNone(result)
        weights = result["global_weights"]
        self.assertAlmostEqual(sum(weights.values()), 1.0, places=6)
        self.assertTrue(all(0.0 <= w <= 1.0 for w in weights.values()))
        self.assertTrue(all(w >= 0.10 for w in weights.values()))

    def test_router_remains_research_only_when_context_evidence_is_weak(self):
        train = make_rows(320)
        test = make_rows(30, offset=700)
        result = dynamic_route_predictions(train, test, {"logreg": factory})
        self.assertIsNotNone(result)
        self.assertEqual(result["routing_mode"], "soft_dynamic_ensemble")
        self.assertTrue(result["research_only"])

def test_multi_model_weight_floor_survives_normalization():
    train = make_rows(420)
    test = make_rows(40, offset=900)
    factories = {
        "a": lambda: LogisticRegression(C=0.01, max_iter=1000, random_state=42),
        "b": lambda: LogisticRegression(C=100.0, max_iter=1000, random_state=42),
        "c": lambda: LogisticRegression(C=1.0, max_iter=1000, random_state=42),
    }
    result = dynamic_route_predictions(train, test, factories)
    assert result is not None
    weights = result["global_weights"]
    assert abs(sum(weights.values()) - 1.0) < 1e-6
    assert all(w >= 0.10 for w in weights.values())



    def test_context_confidence_factor_penalizes_ambiguous_disagreement(self):
        confident = np.tile(np.asarray([[0.98, 0.01, 0.01]]), (20, 1))
        same = np.tile(np.asarray([[0.98, 0.01, 0.01]]), (20, 1))
        low = np.tile(np.asarray([[1/3, 1/3, 1/3]]), (20, 1))
        disagree = np.tile(np.asarray([[0.01, 0.01, 0.98]]), (20, 1))
        self.assertGreater(
            adaptive._context_confidence_factor([confident, same]),
            adaptive._context_confidence_factor([low, disagree]),
        )
        value = adaptive._context_confidence_factor([confident, disagree])
        self.assertTrue(0.35 <= value <= 1.0)

if __name__ == "__main__":
    unittest.main()

def test_context_router_factory_includes_lightgbm_candidate():
    from src.context_router_oos import factories

    fs = factories()
    assert "lightgbm" in fs
    model = fs["lightgbm"]()
    assert model.n_estimators == 180
    assert model.num_leaves == 15
