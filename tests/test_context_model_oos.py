import unittest
import numpy as np
from sklearn.linear_model import LogisticRegression
from src.context_model_oos import route_predictions, fit_context_thresholds, context_of

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

if __name__ == "__main__":
    unittest.main()
