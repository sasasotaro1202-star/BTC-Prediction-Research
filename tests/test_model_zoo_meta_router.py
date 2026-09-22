import unittest


class ModelZooMetaRouterContractTests(unittest.TestCase):
    def test_meta_router_deltas_are_signed_candidate_minus_baseline(self):
        candidate = {"accuracy": 0.50, "logloss": 1.0, "brier": 0.60}
        routed = {"accuracy": 0.49, "logloss": 1.01, "brier": 0.61}
        delta = {
            "accuracy": candidate["accuracy"] - routed["accuracy"],
            "logloss": candidate["logloss"] - routed["logloss"],
            "brier": candidate["brier"] - routed["brier"],
        }
        self.assertEqual(delta, {"accuracy": 0.01, "logloss": -0.01, "brier": -0.01})


if __name__ == "__main__":
    unittest.main()
