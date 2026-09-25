import unittest

from src.archive_refresh_oos import candidate_gate_eligible, factories


class ArchiveRefreshTests(unittest.TestCase):
    def test_candidate_factory_set_is_heterogeneous(self):
        names=set(factories())
        self.assertTrue({"logreg","extra_trees","random_forest","hgb"} <= names)




    def test_relative_scoring_gain_gate_contract_is_strict(self):
        champion = {"accuracy": 0.44, "logloss": 1.0, "brier": 0.60}
        small_gain = {"accuracy": 0.45, "logloss": 0.995, "brier": 0.5958}
        strong_gain = {"accuracy": 0.44, "logloss": 0.96, "brier": 0.592}
        self.assertFalse(candidate_gate_eligible(small_gain, champion, 2998))
        self.assertTrue(candidate_gate_eligible(strong_gain, champion, 2998))
        self.assertFalse(candidate_gate_eligible(strong_gain, champion, 999))

if __name__ == "__main__":
    unittest.main()
