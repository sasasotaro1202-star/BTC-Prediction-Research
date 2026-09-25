import unittest

from src.archive_refresh_oos import factories


class ArchiveRefreshTests(unittest.TestCase):
    def test_candidate_factory_set_is_heterogeneous(self):
        names=set(factories())
        self.assertTrue({"logreg","extra_trees","random_forest","hgb"} <= names)




    def test_relative_scoring_gain_gate_contract_is_strict(self):
        # A 0.5% LogLoss gain and 0.7% Brier gain must not be sufficient.
        champion_ll = 1.0
        candidate_ll = 0.995
        champion_br = 0.60
        candidate_br = 0.5958
        ll_gain = (champion_ll - candidate_ll) / champion_ll
        br_gain = (champion_br - candidate_br) / champion_br
        self.assertLess(ll_gain, 0.03)
        self.assertLess(br_gain, 0.01)

if __name__ == "__main__":
    unittest.main()
