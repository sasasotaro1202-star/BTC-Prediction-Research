import unittest

from src.archive_refresh_oos import factories
from src.binance_history import binance_archive_rows



class ArchiveRefreshTests(unittest.TestCase):
    def test_archive_loader_source_contract_exists(self):
        self.assertTrue(callable(binance_archive_rows))

    def test_candidate_factory_set_is_heterogeneous(self):
        names=set(factories())
        self.assertTrue({"logreg","extra_trees","random_forest","hgb"} <= names)


if __name__ == "__main__":
    unittest.main()
