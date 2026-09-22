import unittest

from src.archive_refresh_oos import factories


class ArchiveRefreshTests(unittest.TestCase):
    def test_candidate_factory_set_is_heterogeneous(self):
        names=set(factories())
        self.assertTrue({"logreg","extra_trees","random_forest","hgb"} <= names)


if __name__ == "__main__":
    unittest.main()
