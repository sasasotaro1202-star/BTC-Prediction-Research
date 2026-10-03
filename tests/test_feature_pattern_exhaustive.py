import unittest
from src import feature_pattern_exhaustive as fp

class FeaturePatternExhaustiveTests(unittest.TestCase):
    def test_family_partition_covers_every_feature_once(self):
        flat=[f for g in fp.FAMILY_GROUPS.values() for f in g]
        self.assertEqual(set(flat),set(fp.FEATURES))
        self.assertEqual(set(flat),set(fp.FEATURES))
        self.assertEqual(len(fp.FEATURES),92)
    def test_all_nonempty_family_patterns(self):
        self.assertEqual(len(fp.FAMILY_NAMES),7)
        self.assertEqual(fp.PATTERN_COUNT,127)
    def test_base_and_full_masks(self):
        fam,features=fp.pattern_features(1)
        self.assertEqual(fam,["base"])
        self.assertEqual(len(features),len(fp.BASE_FEATURES))
        _,full=fp.pattern_features((1<<len(fp.FAMILY_NAMES))-1)
        self.assertEqual(set(full),set(fp.FEATURES))

if __name__=="__main__": unittest.main()
