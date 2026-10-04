import os, unittest
from src import pattern_lab_oos
class PatternLabContractTests(unittest.TestCase):
    def test_candidate_budget(self):
        names=pattern_lab_oos._candidate_names()
        self.assertEqual(len(names),144)
        self.assertEqual(len(set(names)),72)
        self.assertEqual(len(pattern_lab_oos.MODEL_NAMES),8)
        self.assertEqual(len(pattern_lab_oos.FEATURE_GROUPS),6)
        self.assertEqual(len(pattern_lab_oos.WINDOWS),3)
    def test_fingerprint(self):
        self.assertNotEqual(pattern_lab_oos._fingerprint("extra_trees","full_15","expanding"),pattern_lab_oos._fingerprint("extra_trees","core_momentum","expanding"))
    def test_window_purge(self):
        rows=[{"id":i} for i in range(2000)]
        train=pattern_lab_oos._window_train(rows,1500,"trailing_750","5m")
        self.assertEqual(len(train),750)
        self.assertEqual(train[-1]["id"],1487)
    def test_handoff_required(self):
        a=os.environ.pop("TESTS_PASSED",None); b=os.environ.pop("AUDIT_PASSED",None)
        try:
            with self.assertRaises(SystemExit): pattern_lab_oos.main()
        finally:
            if a is not None: os.environ["TESTS_PASSED"]=a
            if b is not None: os.environ["AUDIT_PASSED"]=b
if __name__=="__main__": unittest.main()

# CI contract marker: latest multi-pattern model matrix.
