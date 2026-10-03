import io
import unittest
import numpy as np
from pathlib import Path
from src import feature_pattern_exhaustive as fp


class FeaturePatternExhaustiveTests(unittest.TestCase):
    def test_family_partition_covers_every_feature_once(self):
        flat=[f for g in fp.FAMILY_GROUPS.values() for f in g]
        self.assertEqual(len(flat), len(set(flat)))
        self.assertEqual(len(flat), len(fp.FEATURES))
        self.assertEqual(set(flat), set(fp.FEATURES))
        self.assertEqual(len(fp.FEATURES), 92)

    def test_all_nonempty_family_patterns(self):
        self.assertEqual(len(fp.FAMILY_NAMES), 7)
        self.assertEqual(fp.PATTERN_COUNT, 127)

    def test_base_and_full_masks(self):
        fam,features=fp.pattern_features(1)
        self.assertEqual(fam, ["base"])
        self.assertEqual(len(features), len(fp.BASE_FEATURES))
        _,full=fp.pattern_features((1<<len(fp.FAMILY_NAMES))-1)
        self.assertEqual(set(full), set(fp.FEATURES))


    def test_screen_slices_separates_development_and_frozen_holdout(self):
        X=np.asarray([[float(i), float(i%5)] for i in range(60)], dtype=float)
        y=np.asarray([("DOWN","FLAT","UP")[i%3] for i in range(60)], dtype=object)
        folds=[(30,30,40),(40,40,50),(50,50,60)]
        full,dev,holdout,per_fold=fp.screen_with_slices(X,y,folds)
        self.assertEqual(full["n"],30)
        self.assertEqual(dev["n"],20)
        self.assertEqual(holdout["n"],10)
        self.assertEqual(len(per_fold),3)
        self.assertEqual(dev["n"]+holdout["n"],full["n"])

    def test_feature_screen_evidence_contract_is_fail_closed(self):
        source = Path("src/feature_pattern_exhaustive.py").read_text(encoding="utf-8")
        self.assertIn('"promotion_evidence_eligible":False', source)
        self.assertIn('"pit_evidence_status":"NON_STRICT_ARCHIVE_TIMING"', source)
        self.assertIn('"selection_leakage_guard"', source)
        self.assertEqual(fp.NEUTRAL_BPS, 2.0)
        self.assertEqual(fp.HORIZONS, {"5m": 5, "10m": 10})
        self.assertEqual(fp.FAMILY_NAMES, ("base", "momentum", "volatility", "price_action", "flow", "derivatives", "dependence"))



    def test_panel_reader_returns_dictreader_fieldnames(self):
        rows, fields = fp._read_panel_stream(
            io.StringIO("timestamp,price,ret1\n1,100.0,0.01\n")
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(fields, {"timestamp", "price", "ret1"})

    def test_feature_schema_matches_historical_research_source(self):
        from src import historical_research as hr
        self.assertEqual(fp.BASE_FEATURES, hr.BASE_FEATURES)
        self.assertEqual(fp.FRONTIER_FEATURES, hr.FRONTIER_FEATURES)
        self.assertEqual(fp.FEATURES, hr.FEATURES)
        self.assertEqual(fp.NEUTRAL_BPS, hr.NEUTRAL_BPS)


if __name__=="__main__":
    unittest.main()
