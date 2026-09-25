import unittest
import numpy as np
from src.archive_primary_bridge_oos import _archive_dataset


class ArchivePrimaryBridgeTests(unittest.TestCase):
    def test_pilot_threshold_is_below_full_evaluation_threshold(self):
        from src.archive_primary_bridge_oos import MIN_PRIMARY_ROWS
        self.assertEqual(MIN_PRIMARY_ROWS, 50)
    def test_archive_dataset_enforces_exact_elapsed_target(self):
        rows=[[i*60000,100,101,99,100+i*0.01,10] for i in range(80)]
        from unittest.mock import patch
        with patch("src.archive_primary_bridge_oos.binance_archive_rows",return_value=rows):
            out=_archive_dataset("5m")
        self.assertTrue(out)
        self.assertTrue(all(np.isfinite(np.asarray(r["x"],float)).all() for r in out))
        self.assertTrue(all(
            "y" in r and r["y"] in {"DOWN","FLAT","UP"}
            for r in out
        ))

if __name__=="__main__":
    unittest.main()
