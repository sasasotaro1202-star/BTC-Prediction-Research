import unittest
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

import historical_research


class TestHistoricalResearch(unittest.TestCase):
    def test_normalize(self):
        p = historical_research.normalize([[2.0, 3.0, 5.0]])
        self.assertAlmostEqual(float(p.sum()), 1.0)
        self.assertEqual(p.shape, (1, 3))

    def test_score_is_bounded(self):
        y = ['DOWN', 'FLAT', 'UP']
        p = [[0.9, 0.05, 0.05], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]]
        m = historical_research.score(y, p)
        self.assertEqual(m['accuracy'], 1.0)
        self.assertGreaterEqual(m['logloss'], 0.0)
        self.assertGreaterEqual(m['brier'], 0.0)

    def test_research_module_never_publishes_production(self):
        self.assertTrue(historical_research.OUT.name == 'historical_oos_report.json')
        self.assertEqual(historical_research.ROOT.name, 'BTC-Prediction-Research')


if __name__ == '__main__':
    unittest.main()
