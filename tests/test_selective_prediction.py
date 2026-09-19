import unittest
import numpy as np
from src.selective_prediction import choose_threshold, evaluate

class SelectivePredictionTests(unittest.TestCase):
    def test_evaluate_respects_threshold(self):
        y=np.array([0,1,2,1])
        p=np.array([[.9,.05,.05],[.1,.8,.1],[.1,.1,.8],[.34,.33,.33]])
        summary=evaluate(y,p,0.0)
        self.assertEqual(summary["coverage"],1.0)
        self.assertEqual(summary["n"],4)
        low=evaluate(y,p,.5)
        high=evaluate(y,p,.85)
        self.assertEqual(low["n"],3)
        self.assertEqual(high["n"],1)
        self.assertLess(high["coverage"],low["coverage"])

    def test_threshold_selection_is_development_only_primitive(self):
        y=np.array([0,1,2,1]*80)
        p=np.tile(np.array([[.8,.1,.1],[.1,.8,.1],[.1,.1,.8],[.34,.33,.33]]),(80,1))
        chosen=choose_threshold(y,p)
        self.assertTrue(chosen)
        for value in chosen.values():
            self.assertIn("threshold",value)
            self.assertGreaterEqual(value["threshold"],0.0)
            self.assertLessEqual(value["threshold"],1.0)

if __name__=="__main__":
    unittest.main()
