import os
import unittest
from unittest.mock import patch

from src.live_data_fail_closed import validate_latest


class TestLiveDataFailClosed(unittest.TestCase):
    def test_deferred_cycle_bypasses_stale_prediction_check(self):
        with patch.dict(os.environ, {"BTC_LIVE_DEFERRED": "1"}, clear=False):
            result = validate_latest()
        self.assertEqual(result, {"ok": True, "mode": "deferred"})


if __name__ == "__main__":
    unittest.main()
