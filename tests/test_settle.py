import unittest
from unittest.mock import patch

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

import settle


class TestSettle(unittest.TestCase):
    def test_direction_threshold(self):
        self.assertEqual(settle.direction(100.0, 100.03), 'UP')
        self.assertEqual(settle.direction(100.0, 99.97), 'DOWN')
        self.assertEqual(settle.direction(100.0, 100.01), 'FLAT')

    def test_resolve_targets_deduplicates_and_tolerates_failure(self):
        calls = []

        def fake_target(target, source):
            calls.append((target, source))
            if target == 'bad':
                raise RuntimeError('temporary public-source failure')
            return (123.0, 'test')

        with patch.object(settle, 'target_close_preferred', side_effect=fake_target):
            result = settle.resolve_targets([('a', 'binance_futures'), ('a', 'binance_futures'), ('bad', 'binance_futures')], max_workers=2)

        self.assertEqual(sorted(calls), [('a', 'binance_futures'), ('bad', 'binance_futures')])
        self.assertEqual(result[('a', 'binance_futures')], (123.0, 'test'))
        self.assertEqual(result[('bad', 'binance_futures')], (None, 'unavailable'))


if __name__ == '__main__':
    unittest.main()
