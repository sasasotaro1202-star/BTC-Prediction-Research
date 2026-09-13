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

        def fake_target(target):
            calls.append(target)
            if target == 'bad':
                raise RuntimeError('temporary public-source failure')
            return (123.0, 'test')

        with patch.object(settle, 'target_close_binance', side_effect=fake_target):
            result = settle.resolve_targets(['a', 'a', 'bad'], max_workers=2)

        self.assertEqual(sorted(calls), ['a', 'bad'])
        self.assertEqual(result['a'], (123.0, 'test'))
        self.assertEqual(result['bad'], (None, 'unavailable'))


if __name__ == '__main__':
    unittest.main()
