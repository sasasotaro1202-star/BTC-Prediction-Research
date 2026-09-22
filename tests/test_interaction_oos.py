import unittest

from src import interaction_oos


class InteractionOOSTimingTests(unittest.TestCase):
    def test_explicit_embargo_is_longer_than_target_purge(self):
        self.assertEqual(interaction_oos.PURGE["5m"], 5)
        self.assertEqual(interaction_oos.PURGE["10m"], 10)
        self.assertEqual(interaction_oos.EMBARGO["5m"], 60)
        self.assertEqual(interaction_oos.EMBARGO["10m"], 60)
        self.assertGreater(interaction_oos.EMBARGO["5m"], interaction_oos.PURGE["5m"])
        self.assertGreater(interaction_oos.EMBARGO["10m"], interaction_oos.PURGE["10m"])


if __name__ == "__main__":
    unittest.main()
