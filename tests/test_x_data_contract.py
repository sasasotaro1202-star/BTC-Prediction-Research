import json
import sys
import tempfile
import unittest
from datetime import datetime, timezone, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from x_data_contract import available_at, deduplicate, normalize_post  # noqa: E402


class TestXDataContract(unittest.TestCase):
    def post(self, created=None, post_id="123"):
        created = created or datetime(2026, 9, 17, 12, 0, tzinfo=timezone.utc)
        return {
            "post_id": post_id,
            "author_handle": "dravenip",
            "created_at_utc": created.isoformat(),
            "text": "BTC market information",
            "source_url": f"https://x.com/dravenip/status/{post_id}",
        }

    def test_normalize_requires_timezone(self):
        p = self.post()
        p["created_at_utc"] = "2026-09-17T12:00:00"
        with self.assertRaises(ValueError):
            normalize_post(p, fetched_at_utc="2026-09-17T12:01:00Z")

    def test_rejects_future_post(self):
        p = self.post(datetime(2026, 9, 17, 12, 2, tzinfo=timezone.utc))
        with self.assertRaises(ValueError):
            normalize_post(p, fetched_at_utc="2026-09-17T12:01:00Z")

    def test_pit_boundary_is_inclusive(self):
        p = normalize_post(self.post(), fetched_at_utc="2026-09-17T12:01:00Z")
        self.assertTrue(available_at(p, "2026-09-17T12:00:00Z"))
        self.assertFalse(available_at(p, "2026-09-17T11:59:59Z"))

    def test_deduplicates_by_immutable_post_id(self):
        a = normalize_post(self.post(post_id="1"), fetched_at_utc="2026-09-17T12:01:00Z")
        b = normalize_post(self.post(post_id="1"), fetched_at_utc="2026-09-17T12:02:00Z")
        self.assertEqual(len(deduplicate([a, b])), 1)

    def test_atomic_ingestion_output_is_json(self):
        # Sanity check that normalized records remain JSON serializable.
        p = normalize_post(self.post(), fetched_at_utc="2026-09-17T12:01:00Z")
        self.assertIn("raw_sha256", json.loads(json.dumps(p)))


if __name__ == "__main__":
    unittest.main()
