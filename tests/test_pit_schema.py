import sys
import unittest
from datetime import datetime, timezone, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pit_schema import validate_provenance  # noqa: E402


class TestPITSchema(unittest.TestCase):
    def base(self):
        t = datetime(2026, 9, 18, 0, 0, tzinfo=timezone.utc)
        return {
            "event_time": (t - timedelta(seconds=5)).isoformat(),
            "available_at": t.isoformat(),
            "retrieved_at": (t + timedelta(seconds=1)).isoformat(),
            "prediction_cutoff": (t + timedelta(seconds=2)).isoformat(),
            "publication_time": (t - timedelta(seconds=1)).isoformat(),
            "revision_time": None,
        }

    def test_valid_envelope(self):
        out = validate_provenance(self.base())
        self.assertEqual(out["event_time"], "2026-09-17T23:59:55+00:00")

    def test_requires_all_four_operational_timestamps(self):
        row = self.base()
        row.pop("available_at")
        with self.assertRaisesRegex(ValueError, "missing required PIT field: available_at"):
            validate_provenance(row)

    def test_rejects_available_after_cutoff(self):
        row = self.base()
        row["available_at"] = (datetime(2026, 9, 18, 0, 0, 3, tzinfo=timezone.utc)).isoformat()
        with self.assertRaisesRegex(ValueError, "available_at_after_prediction_cutoff"):
            validate_provenance(row)

    def test_rejects_naive_timestamp(self):
        row = self.base()
        row["event_time"] = "2026-09-17T23:59:55"
        with self.assertRaisesRegex(ValueError, "explicit timezone"):
            validate_provenance(row)

    def test_revision_may_be_later_than_cutoff(self):
        row = self.base()
        row["revision_time"] = (datetime(2026, 9, 18, 1, 0, tzinfo=timezone.utc)).isoformat()
        validate_provenance(row)


if __name__ == "__main__":
    unittest.main()
