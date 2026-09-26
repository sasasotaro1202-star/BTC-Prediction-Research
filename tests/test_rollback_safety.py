import tempfile
import unittest
from pathlib import Path

from src.rollback_safety import MODEL_NAMES, build_manifest, restore_from_backup


class TestRollbackSafety(unittest.TestCase):
    def test_restore_round_trip_is_checksum_verified(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            backup = root / "backup"
            target = root / "target"
            backup.mkdir()
            target.mkdir()
            for i, name in enumerate(MODEL_NAMES):
                data = (f"artifact-{i}".encode()) * 10
                (backup / name).write_bytes(data)
            manifest = build_manifest(backup)
            restored = restore_from_backup(backup, target, manifest)
            self.assertEqual(restored, manifest)
            self.assertEqual(build_manifest(target), manifest)

    def test_checksum_mismatch_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            backup = root / "backup"
            target = root / "target"
            backup.mkdir()
            target.mkdir()
            for i, name in enumerate(MODEL_NAMES):
                (backup / name).write_bytes(f"artifact-{i}".encode())
            manifest = build_manifest(backup)
            (backup / MODEL_NAMES[0]).write_bytes(b"tampered")
            with self.assertRaises(RuntimeError):
                restore_from_backup(backup, target, manifest)


if __name__ == "__main__":
    unittest.main()
