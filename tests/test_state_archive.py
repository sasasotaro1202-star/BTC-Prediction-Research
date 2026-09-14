import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from state_archive import compress_db, restore_db  # noqa: E402


class TestStateArchive(unittest.TestCase):
    def test_round_trip_preserves_sqlite_bytes(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            src = root / 'predictions.db'
            archive = root / 'predictions.db.gz'
            restored = root / 'restored.db'
            with sqlite3.connect(src) as con:
                con.execute('CREATE TABLE predictions (id INTEGER PRIMARY KEY, value TEXT)')
                con.executemany('INSERT INTO predictions(value) VALUES (?)', [('a',), ('b',), ('c',)])
                con.commit()
            original = src.read_bytes()
            info = compress_db(src, archive)
            self.assertGreater(info['bytes'], 0)
            src.unlink()
            restore_db(archive, restored)
            self.assertEqual(restored.read_bytes(), original)
            with sqlite3.connect(restored) as con:
                self.assertEqual(con.execute('SELECT COUNT(*) FROM predictions').fetchone()[0], 3)

    def test_invalid_archive_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            bad = root / 'bad.gz'
            out = root / 'out.db'
            bad.write_bytes(b'not-a-btc-state')
            with self.assertRaises(ValueError):
                restore_db(bad, out)


if __name__ == '__main__':
    unittest.main()
