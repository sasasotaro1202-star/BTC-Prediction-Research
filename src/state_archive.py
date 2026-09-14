"""Repository-safe compression helpers for BTC research state.

The prediction SQLite database is preserved in full; compression is only a
transport/storage format for GitHub, never a data-pruning mechanism.
"""
from __future__ import annotations
import gzip
import hashlib
from pathlib import Path


MAGIC = b"BTCSTATE1\n"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def compress_db(src: Path, dst: Path) -> dict:
    if not src.exists() or src.stat().st_size == 0:
        raise ValueError(f"source database missing or empty: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    with src.open("rb") as fin, tmp.open("wb") as raw:
        raw.write(MAGIC)
        with gzip.GzipFile(fileobj=raw, mode="wb", compresslevel=9, mtime=0) as gz:
            for chunk in iter(lambda: fin.read(1024 * 1024), b""):
                gz.write(chunk)
    tmp.replace(dst)
    return {"bytes": dst.stat().st_size, "sha256": sha256_file(dst)}


def restore_db(src: Path, dst: Path) -> None:
    if not src.exists() or src.stat().st_size == 0:
        raise ValueError(f"archive missing or empty: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    with src.open("rb") as raw:
        magic = raw.read(len(MAGIC))
        if magic != MAGIC:
            raise ValueError("invalid BTC state archive magic")
        with gzip.GzipFile(fileobj=raw, mode="rb") as gz, tmp.open("wb") as out:
            for chunk in iter(lambda: gz.read(1024 * 1024), b""):
                out.write(chunk)
    if tmp.stat().st_size == 0:
        tmp.unlink()
        raise ValueError("restored database is empty")
    tmp.replace(dst)
