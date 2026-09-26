"""Fail-closed production rollback rehearsal for BTC artifacts.

The real production directory is never mutated by the CLI. The module provides
an atomic restore primitive for an explicitly supplied backup directory and the
workflow only exercises the primitive against temporary copies.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

HORIZONS = ("5m", "10m")
MODEL_NAMES = tuple(f"{h}.{suffix}" for h in HORIZONS for suffix in ("joblib", "json"))
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "historical_research" / "rollback_safety.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def build_manifest(model_dir: Path) -> dict[str, str]:
    manifest = {}
    for name in MODEL_NAMES:
        path = model_dir / name
        if not path.is_file() or path.stat().st_size <= 0:
            raise RuntimeError(f"rollback_source_missing:{name}")
        manifest[name] = sha256(path)
    return manifest


def restore_from_backup(
    backup_dir: Path,
    target_dir: Path,
    manifest: dict[str, str],
) -> dict[str, str]:
    target_dir.mkdir(parents=True, exist_ok=True)
    staged: dict[str, Path] = {}
    with tempfile.TemporaryDirectory(prefix="btc-rollback-") as tmp:
        tmpdir = Path(tmp)
        for name, expected in manifest.items():
            src = backup_dir / name
            if not src.is_file() or src.stat().st_size <= 0:
                raise RuntimeError(f"rollback_backup_missing:{name}")
            if sha256(src) != str(expected):
                raise RuntimeError(f"rollback_backup_checksum_mismatch:{name}")
            staged_path = tmpdir / name
            shutil.copy2(src, staged_path)
            if sha256(staged_path) != str(expected):
                raise RuntimeError(f"rollback_stage_checksum_mismatch:{name}")
            staged[name] = staged_path

        for name, staged_path in staged.items():
            destination = target_dir / name
            tmp_target = destination.with_suffix(destination.suffix + ".rollback.tmp")
            shutil.copy2(staged_path, tmp_target)
            os.replace(tmp_target, destination)
            if sha256(destination) != str(manifest[name]):
                raise RuntimeError(f"rollback_restore_checksum_mismatch:{name}")
    return {name: sha256(target_dir / name) for name in manifest}


def rehearse(model_dir: Path) -> dict:
    manifest = build_manifest(model_dir)
    with tempfile.TemporaryDirectory(prefix="btc-rollback-rehearsal-") as tmp:
        root = Path(tmp)
        backup = root / "backup"
        target = root / "target"
        backup.mkdir()
        target.mkdir()
        for name in MODEL_NAMES:
            shutil.copy2(model_dir / name, backup / name)
        before = {name: sha256(model_dir / name) for name in MODEL_NAMES}
        restored = restore_from_backup(backup, target, manifest)
        if restored != manifest:
            raise RuntimeError("rollback_rehearsal_manifest_mismatch")
        if build_manifest(target) != manifest:
            raise RuntimeError("rollback_rehearsal_target_mismatch")
        after = {name: sha256(model_dir / name) for name in MODEL_NAMES}
        if before != after:
            raise RuntimeError("rollback_rehearsal_mutated_source")
    return {
        "status": "PASS",
        "workflow_scope": "temporary_copy_only",
        "production_mutated": False,
        "artifact_count": len(MODEL_NAMES),
        "manifest": manifest,
        "mechanism": "checksum_verified_staged_copy_then_atomic_replace",
    }


def main() -> int:
    result = rehearse(ROOT / "models")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
