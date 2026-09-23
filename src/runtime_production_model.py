"""Resolve the exact production-safe model bundle for live prediction.

Research candidates are never selected here. This module only resolves artifacts
already marked as non-candidate production artifacts and, when available, checks
their version against the durable SQLite model registry.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np

try:
    from db import DB
    from feature_schema import FEATURES
except ModuleNotFoundError:
    from src.db import DB
    from src.feature_schema import FEATURES

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
CLASSES = ("DOWN", "FLAT", "UP")
HORIZONS = ("5m", "10m")


@dataclass(frozen=True)
class ProductionModel:
    horizon: str
    model_version: str
    artifact: Path
    metadata: dict[str, Any]
    sha256: str

    def load(self):
        model = joblib.load(self.artifact)
        classes = [str(x) for x in getattr(model, "classes_", [])]
        if classes != list(CLASSES):
            raise RuntimeError(f"production_model_classes_invalid:{self.horizon}")
        n_features = getattr(model, "n_features_in_", None)
        if n_features is not None and int(n_features) != len(FEATURES):
            raise RuntimeError(f"production_model_feature_count_invalid:{self.horizon}")
        probe = np.zeros((1, len(FEATURES)), dtype=float)
        probs = np.asarray(model.predict_proba(probe), dtype=float)
        if probs.shape != (1, len(CLASSES)) or not np.isfinite(probs).all():
            raise RuntimeError(f"production_model_runtime_probe_invalid:{self.horizon}")
        if np.any(probs < 0) or not np.isclose(float(probs.sum()), 1.0, atol=1e-9):
            raise RuntimeError(f"production_model_probability_contract_invalid:{self.horizon}")
        return model


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _registry_version(horizon: str) -> str | None:
    try:
        with sqlite3.connect(DB) as con:
            row = con.execute(
                "SELECT production_version FROM model_registry WHERE horizon=?",
                (horizon,),
            ).fetchone()
        return str(row[0]) if row and row[0] else None
    except (sqlite3.Error, OSError):
        return None


def resolve_production_model(horizon: str) -> ProductionModel:
    if horizon not in HORIZONS:
        raise ValueError(f"unsupported_horizon:{horizon}")

    model_path = MODEL_DIR / f"{horizon}.joblib"
    meta_path = MODEL_DIR / f"{horizon}.json"
    if not model_path.is_file() or model_path.stat().st_size <= 0:
        raise RuntimeError(f"production_model_missing:{horizon}")
    if not meta_path.is_file() or meta_path.stat().st_size <= 0:
        raise RuntimeError(f"production_metadata_missing:{horizon}")

    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    if metadata.get("horizon") != horizon:
        raise RuntimeError(f"production_metadata_horizon_mismatch:{horizon}")
    if metadata.get("artifact") != model_path.name:
        raise RuntimeError(f"production_metadata_artifact_mismatch:{horizon}")
    if metadata.get("candidate") is True:
        raise RuntimeError(f"production_runtime_refused_candidate:{horizon}")
    if metadata.get("classes") != list(CLASSES):
        raise RuntimeError(f"production_metadata_classes_invalid:{horizon}")
    if metadata.get("features") != list(FEATURES):
        raise RuntimeError(f"production_metadata_features_invalid:{horizon}")
    version = str(metadata.get("model_version", "")).strip()
    if not version:
        raise RuntimeError(f"production_metadata_version_missing:{horizon}")

    registry = _registry_version(horizon)
    if registry is not None and registry != version:
        raise RuntimeError(
            f"production_registry_metadata_mismatch:{horizon}:{registry}!={version}"
        )

    sha = _sha256(model_path)
    manifest_sha = str(metadata.get("artifact_sha256", "")).strip()
    if manifest_sha and manifest_sha != sha:
        raise RuntimeError(f"production_artifact_sha_mismatch:{horizon}")

    bundle = ProductionModel(
        horizon=horizon,
        model_version=version,
        artifact=model_path,
        metadata=metadata,
        sha256=sha,
    )
    bundle.load()
    return bundle


def resolve_all() -> dict[str, ProductionModel]:
    return {h: resolve_production_model(h) for h in HORIZONS}


def runtime_manifest() -> dict[str, Any]:
    resolved = resolve_all()
    return {
        "schema_version": 1,
        "selection_policy": "current_production_artifact_only;research_candidates_never_selected_at_runtime",
        "horizons": {
            h: {
                "model_version": b.model_version,
                "artifact": b.artifact.name,
                "artifact_sha256": b.sha256,
                "candidate": False,
            }
            for h, b in resolved.items()
        },
    }
