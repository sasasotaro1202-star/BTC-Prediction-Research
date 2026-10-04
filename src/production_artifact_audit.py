"""Validate and fingerprint the production BTC model artifacts.

The audit is deliberately observational: it never changes a production model.
It fingerprints the files and also reloads each serialized estimator, checking
its runtime class contract and predict_proba output against metadata.
"""
from __future__ import annotations
import hashlib
import json
import math
import os
import subprocess
from datetime import datetime, timezone
from feature_schema import FEATURES
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
AUDIT = ROOT / "data" / "historical_research" / "production_artifact_audit.json"
CLASSES = ["DOWN", "FLAT", "UP"]
FEATURE_COUNT = len(FEATURES)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()



def current_workspace_sha(root: Path = ROOT) -> str:
    """Resolve the exact checkout under audit; fail closed when CI pins a different SHA."""
    try:
        sha = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        if os.environ.get("GITHUB_SHA"):
            raise SystemExit(
                f"unable to resolve workspace Git SHA under CI: {type(exc).__name__}"
            ) from exc
        return "LOCAL_UNPINNED"

    expected = os.environ.get("GITHUB_SHA")
    if expected and sha != expected:
        raise SystemExit(
            f"production artifact audit workspace SHA mismatch: expected={expected} actual={sha}"
        )
    return sha


def build_provenance(root: Path = ROOT) -> dict:
    """Bind the artifact audit to the exact checkout and policy/config files under evaluation."""
    workspace_sha = current_workspace_sha(root)
    policy_files = ("PROJECT_INSTRUCTIONS.md", "docs/PROJECT_SOURCE.md", "requirements.txt")
    return {
        "git_sha": os.environ.get("GITHUB_SHA") or workspace_sha,
        "workspace_sha": workspace_sha,
        "ref": os.environ.get("GITHUB_REF_NAME") or "LOCAL",
        "policy_files": {
            name: sha256(root / name) if (root / name).is_file() else None
            for name in policy_files
        },
    }

def _estimator_contract(model):
    """Find the fitted estimator contract without assuming a specific sklearn wrapper."""
    candidates = [model]
    for attr in ("named_steps", "steps"):
        value = getattr(model, attr, None)
        if isinstance(value, dict):
            candidates.extend(value.values())
        elif isinstance(value, (list, tuple)):
            candidates.extend(v for _, v in value if hasattr(v, "predict_proba"))
    for candidate in reversed(candidates):
        if hasattr(candidate, "predict_proba"):
            return candidate
    raise SystemExit("serialized production model has no predict_proba")


def audit_one(horizon: str) -> dict:
    model_path = MODEL_DIR / f"{horizon}.joblib"
    meta_path = MODEL_DIR / f"{horizon}.json"
    if not model_path.is_file() or model_path.stat().st_size <= 0:
        raise SystemExit(f"missing production model: {model_path}")
    if not meta_path.is_file() or meta_path.stat().st_size <= 0:
        raise SystemExit(f"missing production metadata: {meta_path}")

    obj = json.loads(meta_path.read_text(encoding="utf-8"))
    if obj.get("horizon") != horizon:
        raise SystemExit(f"{horizon}: metadata horizon mismatch")
    if obj.get("artifact") != model_path.name:
        raise SystemExit(f"{horizon}: metadata artifact mismatch")
    if obj.get("classes") != CLASSES:
        raise SystemExit(f"{horizon}: metadata class order mismatch")
    if obj.get("features") != list(FEATURES):
        raise SystemExit(f"{horizon}: production feature schema mismatch")
    if not isinstance(obj.get("model_version"), str) or not obj["model_version"].strip():
        raise SystemExit(f"{horizon}: missing production model_version")
    
    try:
        loaded = joblib.load(model_path)
        estimator = _estimator_contract(loaded)
        classes = [str(x) for x in getattr(estimator, "classes_", [])]
        if classes != CLASSES:
            raise SystemExit(f"{horizon}: serialized model classes mismatch: {classes}")
        n_features = getattr(loaded, "n_features_in_", getattr(estimator, "n_features_in_", None))
        if n_features is not None and int(n_features) != FEATURE_COUNT:
            raise SystemExit(f"{horizon}: serialized model feature count mismatch: {n_features}")
        probe = np.zeros((1, FEATURE_COUNT), dtype=float)
        probabilities = np.asarray(loaded.predict_proba(probe), dtype=float)
        if probabilities.shape != (1, len(CLASSES)):
            raise SystemExit(f"{horizon}: predict_proba shape mismatch: {probabilities.shape}")
        if not np.all(np.isfinite(probabilities)) or np.any(probabilities < 0):
            raise SystemExit(f"{horizon}: serialized model returned invalid probabilities")
        if not math.isclose(float(probabilities.sum()), 1.0, rel_tol=0.0, abs_tol=1e-9):
            raise SystemExit(f"{horizon}: serialized model probability sum mismatch")
    except SystemExit:
        raise
    except Exception as exc:
        raise SystemExit(f"{horizon}: serialized model reload/probe failed: {type(exc).__name__}: {exc}") from exc

    return {
        "horizon": horizon,
        "model_version": str(obj.get("model_version", "")),
        "model_sha256": sha256(model_path),
        "metadata_sha256": sha256(meta_path),
        "model_bytes": model_path.stat().st_size,
        "metadata_bytes": meta_path.stat().st_size,
        "evaluation_milestone": obj.get("evaluation_milestone"),
        "runtime_model_reload_ok": True,
        "runtime_classes": CLASSES,
        "runtime_feature_count": FEATURE_COUNT,
    }


def main() -> None:
    records = [audit_one(h) for h in ("5m", "10m")]
    provenance = build_provenance(ROOT)
    fingerprint = [{k: r[k] for k in (
        "horizon", "model_version", "model_sha256", "metadata_sha256",
        "model_bytes", "metadata_bytes", "evaluation_milestone",
        "runtime_model_reload_ok", "runtime_classes", "runtime_feature_count"
    )} for r in records]
    previous = None
    previous_provenance = None
    if AUDIT.exists():
        try:
            previous_obj = json.loads(AUDIT.read_text(encoding="utf-8"))
            previous = previous_obj.get("artifacts")
            previous_provenance = previous_obj.get("provenance")
        except Exception:
            previous = None
            previous_provenance = None
    if previous != fingerprint or previous_provenance != provenance:
        AUDIT.parent.mkdir(parents=True, exist_ok=True)
        AUDIT.write_text(json.dumps({
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            "artifacts": fingerprint,
            "provenance": provenance,
            "policy": "sha256_plus_runtime_reload_contract_plus_exact_feature_schema_plus_policy_provenance_plus_workspace_sha_binding",
        }, indent=2) + "\n", encoding="utf-8")
        changed = True
    else:
        changed = False
    print(json.dumps({"ok": True, "changed": changed, "artifacts": fingerprint, "provenance": provenance}, ensure_ascii=False))


if __name__ == "__main__":
    main()
