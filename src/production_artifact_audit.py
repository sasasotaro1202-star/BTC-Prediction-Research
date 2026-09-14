"""Validate and fingerprint the production BTC model artifacts.

The audit is deliberately observational: it never changes a production model and
only records a new state when the model/metadata fingerprint changes. This gives
rollback/provenance evidence without creating a five-minute commit churn loop.
"""
from __future__ import annotations
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
AUDIT = ROOT / "data" / "historical_research" / "production_artifact_audit.json"
CLASSES = ["DOWN", "FLAT", "UP"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def audit_one(horizon: str) -> dict:
    model = MODEL_DIR / f"{horizon}.joblib"
    meta = MODEL_DIR / f"{horizon}.json"
    if not model.is_file() or model.stat().st_size <= 0:
        raise SystemExit(f"missing production model: {model}")
    if not meta.is_file() or meta.stat().st_size <= 0:
        raise SystemExit(f"missing production metadata: {meta}")
    obj = json.loads(meta.read_text(encoding="utf-8"))
    if obj.get("horizon") != horizon:
        raise SystemExit(f"{horizon}: metadata horizon mismatch")
    if obj.get("artifact") != model.name:
        raise SystemExit(f"{horizon}: metadata artifact mismatch")
    if obj.get("classes") != CLASSES:
        raise SystemExit(f"{horizon}: metadata class order mismatch")
    features = obj.get("features")
    if not isinstance(features, list) or len(features) != 15 or len(set(features)) != 15:
        raise SystemExit(f"{horizon}: invalid feature metadata")
    return {
        "horizon": horizon,
        "model_version": str(obj.get("model_version", "")),
        "model_sha256": sha256(model),
        "metadata_sha256": sha256(meta),
        "model_bytes": model.stat().st_size,
        "metadata_bytes": meta.stat().st_size,
        "evaluation_milestone": obj.get("evaluation_milestone"),
    }


def main() -> None:
    records = [audit_one(h) for h in ("5m", "10m")]
    fingerprint = [{k: r[k] for k in ("horizon", "model_version", "model_sha256", "metadata_sha256", "model_bytes", "metadata_bytes", "evaluation_milestone")} for r in records]
    previous = None
    if AUDIT.exists():
        try:
            previous = json.loads(AUDIT.read_text(encoding="utf-8")).get("artifacts")
        except Exception:
            previous = None
    if previous != fingerprint:
        AUDIT.parent.mkdir(parents=True, exist_ok=True)
        AUDIT.write_text(json.dumps({
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            "artifacts": fingerprint,
            "policy": "sha256_fingerprint_only_on_production_artifact_or_metadata_change",
        }, indent=2) + "\n", encoding="utf-8")
        changed = True
    else:
        changed = False
    print(json.dumps({"ok": True, "changed": changed, "artifacts": fingerprint}, ensure_ascii=False))


if __name__ == "__main__":
    main()
