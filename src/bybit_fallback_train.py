"""Train a separately gated Bybit OHLCV fallback model for production continuity."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib

from bootstrap_train import fetch_bybit, build_dataset, train_one, CLASSES, FEATURES, TARGET_ROWS

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
TARGET = max(TARGET_ROWS, 30_000)


def main() -> int:
    rows = fetch_bybit(TARGET)
    if len(rows) < 10_000:
        raise RuntimeError(f"bybit_fallback_insufficient_history:{len(rows)}")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    published = []
    for horizon in ("5m", "10m"):
        X, y = build_dataset(rows, int(horizon[:-1]))
        if len(y) < 1_500 or len(set(y)) < 3:
            raise RuntimeError(f"bybit_fallback_insufficient_dataset:{horizon}:{len(y)}")
        best, baseline, holdout_n, validation_results = train_one(X, y)
        _, _, _, name, model, _, score = best
        if score["logloss"] >= baseline["logloss"] - 0.005:
            raise RuntimeError(
                f"bybit_fallback_holdout_rejected:{horizon}:"
                f"candidate_logloss={score['logloss']:.8f}:baseline={baseline['logloss']:.8f}"
            )
        artifact = MODEL_DIR / f"bybit_{horizon}.joblib"
        metadata = MODEL_DIR / f"bybit_{horizon}.json"
        joblib.dump(model, artifact)
        meta = {
            "model_version": f"bybit_fallback.{name}.v1",
            "source": "Bybit linear BTCUSDT 1m closed klines",
            "horizon": horizon,
            "classes": list(model.classes_),
            "features": FEATURES,
            "artifact": artifact.name,
            "selection_method": "chronological_17pct_validation",
            "holdout_n": holdout_n,
            "holdout_metrics": score,
            "baseline_metrics": baseline,
            "validation_metrics": [
                {"model": r[3], "logloss": r[0], "brier": r[1], "accuracy": -r[2]}
                for r in validation_results
            ],
            "calibration": "not_calibrated_until_fallback_live_settlement",
            "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        metadata.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        published.append(meta)
    print(json.dumps({"ok": True, "rows": len(rows), "models": published}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
