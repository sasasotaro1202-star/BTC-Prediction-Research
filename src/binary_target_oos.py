"""Research-only UP/DOWN binary target evaluation.

Target version: binary_sign_v1
- UP when future return > 0
- DOWN otherwise
- no FLAT class

This pipeline never mutates production artifacts. It performs chronological
walk-forward evaluation on verified historical candles, protects a frozen
holdout from model selection, and records a separate live-Binance-primary
evaluation when enough strictly PIT-qualified settled rows exist.
"""
from __future__ import annotations

import json
import math
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

try:
    from .db import DB
    from .feature_schema import FEATURES
    from .label_policy import BINARY_CLASSES, BINARY_TARGET_VERSION, binary_direction_from_prices
    from .bootstrap_train import make_features
    from .binance_history import binance_archive_rows
    from .model_compare import strict_pit_provenance_reason
except ImportError:
    from db import DB
    from feature_schema import FEATURES
    from label_policy import BINARY_CLASSES, BINARY_TARGET_VERSION, binary_direction_from_prices
    from bootstrap_train import make_features
    from binance_history import binance_archive_rows
    from model_compare import strict_pit_provenance_reason

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "historical_research"
MODEL_DIR = ROOT / "models"

HORIZONS = ("5m", "10m")
ARCHIVE_ROWS = 30000
MIN_TRAIN = 10000
TEST_BLOCK = 2000
FINAL_HOLDOUT_FRAC = 0.15
MIN_LIVE_ROWS = 300
SEED = 42


def _normalize(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-8, 1.0)
    return p / p.sum(axis=1, keepdims=True)


def _ece(y, p):
    y = np.asarray(y, dtype=int)
    p = _normalize(p)
    conf = p.max(axis=1)
    hit = (p.argmax(axis=1) == y).astype(float)
    out = 0.0
    for i in range(10):
        lo, hi = i / 10.0, (i + 1) / 10.0
        mask = (conf >= lo) & ((conf <= hi) if i == 9 else (conf < hi))
        if mask.any():
            out += float(mask.mean()) * abs(float(hit[mask].mean()) - float(conf[mask].mean()))
    return out


def metrics(y, p):
    yi = np.asarray(y, dtype=int)
    p = _normalize(p)
    return {
        "n": int(len(yi)),
        "accuracy": float((p.argmax(1) == yi).mean()),
        "logloss": float(log_loss(yi, p, labels=[0, 1])),
        "brier": float(np.mean(np.sum((p - np.eye(2)[yi]) ** 2, axis=1))),
        "ece": float(_ece(yi, p)),
    }


def _temperature_fit(y, p):
    if len(y) < 100 or len(set(y)) < 2:
        return 1.0
    y = np.asarray(y, dtype=int)
    p = _normalize(p)
    logits = np.log(np.clip(p, 1e-8, 1.0))
    best_t, best_ll = 1.0, float("inf")
    for t in np.linspace(0.5, 3.0, 101):
        z = logits / float(t)
        z -= z.max(axis=1, keepdims=True)
        q = np.exp(z)
        q /= q.sum(axis=1, keepdims=True)
        ll = float(log_loss(y, q, labels=[0, 1]))
        if ll < best_ll:
            best_ll, best_t = ll, float(t)
    return best_t


def _apply_temperature(p, t):
    if float(t) == 1.0:
        return _normalize(p)
    p = _normalize(p)
    z = np.log(np.clip(p, 1e-8, 1.0)) / float(t)
    z -= z.max(axis=1, keepdims=True)
    q = np.exp(z)
    q /= q.sum(axis=1, keepdims=True)
    return q


def _factories():
    return {
        "logreg": lambda: Pipeline([
            ("scale", StandardScaler()),
            ("model", LogisticRegression(C=0.5, max_iter=3000, random_state=SEED)),
        ]),
        "rf": lambda: RandomForestClassifier(
            n_estimators=400,
            max_depth=8,
            min_samples_leaf=10,
            max_features="sqrt",
            random_state=SEED,
            n_jobs=-1,
        ),
        "extra_trees": lambda: ExtraTreesClassifier(
            n_estimators=400,
            max_depth=8,
            min_samples_leaf=10,
            max_features="sqrt",
            random_state=SEED,
            n_jobs=-1,
        ),
        "hgb": lambda: HistGradientBoostingClassifier(
            max_iter=300,
            max_leaf_nodes=15,
            learning_rate=0.04,
            l2_regularization=1.0,
            random_state=SEED,
        ),
    }


def _contiguous(raw):
    ordered = sorted({int(r[0]): r for r in raw}.values(), key=lambda r: int(r[0]))
    if not ordered:
        return []
    start = len(ordered) - 1
    while start > 0 and int(ordered[start][0]) - int(ordered[start - 1][0]) == 60000:
        start -= 1
    return ordered[start:]


def build_archive_rows(horizon):
    steps = int(horizon.rstrip("m"))
    raw = binance_archive_rows(ARCHIVE_ROWS + 40)
    raw = _contiguous(raw)
    out = []
    for i in range(30, len(raw) - steps):
        target = raw[i + steps]
        if int(target[0]) - int(raw[i][0]) != steps * 60000:
            continue
        try:
            x = np.asarray(make_features(raw[: i + 1]), dtype=float)
            if len(x) != len(FEATURES) or not np.isfinite(x).all():
                continue
            y = 1 if float(target[4]) > float(raw[i][4]) else 0
            out.append({
                "created": datetime.fromtimestamp(int(raw[i][0]) / 1000.0, timezone.utc).isoformat(),
                "x": x.tolist(),
                "y": int(y),
                "data_source": "binance_vision",
            })
        except (IndexError, TypeError, ValueError, FloatingPointError, OverflowError):
            continue
    return out


def build_live_primary_rows(horizon):
    actual_col = f"actual_price_{horizon}"
    with sqlite3.connect(DB) as con:
        rows = con.execute(
            f"""SELECT prediction_id, created_at_utc, base_price, feature_json, actual_price_{horizon}, scenario_json
                FROM predictions
                WHERE {actual_col} IS NOT NULL
                ORDER BY created_at_utc, prediction_id"""
        ).fetchall()
    out = []
    for pid, created, base, feature_json, actual, scenario_text in rows:
        try:
            scenario = json.loads(scenario_text or "{}")
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        if str(scenario.get("production_mode", "")) != "binance_primary":
            continue
        if strict_pit_provenance_reason(scenario, created) is not None:
            continue
        try:
            f = json.loads(feature_json)
            x = [float(f[name]) for name in FEATURES]
            if not np.isfinite(np.asarray(x, dtype=float)).all():
                continue
            y_name = binary_direction_from_prices(float(base), float(actual))
            out.append({
                "id": int(pid),
                "created": str(created),
                "x": x,
                "y": 1 if y_name == "UP" else 0,
                "data_source": "live_binance_primary",
            })
        except (TypeError, ValueError, KeyError, json.JSONDecodeError, FloatingPointError):
            continue
    return out


def evaluate_archive(horizon, rows):
    if len(rows) < MIN_TRAIN + TEST_BLOCK * 3:
        return {
            "status": "DEFERRED",
            "reason": "insufficient_archive_rows",
            "n": len(rows),
            "target_version": BINARY_TARGET_VERSION,
            "classes": list(BINARY_CLASSES),
            "research_only": True,
            "production_changed": False,
        }

    holdout_cut = max(MIN_TRAIN + TEST_BLOCK, int(len(rows) * (1.0 - FINAL_HOLDOUT_FRAC)))
    dev = rows[:holdout_cut]
    holdout = rows[holdout_cut:]
    factories = _factories()
    block_results = {name: [] for name in factories}
    prior_oos = {name: {"y": [], "p": []} for name in factories}
    points = list(range(MIN_TRAIN, len(dev), TEST_BLOCK))

    for test_start in points:
        test = dev[test_start:min(test_start + TEST_BLOCK, len(dev))]
        train = dev[:test_start]
        if len(test) == 0:
            continue
        X = np.asarray([r["x"] for r in train], dtype=float)
        y = np.asarray([r["y"] for r in train], dtype=int)
        Xt = np.asarray([r["x"] for r in test], dtype=float)
        yt = np.asarray([r["y"] for r in test], dtype=int)
        for name, factory in factories.items():
            model = factory()
            model.fit(X, y)
            p_raw = _normalize(model.predict_proba(Xt))
            prior = prior_oos[name]
            t = _temperature_fit(prior["y"], np.asarray(prior["p"])) if len(prior["y"]) else 1.0
            p_cal = _apply_temperature(p_raw, t)
            block_results[name].append({
                "test_start": test_start,
                "n": len(test),
                "temperature": float(t),
                "raw": metrics(yt, p_raw),
                "calibrated": metrics(yt, p_cal),
            })
            prior["y"].extend(yt.tolist())
            prior["p"].extend(p_raw.tolist())

    summary = {}
    for name, blocks in block_results.items():
        summary[name] = {
            "blocks": len(blocks),
            "n": int(sum(b["n"] for b in blocks)),
            "raw": metrics(
                np.concatenate([
                    np.asarray([0] * 0, dtype=int)
                ]) if False else
                np.asarray([
                    # reconstructing labels from block metrics is impossible; aggregate below
                ], dtype=int),
                np.empty((0, 2)),
            ) if False else None,
            "mean_block_accuracy": float(np.mean([b["calibrated"]["accuracy"] for b in blocks])),
            "mean_block_logloss": float(np.mean([b["calibrated"]["logloss"] for b in blocks])),
            "mean_block_brier": float(np.mean([b["calibrated"]["brier"] for b in blocks])),
            "mean_block_ece": float(np.mean([b["calibrated"]["ece"] for b in blocks])),
            "recent_block_logloss": float(blocks[-1]["calibrated"]["logloss"]) if blocks else None,
        }

    # Candidate selection uses development only.
    champion = min(
        summary,
        key=lambda name: (
            summary[name]["mean_block_logloss"],
            summary[name]["mean_block_brier"],
        ),
    )

    # Frozen holdout: refit selected model on all development rows; no holdout tuning.
    Xdev = np.asarray([r["x"] for r in dev], dtype=float)
    ydev = np.asarray([r["y"] for r in dev], dtype=int)
    Xho = np.asarray([r["x"] for r in holdout], dtype=float)
    yho = np.asarray([r["y"] for r in holdout], dtype=int)
    model = factories[champion]()
    model.fit(Xdev, ydev)
    p_holdout_raw = _normalize(model.predict_proba(Xho))
    holdout_temp = _temperature_fit(ydev[-min(2000, len(ydev)):], _normalize(model.predict_proba(Xdev[-min(2000, len(ydev)):])) if len(ydev) else np.empty((0, 2)))
    p_holdout_cal = _apply_temperature(p_holdout_raw, holdout_temp)

    baseline = {
        "accuracy": float(max(np.mean(yho == 0), np.mean(yho == 1))),
        "logloss": float(
            log_loss(
                yho,
                np.tile([float(np.mean(ydev == 0)), float(np.mean(ydev == 1))], (len(yho), 1)),
                labels=[0, 1],
            )
        ),
    }

    return {
        "status": "OK",
        "schema_version": 1,
        "target_version": BINARY_TARGET_VERSION,
        "classes": list(BINARY_CLASSES),
        "horizon": horizon,
        "research_only": True,
        "production_changed": False,
        "source": "binance_vision",
        "pit_scope": "archive research; not strict production PIT evidence",
        "n_rows": len(rows),
        "development_rows": len(dev),
        "frozen_holdout_rows": len(holdout),
        "development_blocks": len(points),
        "candidate_selection": {
            "champion": champion,
            "summary": summary,
        },
        "frozen_holdout": {
            "model": champion,
            "temperature": float(holdout_temp),
            "metrics_raw": metrics(yho, p_holdout_raw),
            "metrics_calibrated": metrics(yho, p_holdout_cal),
            "baseline": baseline,
            "protected": True,
        },
        "promotion": {
            "decision": "HOLD",
            "eligible": False,
            "reason": "binary_target_requires_strict_live_pit_oos_robustness_calibration_and_shadow_before_production",
        },
        "blocks_detail": block_results,
    }


def evaluate_live_primary(horizon, archive_result, live_rows):
    if len(live_rows) < MIN_LIVE_ROWS:
        return {
            "status": "DEFERRED",
            "reason": "insufficient_strict_live_binance_primary_rows",
            "n": len(live_rows),
            "minimum": MIN_LIVE_ROWS,
            "target_version": BINARY_TARGET_VERSION,
            "research_only": True,
        }
    champion = archive_result["candidate_selection"]["champion"]
    model = _factories()[champion]()
    train_rows = build_archive_rows(horizon)
    X = np.asarray([r["x"] for r in train_rows], dtype=float)
    y = np.asarray([r["y"] for r in train_rows], dtype=int)
    model.fit(X, y)
    Xt = np.asarray([r["x"] for r in live_rows], dtype=float)
    yt = np.asarray([r["y"] for r in live_rows], dtype=int)
    p = _normalize(model.predict_proba(Xt))
    return {
        "status": "OK",
        "target_version": BINARY_TARGET_VERSION,
        "classes": list(BINARY_CLASSES),
        "source": "live_binance_primary",
        "n": len(live_rows),
        "model": champion,
        "metrics": metrics(yt, p),
        "pit": "strict_primary",
        "research_only": True,
        "production_changed": False,
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "experiment": "btc_binary_target_up_down_v1",
        "target_version": BINARY_TARGET_VERSION,
        "classes": list(BINARY_CLASSES),
        "horizons": list(HORIZONS),
        "research_only": True,
        "production_changed": False,
        "production_change": "FORBIDDEN_IN_THIS_WORKFLOW",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "results": {},
    }
    for h in HORIZONS:
        try:
            archive_rows = build_archive_rows(h)
            archive_result = evaluate_archive(h, archive_rows)
            live_rows = build_live_primary_rows(h)
            live_result = (
                evaluate_live_primary(h, archive_result, live_rows)
                if archive_result.get("status") == "OK"
                else {"status": "DEFERRED", "reason": "archive_evaluation_not_ready", "n": len(live_rows)}
            )
            result = {
                "archive_oos": archive_result,
                "live_primary": live_result,
            }
        except Exception as exc:
            result = {
                "status": "DEFERRED",
                "reason": f"evaluation_error:{type(exc).__name__}:{exc}",
                "target_version": BINARY_TARGET_VERSION,
                "classes": list(BINARY_CLASSES),
                "research_only": True,
                "production_changed": False,
            }
        manifest["results"][h] = result
        (OUT_DIR / f"binary_target_oos_{h}.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    (OUT_DIR / "binary_target_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
