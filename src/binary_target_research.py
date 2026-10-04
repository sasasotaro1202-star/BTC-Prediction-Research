"""Research-only BTC binary-target experiment: UP vs DOWN, no FLAT class."""
from __future__ import annotations

import json
import math
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from bootstrap_train import make_features
    from binance_history import binance_archive_rows
    from label_policy import (
        BINARY_CLASSES,
        BINARY_TARGET_VERSION,
        binary_direction_from_return,
        binary_direction_from_prices,
    )
    from db import DB
    from feature_schema import FEATURES
    from model_compare import strict_pit_provenance_reason
except ModuleNotFoundError:
    from src.bootstrap_train import make_features
    from src.binance_history import binance_archive_rows
    from src.label_policy import (
        BINARY_CLASSES,
        BINARY_TARGET_VERSION,
        binary_direction_from_return,
        binary_direction_from_prices,
    )
    from src.db import DB
    from src.feature_schema import FEATURES
    from src.model_compare import strict_pit_provenance_reason

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "historical_research" / "binary_target_oos.json"

MIN_TRAIN = 5000
TEST_BLOCK = 500
HOLDOUT_FRAC = 0.20
MAX_ROWS = 30000
SEED = 42
Z95 = 1.959963984540054


def _contiguous_suffix(rows):
    ordered = sorted({int(r[0]): r for r in rows}.values(), key=lambda r: int(r[0]))
    if not ordered:
        return []
    start = len(ordered) - 1
    while start > 0 and int(ordered[start][0]) - int(ordered[start - 1][0]) == 60_000:
        start -= 1
    return ordered[start:]


def build_rows(raw_rows, horizon):
    steps = int(horizon.rstrip("m"))
    rows = _contiguous_suffix(raw_rows)
    out = []
    for i in range(30, len(rows) - steps):
        base_close = float(rows[i][4])
        target_close = float(rows[i + steps][4])
        future_return = target_close / base_close - 1.0
        x = np.asarray(make_features(rows[: i + 1]), dtype=float)
        if not np.isfinite(x).all():
            continue
        out.append({
            "id": f"binary:{BINARY_TARGET_VERSION}:{horizon}:{int(rows[i][0])}",
            "created": datetime.fromtimestamp(int(rows[i][0]) / 1000.0, timezone.utc).isoformat(),
            "target": datetime.fromtimestamp(int(rows[i + steps][0]) / 1000.0, timezone.utc).isoformat(),
            "x": x.tolist(),
            "y": binary_direction_from_return(future_return),
            "horizon": horizon,
            "target_version": BINARY_TARGET_VERSION,
        })
    return out[-MAX_ROWS:]


def _factories():
    return {
        "logreg": lambda: Pipeline([
            ("scale", StandardScaler()),
            ("model", LogisticRegression(C=0.5, max_iter=3000, random_state=SEED)),
        ]),
        "rf": lambda: RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=10,
            max_features="sqrt",
            random_state=SEED,
            n_jobs=-1,
        ),
        "extra_trees": lambda: ExtraTreesClassifier(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=10,
            max_features="sqrt",
            random_state=SEED,
            n_jobs=-1,
        ),
    }


def _wilson_ci(successes, n, z=Z95):
    n = int(n)
    if n <= 0:
        return [None, None]
    p = float(successes) / n
    denom = 1.0 + (z * z) / n
    center = (p + (z * z) / (2.0 * n)) / denom
    half = z * math.sqrt((p * (1.0 - p) / n) + (z * z) / (4.0 * n * n)) / denom
    return [max(0.0, center - half), min(1.0, center + half)]


def _effective_sample_size(values, max_lag=100):
    """Approximate ESS from positive autocorrelation of the binary hit series.

    This is a diagnostic, not a formal dependence model. Overlapping rolling
    horizons can make the raw row count materially overstate independent cases.
    """
    arr = np.asarray(values, dtype=float)
    n = int(arr.size)
    if n < 3:
        return float(n)
    centered = arr - float(arr.mean())
    variance = float(np.dot(centered, centered) / n)
    if variance <= 1e-15:
        return float(n)
    total_rho = 0.0
    for lag in range(1, min(max_lag, n - 2) + 1):
        cov = float(np.dot(centered[:-lag], centered[lag:]) / n)
        rho = cov / variance
        if not math.isfinite(rho) or rho <= 0.0:
            break
        total_rho += rho
    ess = n / max(1.0, 1.0 + 2.0 * total_rho)
    return float(max(1.0, min(float(n), ess)))


def _metrics(y_true, probs):
    y = np.asarray(y_true)
    p = np.clip(np.asarray(probs, dtype=float), 1e-6, 1 - 1e-6)
    cls = {c: i for i, c in enumerate(BINARY_CLASSES)}
    yi = np.asarray([cls[str(v)] for v in y], dtype=int)
    pred = np.where(p >= 0.5, "UP", "DOWN")
    conf = np.maximum(p, 1.0 - p)
    hit = (pred == y).astype(float)
    ece = 0.0
    for i in range(10):
        lo, hi = i / 10.0, (i + 1) / 10.0
        mask = (conf >= lo) & ((conf < hi) if hi < 1 else (conf <= hi))
        if mask.any():
            ece += float(mask.mean()) * abs(float(hit[mask].mean()) - float(conf[mask].mean()))
    one = yi.astype(float)
    brier = float(np.mean((p - one) ** 2))
    accuracy = float(np.mean(pred == y))
    return {
        "n": int(len(y)),
        "accuracy": accuracy,
        "accuracy_ci95": _wilson_ci(float(hit.sum()), len(y)),
        "effective_sample_size_accuracy": round(_effective_sample_size(hit), 3),
        "logloss": float(log_loss(yi, p, labels=[0, 1])),
        "brier": brier,
        "ece": ece,
        "up_rate": float(np.mean(y == "UP")),
    }


def _wfo(rows, end_index):
    factories = _factories()
    points = list(range(MIN_TRAIN, end_index, TEST_BLOCK))
    preds = {name: [] for name in factories}
    baseline_preds = []
    block_metrics = []
    for test_start in points:
        test_end = min(test_start + TEST_BLOCK, end_index)
        train_end = test_start - max(1, int(rows[test_start]["horizon"].rstrip("m")))
        train = rows[:train_end]
        test = rows[test_start:test_end]
        if len(train) < MIN_TRAIN or not test:
            continue
        X = np.asarray([r["x"] for r in train], dtype=float)
        y = np.asarray([r["y"] for r in train])
        Xt = np.asarray([r["x"] for r in test], dtype=float)
        yt = [r["y"] for r in test]
        block = {
            "train_end": train[-1]["created"],
            "test_start": test[0]["created"],
            "test_end": test[-1]["created"],
            "test_n": len(test),
        }
        base_p = float(np.mean(y == "UP"))
        baseline_preds.extend(zip(yt, np.full(len(yt), base_p)))
        block["frequency_baseline"] = _metrics(yt, np.full(len(yt), base_p))
        for name, factory in factories.items():
            model = factory()
            model.fit(X, y)
            p = np.asarray(model.predict_proba(Xt), dtype=float)
            classes = [str(c) for c in getattr(model, "classes_", [])]
            up_idx = classes.index("UP")
            up_p = np.clip(p[:, up_idx], 1e-6, 1 - 1e-6)
            preds[name].extend(zip(yt, up_p))
            block[name] = _metrics(yt, up_p)
        block_metrics.append(block)
    summary = {
        "frequency_baseline": (
            _metrics([y for y, _ in baseline_preds], [p for _, p in baseline_preds])
            if baseline_preds else {"status": "DEFERRED", "reason": "no_baseline_predictions", "n": 0}
        )
    }
    for name, items in preds.items():
        if not items:
            summary[name] = {"status": "DEFERRED", "reason": "no_oos_predictions", "n": 0}
        else:
            summary[name] = _metrics([y for y, _ in items], [p for _, p in items]) | {"status": "OK"}
    if not block_metrics:
        return {"status": "DEFERRED", "reason": "insufficient_wfo_history", "blocks": 0, "summary": summary}
    return {"status": "OK", "blocks": len(block_metrics), "summary": summary, "block_metrics": block_metrics}


def _wfo_diagnostics(wfo):
    """Summarize fold stability and effect size versus the same-block baseline."""
    blocks = list(wfo.get("block_metrics") or [])
    models = [name for name in _factories() if name in wfo.get("summary", {})]
    out = {"blocks": len(blocks), "models": {}}
    for name in models:
        observations = []
        for idx, block in enumerate(blocks):
            candidate = block.get(name)
            baseline = block.get("frequency_baseline")
            if not isinstance(candidate, dict) or not isinstance(baseline, dict):
                continue
            base_ll = float(baseline["logloss"])
            cand_ll = float(candidate["logloss"])
            base_brier = float(baseline["brier"])
            cand_brier = float(candidate["brier"])
            observations.append({
                "index": idx,
                "train_end": block.get("train_end"),
                "test_start": block.get("test_start"),
                "test_end": block.get("test_end"),
                "test_n": int(block.get("test_n", candidate.get("n", 0))),
                "logloss_delta": cand_ll - base_ll,
                "logloss_relative_improvement": (base_ll - cand_ll) / base_ll if base_ll else 0.0,
                "brier_delta": cand_brier - base_brier,
                "brier_relative_improvement": (base_brier - cand_brier) / base_brier if base_brier else 0.0,
                "accuracy_delta": float(candidate["accuracy"]) - float(baseline["accuracy"]),
            })
        if not observations:
            out["models"][name] = {"status": "DEFERRED", "reason": "no_comparable_blocks"}
            continue
        worst = max(observations, key=lambda x: x["logloss_delta"])
        newest = observations[-1]
        out["models"][name] = {
            "status": "OK",
            "n_blocks": len(observations),
            "logloss_non_degraded_blocks": sum(x["logloss_delta"] <= 0.0 for x in observations),
            "logloss_non_degraded_fraction": float(np.mean([x["logloss_delta"] <= 0.0 for x in observations])),
            "brier_non_degraded_blocks": sum(x["brier_delta"] <= 0.0 for x in observations),
            "brier_non_degraded_fraction": float(np.mean([x["brier_delta"] <= 0.0 for x in observations])),
            "accuracy_non_degraded_blocks": sum(x["accuracy_delta"] >= 0.0 for x in observations),
            "accuracy_non_degraded_fraction": float(np.mean([x["accuracy_delta"] >= 0.0 for x in observations])),
            "mean_logloss_relative_improvement": float(np.mean([x["logloss_relative_improvement"] for x in observations])),
            "worst_logloss_block": worst,
            "newest_block": newest,
        }
    return out


def load_live_primary_rows(horizon):
    actual_col = f"actual_price_{horizon}"
    with sqlite3.connect(DB) as con:
        rows = con.execute(
            f"""SELECT prediction_id, created_at_utc, base_price, feature_json,
                       {actual_col}, scenario_json
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
            feature_obj = json.loads(feature_json)
            x = [float(feature_obj[name]) for name in FEATURES]
            if not np.isfinite(np.asarray(x, dtype=float)).all():
                continue
            label = binary_direction_from_prices(float(base), float(actual))
            out.append({
                "id": int(pid),
                "created": str(created),
                "x": x,
                "y": label,
                "target_version": BINARY_TARGET_VERSION,
                "data_source": "live_binance_primary",
            })
        except (TypeError, ValueError, KeyError, json.JSONDecodeError, FloatingPointError):
            continue
    return out


def _evaluate_live_primary(horizon, archive_rows, live_rows, candidate):
    if not live_rows:
        return {
            "status": "DEFERRED",
            "reason": "no_strict_live_binance_primary_rows",
            "n": 0,
            "target_version": BINARY_TARGET_VERSION,
            "research_only": True,
            "production_changed": False,
        }

    first_live = min(r["created"] for r in live_rows)
    first_live_dt = datetime.fromisoformat(str(first_live).replace("Z", "+00:00"))
    # Knowledge-time firewall: the training label must have matured strictly
    # before the first live prediction cutoff. Creation time alone is not enough.
    train = []
    for r in archive_rows:
        try:
            target_dt = datetime.fromisoformat(str(r.get("target")).replace("Z", "+00:00"))
        except (TypeError, ValueError):
            continue
        if target_dt < first_live_dt:
            train.append(r)
    if len(train) < MIN_TRAIN:
        return {
            "status": "DEFERRED",
            "reason": "insufficient_pre_live_archive_training_rows",
            "n_live": len(live_rows),
            "n_train": len(train),
            "minimum_train": MIN_TRAIN,
            "target_version": BINARY_TARGET_VERSION,
            "research_only": True,
            "production_changed": False,
        }

    model = _factories()[candidate]()
    X = np.asarray([r["x"] for r in train], dtype=float)
    y = np.asarray([r["y"] for r in train])
    Xt = np.asarray([r["x"] for r in live_rows], dtype=float)
    yt = [r["y"] for r in live_rows]
    model.fit(X, y)
    p = np.asarray(model.predict_proba(Xt), dtype=float)
    classes = [str(c) for c in getattr(model, "classes_", [])]
    up_idx = classes.index("UP")
    up_p = np.clip(p[:, up_idx], 1e-6, 1 - 1e-6)
    metrics = _metrics(yt, up_p)
    return {
        "status": "OK",
        "target_version": BINARY_TARGET_VERSION,
        "classes": list(BINARY_CLASSES),
        "source": "live_binance_primary",
        "model": candidate,
        "n": len(live_rows),
        "train_n": len(train),
        "train_end": train[-1]["created"],
        "live_start": live_rows[0]["created"],
        "metrics": metrics,
        "pit": "strict_primary",
        "cross_source_overlap_guard": True,
        "research_only": True,
        "production_changed": False,
    }


def evaluate(horizon):
    raw = binance_archive_rows(MAX_ROWS + 100)
    rows = build_rows(raw, horizon)
    if len(rows) < MIN_TRAIN + TEST_BLOCK + 200:
        return {
            "status": "DEFERRED",
            "reason": "insufficient_binary_rows",
            "n_rows": len(rows),
            "horizon": horizon,
            "target_version": BINARY_TARGET_VERSION,
            "research_only": True,
            "production_changed": False,
        }
    development_end = max(MIN_TRAIN + TEST_BLOCK, int(len(rows) * (1.0 - HOLDOUT_FRAC)))
    development = rows[:development_end]
    holdout = rows[development_end:]
    wfo = _wfo(development, len(development))
    if wfo.get("status") != "OK":
        return {**wfo, "horizon": horizon, "n_rows": len(rows), "target_version": BINARY_TARGET_VERSION, "research_only": True, "production_changed": False}
    best = min(
        (name for name in wfo["summary"] if name != "frequency_baseline" and wfo["summary"][name].get("status") == "OK"),
        key=lambda name: (wfo["summary"][name]["logloss"], wfo["summary"][name]["brier"], -wfo["summary"][name]["accuracy"]),
    )
    X_dev = np.asarray([r["x"] for r in development], dtype=float)
    y_dev = np.asarray([r["y"] for r in development])
    X_hold = np.asarray([r["x"] for r in holdout], dtype=float)
    y_hold = [r["y"] for r in holdout]
    holdout_scores = {}
    for name, factory in _factories().items():
        model = factory()
        model.fit(X_dev, y_dev)
        p = model.predict_proba(X_hold)
        classes = [str(c) for c in getattr(model, "classes_", [])]
        up_idx = classes.index("UP")
        holdout_scores[name] = _metrics(y_hold, p[:, up_idx])
    live_rows = load_live_primary_rows(horizon)
    live_primary = _evaluate_live_primary(horizon, rows, live_rows, best)
    return {
        "status": "OK",
        "schema_version": 1,
        "research_only": True,
        "production_changed": False,
        "horizon": horizon,
        "target_version": BINARY_TARGET_VERSION,
        "classes": list(BINARY_CLASSES),
        "n_rows": len(rows),
        "development_rows": len(development),
        "holdout_rows": len(holdout),
        "holdout_protected": True,
        "binary_label_contract": {
            "flat_class_exists": False,
            "positive_return": "UP",
            "non_positive_return": "DOWN",
            "threshold": 0.0,
        },
        "oos_summary": wfo["summary"],
        "blocks": wfo["blocks"],
        "wfo_diagnostics": _wfo_diagnostics(wfo),
        "best_development_candidate": best,
        "holdout_descriptive_only": True,
        "holdout_scores": holdout_scores,
        "live_primary": live_primary,
        "promotion": {
            "decision": "HOLD",
            "eligible": False,
            "reason": "new_binary_target_requires_independent_longer_OOS_robustness_calibration_and_holdout_before_production",
        },
    }


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "schema_version": 1,
        "experiment": "btc_binary_target_v1",
        "target_version": BINARY_TARGET_VERSION,
        "research_only": True,
        "production_changed": False,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "horizons": {h: evaluate(h) for h in ("5m", "10m")},
    }
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "experiment": result["experiment"],
        "target_version": result["target_version"],
        "horizons": {
            h: {
                "status": v.get("status"),
                "n_rows": v.get("n_rows"),
                "best": v.get("best_development_candidate"),
                "holdout_protected": v.get("holdout_protected"),
            }
            for h, v in result["horizons"].items()
        },
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
