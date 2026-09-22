"""Research-only multi-scale feature/model replay against the frozen BTC Champion.

This lane is deliberately archive-only and non-promotable. It builds features
strictly from candles available at each timestamp, performs nested chronological
train/selection/gate evaluation with horizon-specific purge gaps, freezes the
selected recipe, then tests it across four contiguous future replay windows.
"""
from __future__ import annotations

import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for p in (ROOT, SRC):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from binance_history import binance_archive_rows
from bootstrap_train import make_features
from label_policy import CLASSES, direction_from_return
from model_compare import aligned, apply_temperature, metrics, _temperature

try:
    from lightgbm import LGBMClassifier
except ImportError:
    LGBMClassifier = None

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None

OUT = ROOT / "data" / "historical_research" / "multiscale_frozen_replay_oos.json"
ARCHIVE_ROWS = 60_000
FINAL_REPLAY_WINDOWS = 4
FINAL_HOLDOUT_FRAC = 0.20
TRAIN_FRAC = 0.60
SELECTION_FRAC = 0.10
GATE_FRAC = 0.10
REPLAY_MIN_ROWS = 500
MIN_TRAIN = 12_000
MIN_SELECTION = 1_000
MIN_GATE = 1_000
PURGE = {"5m": 5, "10m": 10}

BASE_FEATURES = (
    "ret_1m", "ret_3m", "ret_5m", "ret_10m", "acceleration",
    "volatility_5m", "volatility_10m", "range_position_10m",
    "body_1m", "upper_wick_1m", "lower_wick_1m", "volume_ratio",
    "volume_trend", "ema_gap_5m", "ema_gap_10m",
)
MULTISCALE_FEATURES = (
    "ret_15m", "ret_30m", "ret_60m",
    "volatility_30m", "volatility_60m",
    "range_position_30m", "range_position_60m",
    "volume_ratio_20m", "volume_z_30m",
    "trend_alignment", "momentum_curvature",
    "body_to_range", "wick_imbalance",
)
TIME_FEATURES = (
    "utc_minute_sin", "utc_minute_cos",
    "utc_week_sin", "utc_week_cos", "utc_weekend",
)
VARIANTS = {
    "base": BASE_FEATURES,
    "multiscale": BASE_FEATURES + MULTISCALE_FEATURES,
    "time": BASE_FEATURES + TIME_FEATURES,
    "full": BASE_FEATURES + MULTISCALE_FEATURES + TIME_FEATURES,
}


def _finite_float(value):
    try:
        x = float(value)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError, OverflowError):
        return None


def _stats(values):
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return 0.0, 0.0
    return float(np.mean(arr)), float(np.std(arr))


def _timestamp_features(ts_ms):
    dt = datetime.fromtimestamp(int(ts_ms) / 1000.0, timezone.utc)
    minute_of_day = dt.hour * 60 + dt.minute
    minute_of_week = dt.weekday() * 1440 + minute_of_day
    return (
        math.sin(2.0 * math.pi * minute_of_day / 1440.0),
        math.cos(2.0 * math.pi * minute_of_day / 1440.0),
        math.sin(2.0 * math.pi * minute_of_week / 10080.0),
        math.cos(2.0 * math.pi * minute_of_week / 10080.0),
        1.0 if dt.weekday() >= 5 else 0.0,
    )


def _vector(raw, i):
    window = raw[: i + 1]
    if len(window) < 60:
        raise ValueError("insufficient lookback")

    base = make_features(window)
    closes = np.asarray([r[4] for r in window], dtype=float)
    opens = np.asarray([r[1] for r in window], dtype=float)
    highs = np.asarray([r[2] for r in window], dtype=float)
    lows = np.asarray([r[3] for r in window], dtype=float)
    volumes = np.asarray([r[5] for r in window], dtype=float)
    price = float(closes[-1])

    ret15 = price / closes[-16] - 1.0
    ret30 = price / closes[-31] - 1.0
    ret60 = price / closes[-61] - 1.0
    rv30 = float(np.std(np.diff(closes[-31:]) / closes[-31:-1]))
    rv60 = float(np.std(np.diff(closes[-61:]) / closes[-61:-1]))

    hi30, lo30 = float(np.max(highs[-30:])), float(np.min(lows[-30:]))
    hi60, lo60 = float(np.max(highs[-60:])), float(np.min(lows[-60:]))
    rp30 = (price - lo30) / (hi30 - lo30) if hi30 > lo30 else 0.5
    rp60 = (price - lo60) / (hi60 - lo60) if hi60 > lo60 else 0.5

    mean5 = float(np.mean(volumes[-5:]))
    mean20 = float(np.mean(volumes[-20:]))
    mean30, std30 = _stats(volumes[-30:])
    vol20 = mean5 / max(1e-12, mean20)
    volz30 = (mean5 - mean30) / max(1e-12, std30)

    signs = np.sign(np.asarray([ret5 := base[2], ret15, ret30], dtype=float))
    trend_alignment = float(np.mean(signs))

    momentum_curvature = float(base[0] - ret5 / 3.0)
    last_range = max(1e-12, float(highs[-1] - lows[-1]))
    body_to_range = abs(float(closes[-1] - opens[-1])) / last_range
    upper_wick = float(highs[-1] - max(opens[-1], closes[-1]))
    lower_wick = float(min(opens[-1], closes[-1]) - lows[-1])
    wick_imbalance = (lower_wick - upper_wick) / last_range

    return (
        list(base)
        + [
            ret15, ret30, ret60, rv30, rv60, rp30, rp60,
            vol20, volz30, trend_alignment, momentum_curvature,
            body_to_range, wick_imbalance,
        ]
        + list(_timestamp_features(raw[i][0]))
    )


def _dataset():
    raw = binance_archive_rows(ARCHIVE_ROWS)
    rows = []
    horizon_steps = (5, 10)
    max_step = max(horizon_steps)

    for i in range(60, len(raw) - max_step):
        try:
            x = np.asarray(_vector(raw, i), dtype=float)
            if x.shape != (len(BASE_FEATURES) + len(MULTISCALE_FEATURES) + len(TIME_FEATURES),):
                continue
            if not np.isfinite(x).all():
                continue
            base_price = float(raw[i][4])
            rows.append({
                "ts": int(raw[i][0]),
                "x": x.tolist(),
                "y": {
                    "5m": direction_from_return(float(raw[i + 5][4]) / base_price - 1.0),
                    "10m": direction_from_return(float(raw[i + 10][4]) / base_price - 1.0),
                },
            })
        except (IndexError, ValueError, TypeError, FloatingPointError):
            continue

    if len(rows) < MIN_TRAIN + MIN_SELECTION + MIN_GATE + FINAL_REPLAY_WINDOWS * REPLAY_MIN_ROWS:
        raise RuntimeError(f"insufficient archive dataset: {len(rows)}")
    return rows


def factories():
    out = {
        "logreg": lambda: Pipeline([
            ("scale", StandardScaler()),
            ("model", LogisticRegression(C=0.20, max_iter=3000)),
        ]),
        "extra_trees": lambda: ExtraTreesClassifier(
            n_estimators=520,
            max_depth=11,
            min_samples_leaf=10,
            max_features="sqrt",
            random_state=42,
            n_jobs=-1,
        ),
        "hgb": lambda: HistGradientBoostingClassifier(
            max_iter=240,
            max_leaf_nodes=15,
            learning_rate=0.035,
            l2_regularization=1.5,
            random_state=42,
        ),
    }
    if LGBMClassifier is not None:
        out["lightgbm"] = lambda: LGBMClassifier(
            objective="multiclass",
            num_class=3,
            n_estimators=280,
            num_leaves=15,
            learning_rate=0.03,
            min_child_samples=30,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=2.0,
            reg_alpha=0.05,
            random_state=42,
            n_jobs=-1,
            verbosity=-1,
        )
    if XGBClassifier is not None:
        out["xgboost"] = lambda: XGBClassifier(
            objective="multi:softprob",
            num_class=3,
            n_estimators=260,
            max_depth=4,
            learning_rate=0.03,
            min_child_weight=10,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=2.0,
            reg_alpha=0.05,
            eval_metric="mlogloss",
            random_state=42,
            n_jobs=-1,
            tree_method="hist",
        )
    return out


def _fit_temperature(factory, train_rows, horizon, feature_idx):
    split = int(len(train_rows) * 0.80)
    if split < 3_000 or len(train_rows) - split < 500:
        return 1.0

    a = train_rows[:split]
    b = train_rows[split:]
    X_a = np.asarray([r["x"][:feature_idx] for r in a], dtype=float)
    y_a = np.asarray([r["y"][horizon] for r in a])
    X_b = np.asarray([r["x"][:feature_idx] for r in b], dtype=float)
    y_b = [r["y"][horizon] for r in b]

    model = factory()
    model.fit(X_a, y_a)
    p_b = aligned(model, X_b)
    return float(_temperature(p_b, y_b))


def _fit_and_score(factory, train_rows, eval_rows, horizon, feature_idx, temperature):
    model = factory()
    X_train = np.asarray([r["x"][:feature_idx] for r in train_rows], dtype=float)
    y_train = np.asarray([r["y"][horizon] for r in train_rows])
    model.fit(X_train, y_train)

    X_eval = np.asarray([r["x"][:feature_idx] for r in eval_rows], dtype=float)
    probs = apply_temperature(aligned(model, X_eval), temperature)
    return model, probs


def _slice_partition(rows, horizon):
    n = len(rows)
    selection_start = int(n * 0.60)
    gate_start = int(n * 0.70)
    replay_start = int(n * 0.80)
    purge = PURGE[horizon]

    # Purge the tail of each development slice so a forward h-minute label
    # cannot reach into the next evaluation regime.
    train_end = max(0, selection_start - purge)
    selection_end = max(selection_start, gate_start - purge)
    gate_end = max(gate_start, replay_start - purge)

    train = rows[:train_end]
    selection = rows[selection_start:selection_end]
    gate = rows[gate_start:gate_end]
    replay = rows[replay_start:]

    if min(len(train), len(selection), len(gate), len(replay)) < 1:
        raise RuntimeError("invalid partition")
    return train, selection, gate, replay


def _score_model(candidate, champion_probs, y):
    cm = metrics(y, candidate)
    pm = metrics(y, champion_probs)
    return {
        "candidate": cm,
        "champion": pm,
        "delta": {
            "accuracy": cm["accuracy"] - pm["accuracy"],
            "logloss": cm["logloss"] - pm["logloss"],
            "brier": cm["brier"] - pm["brier"],
        },
    }


def evaluate_horizon(rows, horizon):
    train, selection, gate, replay = _slice_partition(rows, horizon)
    if len(train) < MIN_TRAIN or len(selection) < MIN_SELECTION or len(gate) < MIN_GATE:
        return {
            "status": "DEFERRED",
            "reason": "development_slices_too_small",
            "n": len(rows),
        }

    champion = joblib.load(ROOT / "models" / f"{horizon}.joblib")
    champion_gate = aligned(
        champion,
        np.asarray([r["x"][:len(BASE_FEATURES)] for r in gate], dtype=float),
    )
    y_selection = [r["y"][horizon] for r in selection]
    y_gate = [r["y"][horizon] for r in gate]

    candidates = []
    for variant_name, feature_names in VARIANTS.items():
        feature_idx = len(feature_names)
        for model_name, factory in factories().items():
            try:
                temp = _fit_temperature(factory, train, horizon, feature_idx)
                _, selection_probs = _fit_and_score(
                    factory, train, selection, horizon, feature_idx, temp
                )
                sel_m = _score_model(
                    selection_probs,
                    aligned(
                        champion,
                        np.asarray([r["x"][:len(BASE_FEATURES)] for r in selection], dtype=float),
                    ),
                    y_selection,
                )
                candidates.append({
                    "variant": variant_name,
                    "model": model_name,
                    "feature_count": feature_idx,
                    "temperature": temp,
                    "selection": sel_m,
                })
            except (TypeError, ValueError, RuntimeError):
                continue

    if not candidates:
        return {"status": "DEFERRED", "reason": "no_candidate_completed", "n": len(rows)}

    selected = min(
        candidates,
        key=lambda x: (
            x["selection"]["delta"]["logloss"],
            x["selection"]["delta"]["brier"],
            -x["selection"]["delta"]["accuracy"],
        ),
    )

    factory = factories()[selected["model"]]
    temp = float(selected["temperature"])
    feature_idx = int(selected["feature_count"])
    _, gate_probs = _fit_and_score(factory, train + selection, gate, horizon, feature_idx, temp)
    gate_score = _score_model(
        gate_probs,
        champion_gate,
        y_gate,
    )

    gate_pass = bool(
        gate_score["delta"]["logloss"] <= -0.002
        and gate_score["delta"]["brier"] <= -0.001
        and gate_score["delta"]["accuracy"] >= -0.01
    )

    if not gate_pass:
        return {
            "status": "OK",
            "research_only": True,
            "production_changed": False,
            "promotion_evidence_eligible": False,
            "replay_holdout_protected": True,
            "selected": {
                "variant": selected["variant"],
                "model": selected["model"],
                "feature_count": feature_idx,
                "temperature": temp,
            },
            "selection": selected["selection"],
            "gate": gate_score,
            "gate_pass": False,
            "replay_gate": False,
            "windows": [],
        }

    # Freeze the chosen recipe once before any replay window is scored.
    frozen_model = factory()
    frozen_model.fit(
        np.asarray([r["x"][:feature_idx] for r in train + selection], dtype=float),
        np.asarray([r["y"][horizon] for r in train + selection]),
    )

    window_size = len(replay) // FINAL_REPLAY_WINDOWS
    windows = []
    for j in range(FINAL_REPLAY_WINDOWS):
        start = j * window_size
        end = len(replay) if j == FINAL_REPLAY_WINDOWS - 1 else (j + 1) * window_size
        block = replay[start:end]
        if len(block) < REPLAY_MIN_ROWS:
            continue

        y = [r["y"][horizon] for r in block]
        X = np.asarray([r["x"][:feature_idx] for r in block], dtype=float)
        candidate_probs = apply_temperature(aligned(frozen_model, X), temp)
        champion_probs = aligned(champion, np.asarray(
            [r["x"][:len(BASE_FEATURES)] for r in block], dtype=float
        ))
        score = _score_model(candidate_probs, champion_probs, y)

        windows.append({
            "n": len(block),
            "start_ts": int(block[0]["ts"]),
            "end_ts": int(block[-1]["ts"]),
            **score,
        })

    if len(windows) < 3:
        return {
            "status": "DEFERRED",
            "reason": "too_few_future_replay_windows",
            "selected": {
                "variant": selected["variant"],
                "model": selected["model"],
                "feature_count": feature_idx,
                "temperature": temp,
            },
            "gate": gate_score,
        }

    ll = np.asarray([w["delta"]["logloss"] for w in windows], dtype=float)
    br = np.asarray([w["delta"]["brier"] for w in windows], dtype=float)
    ac = np.asarray([w["delta"]["accuracy"] for w in windows], dtype=float)
    summary = {
        "windows": len(windows),
        "samples": int(sum(w["n"] for w in windows)),
        "mean_logloss_delta": float(ll.mean()),
        "mean_brier_delta": float(br.mean()),
        "mean_accuracy_delta": float(ac.mean()),
        "improved_logloss_ratio": float(np.mean(ll < 0)),
        "improved_brier_ratio": float(np.mean(br < 0)),
        "non_worse_accuracy_ratio": float(np.mean(ac >= -0.005)),
    }
    replay_gate = bool(
        summary["windows"] >= 3
        and summary["improved_logloss_ratio"] >= 0.75
        and summary["improved_brier_ratio"] >= 0.75
        and summary["mean_logloss_delta"] <= -0.003
        and summary["mean_brier_delta"] <= -0.0015
        and summary["non_worse_accuracy_ratio"] >= 0.75
    )

    return {
        "status": "OK",
        "schema_version": 1,
        "research_only": True,
        "production_changed": False,
        "promotion_evidence_eligible": False,
        "replay_holdout_protected": True,
        "replay_holdout_used_for_selection": False,
        "selection_candidate_count": len(candidates),
        "selected": {
            "variant": selected["variant"],
            "model": selected["model"],
            "feature_count": feature_idx,
            "temperature": temp,
        },
        "selection": selected["selection"],
        "gate": gate_score,
        "gate_pass": True,
        "summary": summary,
        "replay_gate": replay_gate,
        "windows": windows,
    }


def main():
    dataset = _dataset()
    payload = {
        "schema_version": 1,
        "research_only": True,
        "production_changed": False,
        "promotion_evidence_eligible": False,
        "policy": (
            "archive_only_multi_scale_feature_and_model_zoo_nested_selection_"
            "purged_gate_then_four_window_frozen_replay"
        ),
        "feature_groups": {
            "base": list(BASE_FEATURES),
            "multiscale": list(MULTISCALE_FEATURES),
            "time": list(TIME_FEATURES),
        },
        "horizons": {
            "5m": evaluate_horizon(dataset, "5m"),
            "10m": evaluate_horizon(dataset, "10m"),
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
