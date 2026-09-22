"""Research-only calibration zoo with frozen chronological replay.

The production Champion is frozen. Calibrators are trained/selected only on
chronological development slices, then frozen and evaluated on future replay
windows. No production artifact, registry, or prediction state is modified.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import sys

import joblib
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for p in (ROOT, SRC):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from binance_history import binance_archive_rows
from bootstrap_train import make_features
from label_policy import CLASSES, direction_from_return
from model_compare import aligned, apply_temperature, metrics, _temperature

OUT = ROOT / "data" / "historical_research" / "calibration_frozen_replay_oos.json"
ARCHIVE_ROWS = 60_000
MIN_ROWS = 12_000
PURGE = {"5m": 5, "10m": 10}
WINDOWS = 4
MIN_WINDOW = 500


def _dataset():
    raw = binance_archive_rows(ARCHIVE_ROWS)
    rows = []
    for i in range(30, len(raw) - 10):
        try:
            x = np.asarray(make_features(raw[: i + 1]), dtype=float)
            if x.shape != (15,) or not np.isfinite(x).all():
                continue
            base = float(raw[i][4])
            rows.append({
                "ts": int(raw[i][0]),
                "x": x.tolist(),
                "y": {
                    "5m": direction_from_return(float(raw[i + 5][4]) / base - 1.0),
                    "10m": direction_from_return(float(raw[i + 10][4]) / base - 1.0),
                },
            })
        except (IndexError, TypeError, ValueError, FloatingPointError):
            continue
    if len(rows) < MIN_ROWS:
        raise RuntimeError(f"insufficient archive rows for calibration replay: {len(rows)}")
    return rows


def _ece_mce(y, probs):
    idx = {c: i for i, c in enumerate(CLASSES)}
    yi = np.asarray([idx[v] for v in y], dtype=int)
    p = np.asarray(probs, dtype=float)
    conf = p.max(axis=1)
    hit = (p.argmax(axis=1) == yi).astype(float)
    ece = 0.0
    mce = 0.0
    for lo in np.linspace(0.0, 0.9, 10):
        hi = lo + 0.1
        mask = (conf >= lo) & ((conf < hi) | ((hi >= 1.0) & (conf <= hi)))
        if not mask.any():
            continue
        gap = abs(float(hit[mask].mean()) - float(conf[mask].mean()))
        ece += float(mask.mean()) * gap
        mce = max(mce, gap)
    return float(ece), float(mce)


def _quality(y, probs):
    base = metrics(y, probs)
    ece, mce = _ece_mce(y, probs)
    return {**base, "ece": ece, "mce": mce}


class IdentityCalibrator:
    def fit(self, x, y):
        return self

    def transform(self, probs):
        return np.asarray(probs, dtype=float)


class TemperatureCalibrator:
    def __init__(self):
        self.temperature = 1.0

    def fit(self, probs, y):
        self.temperature = float(_temperature(np.asarray(probs), list(y)))
        return self

    def transform(self, probs):
        return apply_temperature(np.asarray(probs, dtype=float), self.temperature)


class ClasswiseIsotonicCalibrator:
    def __init__(self):
        self.models = []

    def fit(self, probs, y):
        p = np.clip(np.asarray(probs, dtype=float), 1e-8, 1.0)
        yi = np.asarray([{c: i for i, c in enumerate(CLASSES)}[v] for v in y], dtype=int)
        self.models = []
        for k in range(3):
            model = IsotonicRegression(out_of_bounds="clip")
            model.fit(p[:, k], (yi == k).astype(float))
            self.models.append(model)
        return self

    def transform(self, probs):
        p = np.clip(np.asarray(probs, dtype=float), 1e-8, 1.0)
        q = np.column_stack([m.predict(p[:, k]) for k, m in enumerate(self.models)])
        q = np.clip(q, 1e-8, 1.0)
        return q / q.sum(axis=1, keepdims=True)


class VectorScalingCalibrator:
    """Low-dimensional multinomial logit recalibration of log probabilities."""
    def __init__(self):
        self.model = None

    def fit(self, probs, y):
        p = np.clip(np.asarray(probs, dtype=float), 1e-8, 1.0)
        X = np.log(p)
        yi = np.asarray([{c: i for i, c in enumerate(CLASSES)}[v] for v in y], dtype=int)
        self.model = LogisticRegression(C=0.25, max_iter=3000, random_state=42)
        self.model.fit(X, yi)
        return self

    def transform(self, probs):
        p = np.clip(np.asarray(probs, dtype=float), 1e-8, 1.0)
        return np.asarray(self.model.predict_proba(np.log(p)), dtype=float)


CALIBRATORS = {
    "raw": IdentityCalibrator,
    "temperature": TemperatureCalibrator,
    "isotonic": ClasswiseIsotonicCalibrator,
    "vector_scaling": VectorScalingCalibrator,
}


def _raw_probabilities(horizon, rows):
    model_path = ROOT / "models" / f"{horizon}.joblib"
    if not model_path.is_file():
        raise RuntimeError(f"{horizon}: Champion artifact missing")
    model = joblib.load(model_path)
    X = np.asarray([r["x"] for r in rows], dtype=float)
    return aligned(model, X)


def _fit_candidates(base_probs, y, split_a, split_b):
    train_p, train_y = base_probs[:split_a], y[:split_a]
    select_p, select_y = base_probs[split_a:split_b], y[split_a:split_b]
    candidates = []
    for name, ctor in CALIBRATORS.items():
        try:
            cal = ctor().fit(train_p, train_y)
            sel = cal.transform(select_p)
            q = _quality(select_y, sel)
            # Primary objective is log loss; calibration error breaks close ties.
            candidates.append({
                "name": name,
                "calibrator": cal,
                "selection": q,
            })
        except (TypeError, ValueError, RuntimeError):
            continue
    if not candidates:
        raise RuntimeError("no calibration candidate completed")
    selected = min(
        candidates,
        key=lambda x: (x["selection"]["logloss"], x["selection"]["brier"],
                       x["selection"]["ece"], -x["selection"]["accuracy"]),
    )
    return selected, candidates


def evaluate(horizon, rows):
    n = len(rows)
    purge = PURGE[horizon]
    dev_end = int(n * 0.80)
    train_end = int(dev_end * 0.65)
    # Keep a purge immediately before the selection boundary.
    train_end = max(1, train_end - purge)
    select_end = max(train_end + 1, dev_end - purge)

    y = [r["y"][horizon] for r in rows]
    probs = _raw_probabilities(horizon, rows)

    if train_end < 3_000 or (select_end - train_end) < 1_000 or (n - dev_end) < WINDOWS * MIN_WINDOW:
        return {
            "status": "DEFERRED",
            "reason": "insufficient_calibration_slices",
            "n": n,
        }

    selected, candidates = _fit_candidates(probs, y, train_end, select_end)
    method = selected["name"]

    # Fit the selected calibrator on all development observations only.
    frozen = CALIBRATORS[method]().fit(probs[:select_end], y[:select_end])
    replay = rows[dev_end:]
    replay_probs = probs[dev_end:]
    size = len(replay) // WINDOWS
    windows = []

    for j in range(WINDOWS):
        start = j * size
        end = len(replay) if j == WINDOWS - 1 else (j + 1) * size
        block_y = y[dev_end + start:dev_end + end]
        block_p = replay_probs[start:end]
        if len(block_y) < MIN_WINDOW:
            continue
        raw_q = _quality(block_y, block_p)
        cal_q = _quality(block_y, frozen.transform(block_p))
        windows.append({
            "n": len(block_y),
            "start_ts": int(replay[start]["ts"]),
            "end_ts": int(replay[end - 1]["ts"]),
            "raw": raw_q,
            "calibrated": cal_q,
            "delta": {
                "accuracy": cal_q["accuracy"] - raw_q["accuracy"],
                "logloss": cal_q["logloss"] - raw_q["logloss"],
                "brier": cal_q["brier"] - raw_q["brier"],
                "ece": cal_q["ece"] - raw_q["ece"],
                "mce": cal_q["mce"] - raw_q["mce"],
            },
        })

    if len(windows) < 3:
        return {
            "status": "DEFERRED",
            "reason": "too_few_replay_windows",
            "selected": method,
            "n": n,
        }

    ll = np.asarray([w["delta"]["logloss"] for w in windows], float)
    br = np.asarray([w["delta"]["brier"] for w in windows], float)
    ece = np.asarray([w["delta"]["ece"] for w in windows], float)
    ac = np.asarray([w["delta"]["accuracy"] for w in windows], float)
    replay_gate = bool(
        np.mean(ll < 0) >= 0.75
        and np.mean(br < 0) >= 0.75
        and np.mean(ece <= 0) >= 0.75
        and ac.mean() >= -0.005
    )

    return {
        "status": "OK",
        "schema_version": 1,
        "research_only": True,
        "production_changed": False,
        "promotion_evidence_eligible": False,
        "replay_holdout_protected": True,
        "replay_holdout_used_for_selection": False,
        "selected_calibrator": method,
        "selection": selected["selection"],
        "candidate_selection": {
            x["name"]: x["selection"] for x in candidates
        },
        "replay_summary": {
            "windows": len(windows),
            "samples": int(sum(w["n"] for w in windows)),
            "mean_logloss_delta": float(ll.mean()),
            "mean_brier_delta": float(br.mean()),
            "mean_ece_delta": float(ece.mean()),
            "improved_logloss_ratio": float(np.mean(ll < 0)),
            "improved_brier_ratio": float(np.mean(br < 0)),
            "improved_ece_ratio": float(np.mean(ece <= 0)),
            "non_worse_accuracy_ratio": float(np.mean(ac >= -0.005)),
        },
        "replay_gate": replay_gate,
        "windows": windows,
    }


def main():
    rows = _dataset()
    payload = {
        "schema_version": 1,
        "research_only": True,
        "production_changed": False,
        "promotion_evidence_eligible": False,
        "policy": "frozen_champion_calibration_zoo_development_selection_four_future_replay_windows",
        "horizons": {
            h: evaluate(h, rows)
            for h in ("5m", "10m")
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
