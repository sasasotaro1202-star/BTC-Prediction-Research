"""Research-only fresh Champion refresh against a future-like Binance archive holdout.

This lane evaluates production-schema candidates on a recent contiguous closed-candle
cohort, while keeping the final 20% chronologically frozen and descriptive-only.
No production registry/model artifact is modified.
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
import sys
for p in (ROOT, SRC):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from binance_history import binance_archive_rows
from bootstrap_train import make_features
from label_policy import CLASSES, direction_from_return
from model_compare import metrics, aligned, apply_temperature, _temperature

OUT = ROOT / "data" / "historical_research" / "archive_refresh_oos.json"
MODEL_DIR = ROOT / "models"

ARCHIVE_ROWS = 60000
FINAL_HOLDOUT_FRAC = 0.20
TRAIN_FRAC = 0.60
SELECTION_FRAC = 0.15
GATE_FRAC = 0.05
MIN_ROWS = 12000
MIN_GATE = 1000


def factories():
    return {
        "logreg": lambda: Pipeline([
            ("scale", StandardScaler()),
            ("model", LogisticRegression(C=0.25, max_iter=3000)),
        ]),
        "extra_trees": lambda: ExtraTreesClassifier(
            n_estimators=500,
            max_depth=12,
            min_samples_leaf=10,
            max_features="sqrt",
            random_state=42,
            n_jobs=-1,
        ),
        "random_forest": lambda: RandomForestClassifier(
            n_estimators=500,
            max_depth=10,
            min_samples_leaf=10,
            max_features="sqrt",
            random_state=42,
            n_jobs=-1,
        ),
        "hgb": lambda: HistGradientBoostingClassifier(
            max_iter=220,
            max_leaf_nodes=15,
            learning_rate=0.04,
            l2_regularization=1.5,
            random_state=42,
        ),
    }


def build_dataset():
    raw = binance_archive_rows(ARCHIVE_ROWS)
    out = []
    for i in range(30, len(raw) - 10):
        try:
            x = np.asarray(make_features(raw[: i + 1]), dtype=float)
            if x.shape != (15,) or not np.isfinite(x).all():
                continue
            base = float(raw[i][4])
            future = float(raw[i + 5][4]) / base - 1.0
            y5 = direction_from_return(future)
            future10 = float(raw[i + 10][4]) / base - 1.0
            y10 = direction_from_return(future10)
            ts = int(raw[i][0])
            out.append({
                "ts": ts,
                "x": x.tolist(),
                "y5m": y5,
                "y10m": y10,
            })
        except (IndexError, ValueError, TypeError, FloatingPointError):
            continue
    if len(out) < MIN_ROWS:
        raise RuntimeError(f"insufficient archive rows: {len(out)}")
    return out


def _fit_with_temperature(train_x, train_y, test_x, factory):
    split = int(len(train_x) * 0.80)
    if split < 300 or len(train_x) - split < 100:
        return None
    cal_model = factory()
    cal_model.fit(train_x[:split], train_y[:split])
    cal_probs = aligned(cal_model, test_x[:0]) if False else None
    raw_cal = cal_model.predict_proba(train_x[split:])
    cal = np.full((len(raw_cal), 3), 1e-7, dtype=float)
    for j, c in enumerate(cal_model.classes_):
        if str(c) in CLASSES:
            cal[:, CLASSES.index(str(c))] = raw_cal[:, j]
        elif isinstance(c, (int, np.integer)) and 0 <= int(c) < 3:
            cal[:, int(c)] = raw_cal[:, j]
    cal /= cal.sum(axis=1, keepdims=True)
    temp = _temperature(cal, train_y[split:].tolist())

    model = factory()
    model.fit(train_x, train_y)
    raw = model.predict_proba(test_x)
    p = np.full((len(raw), 3), 1e-7, dtype=float)
    for j, c in enumerate(model.classes_):
        if str(c) in CLASSES:
            p[:, CLASSES.index(str(c))] = raw[:, j]
        elif isinstance(c, (int, np.integer)) and 0 <= int(c) < 3:
            p[:, int(c)] = raw[:, j]
    p /= p.sum(axis=1, keepdims=True)
    return apply_temperature(p, temp), temp, model


def _load_champion(horizon):
    meta_path = MODEL_DIR / f"{horizon}.json"
    model_path = MODEL_DIR / f"{horizon}.joblib"
    if not meta_path.is_file() or not model_path.is_file():
        return None
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("features") and len(meta["features"]) != 15:
        return None
    model = joblib.load(model_path)
    return meta, model


def _champion_probs(model, X):
    p = model.predict_proba(X)
    out = np.full((len(X), 3), 1e-7, dtype=float)
    for j, c in enumerate(model.classes_):
        if str(c) in CLASSES:
            out[:, CLASSES.index(str(c))] = p[:, j]
        elif isinstance(c, (int, np.integer)) and 0 <= int(c) < 3:
            out[:, int(c)] = p[:, j]
    out /= out.sum(axis=1, keepdims=True)
    return out


def frozen_holdout_confirms(candidate: dict, champion: dict) -> bool:
    """Confirm no material degradation on the untouched final holdout."""
    return bool(
        float(candidate["logloss"]) <= float(champion["logloss"]) + 0.001
        and float(candidate["brier"]) <= float(champion["brier"]) + 0.001
        and float(candidate["accuracy"]) >= float(champion["accuracy"]) - 0.01
    )


def candidate_gate_eligible(candidate: dict, champion: dict, n: int) -> bool:
    """Apply the conservative archive-refresh promotion evidence gate."""
    champion_ll = max(abs(float(champion["logloss"])), 1e-12)
    champion_br = max(abs(float(champion["brier"])), 1e-12)
    ll_relative_gain = (
        float(champion["logloss"]) - float(candidate["logloss"])
    ) / champion_ll
    br_relative_gain = (
        float(champion["brier"]) - float(candidate["brier"])
    ) / champion_br
    return bool(
        float(candidate["logloss"]) <= float(champion["logloss"]) - 0.005
        and float(candidate["brier"]) <= float(champion["brier"]) - 0.002
        and ll_relative_gain >= 0.03
        and br_relative_gain >= 0.01
        and float(candidate["accuracy"]) >= float(champion["accuracy"]) - 0.01
        and int(n) >= MIN_GATE
    )


def _choose_candidate(results):
    valid = [r for r in results if r["gate_n"] >= MIN_GATE]
    if not valid:
        return None
    # Selection uses only the gate slice; final holdout is never consulted.
    return min(valid, key=lambda r: (r["gate"]["logloss"], r["gate"]["brier"], -r["gate"]["accuracy"]))


def evaluate_horizon(dataset, horizon):
    y_key = "y5m" if horizon == "5m" else "y10m"
    steps = 5 if horizon == "5m" else 10
    n = len(dataset)
    hold_start = int(n * (1.0 - FINAL_HOLDOUT_FRAC))
    gate_start = int(n * (1.0 - FINAL_HOLDOUT_FRAC - GATE_FRAC))
    selection_start = int(n * TRAIN_FRAC)
    train = dataset[:selection_start]
    selection = dataset[selection_start:gate_start]
    gate = dataset[gate_start:hold_start]
    holdout = dataset[hold_start:]

    X_train = np.asarray([r["x"] for r in train], float)
    y_train = np.asarray([r[y_key] for r in train])
    X_sel = np.asarray([r["x"] for r in selection], float)
    y_sel = np.asarray([r[y_key] for r in selection])
    X_gate = np.asarray([r["x"] for r in gate], float)
    y_gate = np.asarray([r[y_key] for r in gate])
    X_hold = np.asarray([r["x"] for r in holdout], float)
    y_hold = np.asarray([r[y_key] for r in holdout])

    champion = _load_champion(horizon)
    if champion is None:
        raise RuntimeError(f"{horizon}: production champion artifact unavailable")
    champion_meta, champion_model = champion
    champion_sel = _champion_probs(champion_model, X_sel)
    champion_gate = _champion_probs(champion_model, X_gate)
    champion_hold = _champion_probs(champion_model, X_hold)

    candidates = []
    for name, factory in factories().items():
        fitted = _fit_with_temperature(X_train, y_train, X_sel, factory)
        if fitted is None:
            continue
        _, temp, model_sel = fitted
        cand_sel = _champion_probs(model_sel, X_sel)
        # Refit on train + selection after freezing the selection choice.
        X_dev = np.concatenate([X_train, X_sel], axis=0)
        y_dev = np.concatenate([y_train, y_sel], axis=0)
        model = factory()
        model.fit(X_dev, y_dev)
        raw_gate = model.predict_proba(X_gate)
        gate_p = np.full((len(gate), 3), 1e-7)
        for j, c in enumerate(model.classes_):
            if str(c) in CLASSES:
                gate_p[:, CLASSES.index(str(c))] = raw_gate[:, j]
            elif isinstance(c, (int, np.integer)) and 0 <= int(c) < 3:
                gate_p[:, int(c)] = raw_gate[:, j]
        gate_p /= gate_p.sum(axis=1, keepdims=True)
        gate_p = apply_temperature(gate_p, temp)
        gate_metrics = metrics(y_gate.tolist(), gate_p)
        cand_sel_metrics = metrics(y_sel.tolist(), cand_sel)
        candidates.append({
            "name": name,
            "temperature": float(temp),
            "selection": cand_sel_metrics,
            "gate": gate_metrics,
            "gate_n": len(y_gate),
            "gate_vs_champion": {
                "accuracy": gate_metrics["accuracy"] - metrics(y_gate.tolist(), champion_gate)["accuracy"],
                "logloss": gate_metrics["logloss"] - metrics(y_gate.tolist(), champion_gate)["logloss"],
                "brier": gate_metrics["brier"] - metrics(y_gate.tolist(), champion_gate)["brier"],
            },
            "_model": model,
        })

    selected = _choose_candidate(candidates)
    if selected is None:
        return {"status": "DEFERRED", "n": n, "reason": "no_candidate_gate_evidence"}

    selected_name = selected["name"]
    hold_model = selected.pop("_model")
    # Final holdout uses the frozen candidate/temperature selected before it.
    raw_hold = hold_model.predict_proba(X_hold)
    hold_p = np.full((len(holdout), 3), 1e-7)
    for j, c in enumerate(hold_model.classes_):
        if str(c) in CLASSES:
            hold_p[:, CLASSES.index(str(c))] = raw_hold[:, j]
        elif isinstance(c, (int, np.integer)) and 0 <= int(c) < 3:
            hold_p[:, int(c)] = raw_hold[:, j]
    hold_p /= hold_p.sum(axis=1, keepdims=True)
    hold_p = apply_temperature(hold_p, selected["temperature"])

    hold_champ = metrics(y_hold.tolist(), champion_hold)
    hold_cand = metrics(y_hold.tolist(), hold_p)
    gate_champ = metrics(y_gate.tolist(), champion_gate)
    gate_cand = selected["gate"]
    # Candidate adoption evidence is intentionally dual-gated: require both
    # absolute and relative proper-score gains.
    champion_ll = max(abs(float(gate_champ["logloss"])), 1e-12)
    champion_br = max(abs(float(gate_champ["brier"])), 1e-12)
    ll_relative_gain = (float(gate_champ["logloss"]) - float(gate_cand["logloss"])) / champion_ll
    br_relative_gain = (float(gate_champ["brier"]) - float(gate_cand["brier"])) / champion_br
    eligible = candidate_gate_eligible(gate_cand, gate_champ, len(y_gate))

    return {
        "status": "OK",
        "horizon": horizon,
        "archive_rows": n,
        "train_n": len(train),
        "selection_n": len(selection),
        "gate_n": len(gate),
        "final_holdout_n": len(holdout),
        "selected_candidate": selected_name,
        "production_changed": False,
        "research_only": True,
        "final_holdout_protected": True,
        "final_holdout_used_for_selection": False,
        "selection": {c["name"]: {"metrics": c["selection"], "temperature": c["temperature"]} for c in candidates},
        "gate": {
            "champion": gate_champ,
            "candidate": gate_cand,
            "delta": {
                "accuracy": gate_cand["accuracy"] - gate_champ["accuracy"],
                "logloss": gate_cand["logloss"] - gate_champ["logloss"],
                "brier": gate_cand["brier"] - gate_champ["brier"],
                "logloss_relative_gain": ll_relative_gain,
                "brier_relative_gain": br_relative_gain,
            },
            "eligible_pending_frozen_holdout_confirmation": eligible,
        },
        "final_holdout": {
            "champion": hold_champ,
            "candidate": hold_cand,
            "confirmation_pass": frozen_holdout_confirms(hold_cand, hold_champ),
            "delta": {
                "accuracy": hold_cand["accuracy"] - hold_champ["accuracy"],
                "logloss": hold_cand["logloss"] - hold_champ["logloss"],
                "brier": hold_cand["brier"] - hold_champ["brier"],
            },
        },
        "candidate_temperatures": {c["name"]: c["temperature"] for c in candidates},
        "source": "Binance Vision contiguous closed 1m archive",
        "target_policy": "canonical_2bps",
        "candidate_model_version": f"archive_refresh_candidate.{selected_name}",
        "champion_model_version": champion_meta.get("model_version"),
    }


def main():
    dataset = build_dataset()
    payload = {
        "schema_version": 1,
        "research_only": True,
        "production_changed": False,
        "archive_rows": len(dataset),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": "archive_refresh_same_production_features_2bps_target_selection_gate_final_holdout",
        "horizons": {h: evaluate_horizon(dataset, h) for h in ("5m", "10m")},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
