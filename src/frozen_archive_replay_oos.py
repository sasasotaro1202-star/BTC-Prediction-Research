"""Research-only multi-window frozen replay for BTC archive candidates.

Candidate selection and calibration are completed before the replay windows.
The replay windows are never used to tune the candidate. The same fitted
candidate and fixed temperature are evaluated across several contiguous future
windows, alongside the current production Champion artifact. This is designed
to reject one-window archive improvements that disappear immediately out of
sample.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
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

OUT = ROOT / "data" / "historical_research" / "frozen_archive_replay_oos.json"
MODEL_DIR = ROOT / "models"

ARCHIVE_ROWS = 60_000
FINAL_REPLAY_FRAC = 0.20
TRAIN_FRAC = 0.60
SELECTION_FRAC = 0.10
GATE_FRAC = 0.10
REPLAY_WINDOWS = 4
MIN_GATE = 1_000
MIN_WINDOW = 500
PURGE = {"5m": 5, "10m": 10}


def factories():
    return {
        "logreg": lambda: Pipeline([
            ("scale", StandardScaler()),
            ("model", LogisticRegression(C=0.25, max_iter=3000)),
        ]),
        "extra_trees": lambda: ExtraTreesClassifier(
            n_estimators=500, max_depth=12, min_samples_leaf=10,
            max_features="sqrt", random_state=42, n_jobs=-1,
        ),
        "random_forest": lambda: RandomForestClassifier(
            n_estimators=500, max_depth=10, min_samples_leaf=10,
            max_features="sqrt", random_state=42, n_jobs=-1,
        ),
        "hgb": lambda: HistGradientBoostingClassifier(
            max_iter=220, max_leaf_nodes=15, learning_rate=0.04,
            l2_regularization=1.5, random_state=42,
        ),
    }


def _probabilities(model, X):
    return aligned(model, np.asarray(X, dtype=float))


def _fit_temperature(model, X, y, horizon):
    split = int(len(y) * 0.80)
    gap = PURGE[horizon]
    cal_train_end = max(0, split - gap)
    if cal_train_end < 300 or len(y) - split < 100:
        return 1.0
    # Purge the final horizon rows of the calibration-training slice.
    model.fit(X[:cal_train_end], y[:cal_train_end])
    p = _probabilities(model, X[split:])
    return float(_temperature(p, y[split:].tolist()))


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
        except (IndexError, ValueError, TypeError, FloatingPointError):
            continue
    if len(rows) < 12_000:
        raise RuntimeError(f"insufficient archive dataset for frozen replay: {len(rows)}")
    return rows


def _load_champion(horizon):
    path = MODEL_DIR / f"{horizon}.joblib"
    if not path.is_file():
        raise RuntimeError(f"{horizon}: production Champion artifact missing")
    return joblib.load(path)


def _candidate_gate(dataset, horizon):
    n = len(dataset)
    train_end = int(n * TRAIN_FRAC)
    selection_end = int(n * (TRAIN_FRAC + SELECTION_FRAC))
    gate_end = int(n * (TRAIN_FRAC + SELECTION_FRAC + GATE_FRAC))
    train = dataset[:train_end]
    selection = dataset[train_end:selection_end]
    gate = dataset[selection_end:gate_end]
    if min(len(train), len(selection), len(gate)) < MIN_GATE:
        raise RuntimeError(f"{horizon}: development slices too small")

    X_train = np.asarray([r["x"] for r in train], dtype=float)
    y_train = np.asarray([r["y"][horizon] for r in train])
    X_sel = np.asarray([r["x"] for r in selection], dtype=float)
    y_sel = np.asarray([r["y"][horizon] for r in selection])
    X_gate = np.asarray([r["x"] for r in gate], dtype=float)
    y_gate = np.asarray([r["y"][horizon] for r in gate])

    champion = _load_champion(horizon)
    champion_gate = _probabilities(champion, X_gate)
    champion_gate_metrics = metrics(y_gate.tolist(), champion_gate)

    selection_rows = []
    for name, factory in factories().items():
        calibration_model = factory()
        temperature = _fit_temperature(calibration_model, X_train, y_train, horizon)

        # Selection is strictly out-of-sample: selection labels are not passed
        # to fit before the selection prediction is produced.
        selection_model = factory()
        selection_model.fit(X_train, y_train)
        sel_p = apply_temperature(_probabilities(selection_model, X_sel), temperature)

        # Gate is a later independent validation slice.
        gate_model = factory()
        gate_model.fit(
            np.concatenate([X_train, X_sel]),
            np.concatenate([y_train, y_sel]),
        )
        gate_p = apply_temperature(_probabilities(gate_model, X_gate), temperature)

        sm = metrics(y_sel.tolist(), sel_p)
        gm = metrics(y_gate.tolist(), gate_p)
        selection_rows.append({
            "name": name,
            "temperature": temperature,
            "selection": sm,
            "gate": gm,
            "gate_delta": {
                "accuracy": gm["accuracy"] - champion_gate_metrics["accuracy"],
                "logloss": gm["logloss"] - champion_gate_metrics["logloss"],
                "brier": gm["brier"] - champion_gate_metrics["brier"],
            },
        })

    valid = [
        x for x in selection_rows
        if len(y_gate) >= MIN_GATE
        and x["gate"]["logloss"] <= champion_gate_metrics["logloss"] - 0.002
        and x["gate"]["brier"] <= champion_gate_metrics["brier"] - 0.001
        and x["gate"]["accuracy"] >= champion_gate_metrics["accuracy"] - 0.01
    ]
    if not valid:
        selected = min(
            selection_rows,
            key=lambda x: (x["gate"]["logloss"], x["gate"]["brier"], -x["gate"]["accuracy"]),
        )
        eligible = False
    else:
        selected = min(
            valid,
            key=lambda x: (x["gate"]["logloss"], x["gate"]["brier"], -x["gate"]["accuracy"]),
        )
        eligible = True

    return {
        "selected": selected,
        "candidates": selection_rows,
        "eligible": eligible,
        "train_end": train_end,
        "selection_end": selection_end,
        "gate_end": gate_end,
        "champion_gate_metrics": champion_gate_metrics,
    }

def evaluate_horizon(dataset, horizon):
    n = len(dataset)
    prep = _candidate_gate(dataset, horizon)
    selected = prep["selected"]
    temperature = float(selected["temperature"])
    gate_end = prep["gate_end"]
    replay = dataset[gate_end:]

    # Refit only on observations strictly before replay.
    model = factories()[selected["name"]]()
    development = dataset[:gate_end]
    model.fit(
        np.asarray([r["x"] for r in development], dtype=float),
        np.asarray([r["y"][horizon] for r in development]),
    )
    if len(replay) < REPLAY_WINDOWS * MIN_WINDOW:
        return {
            "status": "DEFERRED",
            "reason": "insufficient_replay_rows",
            "n": n,
        }

    champion = _load_champion(horizon)
    window_size = len(replay) // REPLAY_WINDOWS
    windows = []
    ll_delta = []
    br_delta = []
    ac_delta = []

    for i in range(REPLAY_WINDOWS):
        start = i * window_size
        end = len(replay) if i == REPLAY_WINDOWS - 1 else (i + 1) * window_size
        block = replay[start:end]
        if len(block) < MIN_WINDOW:
            continue
        X = np.asarray([r["x"] for r in block], dtype=float)
        y = [r["y"][horizon] for r in block]
        candidate_p = apply_temperature(_probabilities(model, X), temperature)
        champion_p = _probabilities(champion, X)
        cm = metrics(y, candidate_p)
        pm = metrics(y, champion_p)
        d = {
            "accuracy": cm["accuracy"] - pm["accuracy"],
            "logloss": cm["logloss"] - pm["logloss"],
            "brier": cm["brier"] - pm["brier"],
        }
        windows.append({
            "n": len(block),
            "start_ts": int(block[0]["ts"]),
            "end_ts": int(block[-1]["ts"]),
            "candidate": cm,
            "champion": pm,
            "delta": d,
        })
        ll_delta.append(d["logloss"])
        br_delta.append(d["brier"])
        ac_delta.append(d["accuracy"])

    if len(windows) < 3:
        return {"status": "DEFERRED", "reason": "too_few_replay_windows", "n": n}

    ll = np.asarray(ll_delta, dtype=float)
    br = np.asarray(br_delta, dtype=float)
    ac = np.asarray(ac_delta, dtype=float)
    summary = {
        "replay_windows": len(windows),
        "replay_samples": int(sum(w["n"] for w in windows)),
        "mean_accuracy_delta": float(ac.mean()),
        "mean_logloss_delta": float(ll.mean()),
        "mean_brier_delta": float(br.mean()),
        "improved_logloss_ratio": float(np.mean(ll < 0)),
        "improved_brier_ratio": float(np.mean(br < 0)),
        "non_worse_accuracy_ratio": float(np.mean(ac >= -0.005)),
    }
    replay_gate = bool(
        prep["eligible"]
        and summary["replay_windows"] >= 3
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
        "policy": "archive_candidate_gate_then_fixed_multi_window_future_replay",
        "n": n,
        "selected_candidate": selected["name"],
        "selected_temperature": temperature,
        "selection_protocol": "train_only_selection; train_plus_selection_gate; pre_replay_refit; replay_never_used_for_selection",
        "development": {
            "train_n": prep["train_end"],
            "selection_n": prep["selection_end"] - prep["train_end"],
            "gate_n": prep["gate_end"] - prep["selection_end"],
            "gate_eligible": prep["eligible"],
            "champion_gate_metrics": prep["champion_gate_metrics"],
            "selected_gate_metrics": selected["gate"],
        },
        "candidate_selection": {
            k: {
                "selection": x["selection"],
                "gate": x["gate"],
                "gate_delta": x["gate_delta"],
                "temperature": x["temperature"],
            }
            for k, x in ((r["name"], r) for r in prep["candidates"])
        },
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
        "replay_holdout_protected": True,
        "horizons": {h: evaluate_horizon(dataset, h) for h in ("5m", "10m")},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
