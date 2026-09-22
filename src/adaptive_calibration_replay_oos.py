"""Research-only prequential/adaptive calibration replay.

The frozen production Champion is not changed. For each future replay window,
calibration parameters are fitted only from observations strictly before that
window, with a horizon-specific purge gap. The current window is never used to
fit its own calibrator.
"""
from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
import sys

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for p in (ROOT, SRC):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from binance_history import binance_archive_rows
from bootstrap_train import make_features
from label_policy import direction_from_return
from model_compare import aligned, apply_temperature, metrics, _temperature

OUT = ROOT / "data" / "historical_research" / "adaptive_calibration_replay_oos.json"
ARCHIVE_ROWS = 60_000
MIN_ROWS = 12_000
WINDOWS = 4
MIN_WINDOW = 500
MIN_CALIBRATION = 800
PURGE = {"5m": 5, "10m": 10}


def _post_champion_cutoff_ms():
    cutoffs = []
    for horizon in ("5m", "10m"):
        try:
            meta = json.loads(
                (ROOT / "models" / f"{horizon}.json").read_text(encoding="utf-8")
            )
            trained = datetime.fromisoformat(
                str(meta["trained_at_utc"]).replace("Z", "+00:00")
            )
            cutoffs.append(int(trained.timestamp() * 1000))
        except (OSError, TypeError, KeyError, ValueError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"invalid Champion training metadata for {horizon}") from exc
    return max(cutoffs)


def _dataset():
    raw = [r for r in binance_archive_rows(ARCHIVE_ROWS)
           if int(r[0]) > _post_champion_cutoff_ms()]
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
        raise RuntimeError(f"insufficient post-Champion archive rows: {len(rows)}")
    return rows


def _quality(y, p):
    m = metrics(y, p)
    pred = np.argmax(np.asarray(p), axis=1)
    conf = np.max(np.asarray(p), axis=1)
    idx = {"DOWN": 0, "FLAT": 1, "UP": 2}
    yi = np.asarray([idx[v] for v in y], dtype=int)
    hit = (pred == yi).astype(float)
    ece = 0.0
    for lo in np.linspace(0.0, 0.9, 10):
        hi = lo + 0.1
        mask = (conf >= lo) & ((conf < hi) | ((hi >= 1.0) & (conf <= hi)))
        if mask.any():
            ece += float(mask.mean()) * abs(float(hit[mask].mean()) - float(conf[mask].mean()))
    return {**m, "ece": float(ece)}


def _fit_temperature(p, y):
    if len(y) < MIN_CALIBRATION or len(set(y)) < 3:
        return 1.0
    return float(_temperature(np.asarray(p, dtype=float), list(y)))


def _champion(horizon):
    path = ROOT / "models" / f"{horizon}.joblib"
    if not path.is_file():
        raise RuntimeError(f"{horizon}: Champion artifact missing")
    return joblib.load(path)


def evaluate(horizon, rows):
    n = len(rows)
    champion = _champion(horizon)
    X = np.asarray([r["x"] for r in rows], dtype=float)
    y = [r["y"][horizon] for r in rows]
    champion_p = aligned(champion, X)

    replay_start = int(n * 0.70)
    replay = rows[replay_start:]
    if len(replay) < WINDOWS * MIN_WINDOW:
        return {"status": "DEFERRED", "reason": "insufficient_replay_rows", "n": n}

    # Initial fixed calibrator uses only pre-replay development data.
    fixed_train_end = max(1, replay_start - PURGE[horizon])
    fixed_temp = _fit_temperature(champion_p[:fixed_train_end], y[:fixed_train_end])

    size = len(replay) // WINDOWS
    windows = []
    adaptive_temps = []

    for j in range(WINDOWS):
        start = j * size
        end = len(replay) if j == WINDOWS - 1 else (j + 1) * size
        block_y = y[replay_start + start:replay_start + end]
        if len(block_y) < MIN_WINDOW:
            continue

        # Delayed feedback: all fitting observations end before the current
        # replay window, leaving a horizon-specific purge gap.
        fit_end = replay_start + start - PURGE[horizon]
        history_p = champion_p[:max(0, fit_end)]
        history_y = y[:max(0, fit_end)]
        adaptive_temp = _fit_temperature(history_p, history_y)
        adaptive_temps.append(adaptive_temp)

        block_p = champion_p[replay_start + start:replay_start + end]
        raw_q = _quality(block_y, block_p)
        fixed_q = _quality(block_y, apply_temperature(block_p, fixed_temp))
        adaptive_q = _quality(block_y, apply_temperature(block_p, adaptive_temp))

        windows.append({
            "n": len(block_y),
            "start_ts": int(replay[start]["ts"]),
            "end_ts": int(replay[end - 1]["ts"]),
            "fit_end_before_purge_ts": int(rows[fit_end - 1]["ts"]) if fit_end > 0 else None,
            "fixed_temperature": float(fixed_temp),
            "adaptive_temperature": float(adaptive_temp),
            "raw": raw_q,
            "fixed": fixed_q,
            "adaptive": adaptive_q,
            "adaptive_delta_vs_raw": {
                "accuracy": adaptive_q["accuracy"] - raw_q["accuracy"],
                "logloss": adaptive_q["logloss"] - raw_q["logloss"],
                "brier": adaptive_q["brier"] - raw_q["brier"],
                "ece": adaptive_q["ece"] - raw_q["ece"],
            },
        })

    if len(windows) < 3:
        return {"status": "DEFERRED", "reason": "too_few_replay_windows", "n": n}

    raw_to_adaptive = [
        (
            w["adaptive"]["logloss"] - w["raw"]["logloss"],
            w["adaptive"]["brier"] - w["raw"]["brier"],
            w["adaptive"]["ece"] - w["raw"]["ece"],
            w["adaptive"]["accuracy"] - w["raw"]["accuracy"],
        )
        for w in windows
    ]
    arr = np.asarray(raw_to_adaptive, dtype=float)
    summary = {
        "windows": len(windows),
        "samples": int(sum(w["n"] for w in windows)),
        "mean_logloss_delta": float(arr[:, 0].mean()),
        "mean_brier_delta": float(arr[:, 1].mean()),
        "mean_ece_delta": float(arr[:, 2].mean()),
        "mean_accuracy_delta": float(arr[:, 3].mean()),
        "improved_logloss_ratio": float(np.mean(arr[:, 0] < 0)),
        "improved_brier_ratio": float(np.mean(arr[:, 1] < 0)),
        "improved_ece_ratio": float(np.mean(arr[:, 2] <= 0)),
        "non_worse_accuracy_ratio": float(np.mean(arr[:, 3] >= -0.005)),
        "adaptive_temperature_min": float(min(adaptive_temps)) if adaptive_temps else None,
        "adaptive_temperature_max": float(max(adaptive_temps)) if adaptive_temps else None,
    }
    replay_gate = bool(
        summary["improved_logloss_ratio"] >= 0.75
        and summary["improved_brier_ratio"] >= 0.75
        and summary["improved_ece_ratio"] >= 0.75
        and summary["non_worse_accuracy_ratio"] >= 0.75
        and summary["mean_logloss_delta"] <= -0.002
        and summary["mean_brier_delta"] <= -0.001
    )

    return {
        "status": "OK",
        "schema_version": 1,
        "research_only": True,
        "production_changed": False,
        "promotion_evidence_eligible": False,
        "replay_holdout_protected": True,
        "replay_holdout_used_for_selection": False,
        "policy": "post_champion_archive_prequential_calibration_with_horizon_purge",
        "fixed_temperature": float(fixed_temp),
        "summary": summary,
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
        "policy": "adaptive_prequential_calibration_research_only",
        "post_champion_cutoff_ms": _post_champion_cutoff_ms(),
        "horizons": {h: evaluate(h, rows) for h in ("5m", "10m")},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
