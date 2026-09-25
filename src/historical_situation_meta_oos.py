"""Historical frozen-model situation meta OOS for BTC.

This research reconstructs prediction-time situation state from closed Binance
Vision candles strictly after the frozen production model's training timestamp.
Microstructure is not synthesized: historical order-book/taker fields are
explicitly absent from this lane. Production promotion is permanently false.
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from binance_history import binance_archive_rows
from feature_schema import FEATURES
from label_policy import direction_from_return
from model_compare import CLASSES, EMBARGO_BARS
from situation import summarize_situation

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "historical_research" / "historical_situation_meta_oos.json"

MIN_TRAIN = 3000
TEST_BLOCK = 500
FINAL_HOLDOUT_FRAC = 0.20
MIN_BLOCKS = 8
MAX_ROWS = 20000
EPS = 1e-7

SITUATION_FEATURES = (
    "trend_strength",
    "volatility_expansion_ratio",
    "normalized_entropy",
    "probability_margin",
    "ret_5m",
    "ret_15m",
    "ret_30m",
    "volatility_5m",
    "volatility_10m",
    "range_position_10m",
    "range_position_30m",
    "ema_gap_5m",
    "ema_gap_10m",
)
CAT_KEYS = (
    ("trend_state", ("RANGE", "TREND_UP", "TREND_DOWN")),
    ("volatility_state", ("STABLE", "EXPANDING", "COMPRESSING")),
    ("horizon_alignment", ("AGREE", "CONFLICT")),
    ("signal_quality", ("LOW", "MEDIUM", "HIGH")),
    ("direction_5m", CLASSES),
    ("direction_10m", CLASSES),
)


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return float(default)
    return x if math.isfinite(x) else float(default)


def _parse_utc(value: Any) -> datetime | None:
    try:
        value = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    return value.astimezone(timezone.utc) if value.tzinfo else None


def _contiguous_suffix(rows: list[list[Any]]) -> list[list[Any]]:
    ordered = sorted(
        {int(row[0]): row for row in rows}.values(),
        key=lambda row: int(row[0]),
    )
    if not ordered:
        return []
    start = len(ordered) - 1
    while start > 0 and int(ordered[start][0]) - int(ordered[start - 1][0]) == 60000:
        start -= 1
    return ordered[start:]


def _feature_snapshot(rows: list[list[Any]]) -> dict[str, float]:
    if len(rows) < 31:
        raise ValueError("insufficient_history")
    c = np.asarray([float(r[4]) for r in rows], dtype=float)
    o = np.asarray([float(r[1]) for r in rows], dtype=float)
    h = np.asarray([float(r[2]) for r in rows], dtype=float)
    l = np.asarray([float(r[3]) for r in rows], dtype=float)
    v = np.asarray([float(r[5]) for r in rows], dtype=float)
    if not all(np.isfinite(a).all() for a in (c, o, h, l, v)):
        raise ValueError("nonfinite_archive_input")
    if np.any(c <= 0) or np.any(o <= 0) or np.any(h <= 0) or np.any(l <= 0) or np.any(v < 0):
        raise ValueError("archive_input_domain")
    if np.any(h < np.maximum(o, c)) or np.any(l > np.minimum(o, c)):
        raise ValueError("archive_ohlc_invalid")

    def ret(n: int) -> float:
        return float(c[-1] / c[-1 - n] - 1.0)

    r1, r3, r5, r10, r15, r30 = [ret(n) for n in (1, 3, 5, 10, 15, 30)]
    rv5 = float(np.std(np.diff(c[-6:]) / c[-6:-1]))
    rv10 = float(np.std(np.diff(c[-11:]) / c[-11:-1]))
    hi10, lo10 = float(np.max(h[-10:])), float(np.min(l[-10:]))
    hi30, lo30 = float(np.max(h[-30:])), float(np.min(l[-30:]))

    p = float(c[-1])
    trend_alignment = 0.50 * r5 + 0.30 * r15 + 0.20 * r30
    return {
        "ret_1m": r1,
        "ret_3m": r3,
        "ret_5m": r5,
        "ret_10m": r10,
        "ret_15m": r15,
        "ret_30m": r30,
        "acceleration": r1 - r3 / 3.0,
        "volatility_5m": rv5,
        "volatility_10m": rv10,
        "range_position_10m": (p - lo10) / (hi10 - lo10) if hi10 > lo10 else 0.5,
        "range_position_30m": (p - lo30) / (hi30 - lo30) if hi30 > lo30 else 0.5,
        "body_1m": (p - float(o[-1])) / p,
        "upper_wick_1m": (float(h[-1]) - max(float(o[-1]), p)) / p,
        "lower_wick_1m": (min(float(o[-1]), p) - float(l[-1])) / p,
        "volume_ratio": float(np.mean(v[-5:])) / max(1e-12, float(np.mean(v[-15:-5]))),
        "volume_trend": float(np.mean(v[-5:])) / max(1e-12, float(np.mean(v[-10:]))),
        "ema_gap_5m": p / _ema(c[-20:], 5) - 1.0,
        "ema_gap_10m": p / _ema(c[-30:], 10) - 1.0,
        "trend_alignment": trend_alignment,
    }


def _ema(values: np.ndarray, span: int) -> float:
    alpha = 2.0 / (span + 1.0)
    out = float(values[0])
    for value in values[1:]:
        out = alpha * float(value) + (1.0 - alpha) * out
    return out


def _model_probs(model: object, x: list[float]) -> dict[str, float]:
    raw = np.asarray(model.predict_proba(np.asarray([x], dtype=float))[0], dtype=float)
    out = np.full(3, EPS, dtype=float)
    for cls, prob in zip(getattr(model, "classes_", []), raw):
        if str(cls) in CLASSES:
            out[CLASSES.index(str(cls))] = float(prob)
    out = np.clip(out, EPS, 1.0)
    out /= out.sum()
    return {cls: float(out[i]) for i, cls in enumerate(CLASSES)}


def _apply_temperature(probs: dict[str, float], temperature: float) -> dict[str, float]:
    if not (0.5 <= temperature <= 3.0) or temperature == 1.0:
        return probs
    p = np.asarray([probs[c] for c in CLASSES], dtype=float)
    z = np.log(np.clip(p, EPS, 1.0)) / temperature
    z -= z.max()
    q = np.exp(z)
    q /= q.sum()
    return {cls: float(q[i]) for i, cls in enumerate(CLASSES)}


def _load_frozen(horizon: str) -> tuple[object, datetime, float]:
    model_meta = json.loads((ROOT / "models" / f"{horizon}.json").read_text(encoding="utf-8"))
    model = joblib.load(ROOT / "models" / f"{horizon}.joblib")
    trained = _parse_utc(model_meta.get("trained_at_utc"))
    if trained is None:
        raise ValueError(f"{horizon}:invalid_trained_at")
    cal_path = ROOT / "models" / f"{horizon}.calibration.json"
    temperature = 1.0
    if cal_path.is_file():
        cal = json.loads(cal_path.read_text(encoding="utf-8"))
        if (
            cal.get("model_version") == model_meta.get("model_version")
            and int(cal.get("n_settled", 0)) >= 300
        ):
            temperature = _finite(cal.get("temperature"), 1.0)
    return model, trained, temperature


def _build_rows(max_rows: int) -> list[dict[str, Any]]:
    model5, trained5, temp5 = _load_frozen("5m")
    model10, trained10, temp10 = _load_frozen("10m")
    common_cutoff = max(trained5, trained10)

    raw = _contiguous_suffix(binance_archive_rows(max_rows + 40))
    if len(raw) < MIN_TRAIN + 100:
        return []

    out: list[dict[str, Any]] = []
    for i in range(30, len(raw) - 10):
        created = datetime.fromtimestamp(int(raw[i][0]) / 1000.0, timezone.utc)
        if created <= common_cutoff:
            continue

        snapshot = _feature_snapshot(raw[: i + 1])
        base = [float(snapshot[k]) for k in FEATURES]
        if not np.isfinite(np.asarray(base, dtype=float)).all():
            continue

        p5 = _apply_temperature(_model_probs(model5, base), temp5)
        p10 = _apply_temperature(_model_probs(model10, base), temp10)
        situation = summarize_situation(snapshot, {}, p5, p10, data_quality=None)
        target5 = datetime.fromtimestamp(int(raw[i + 5][0]) / 1000.0, timezone.utc)
        target10 = datetime.fromtimestamp(int(raw[i + 10][0]) / 1000.0, timezone.utc)
        if (target5 - created).total_seconds() != 300 or (target10 - created).total_seconds() != 600:
            continue

        future_return = float(raw[i + 5][4]) / float(raw[i][4]) - 1.0
        future_return10 = float(raw[i + 10][4]) / float(raw[i][4]) - 1.0
        out.append(
            {
                "created": created.isoformat(),
                "target": target5.isoformat(),
                "target10": target10.isoformat(),
                "y": direction_from_return(future_return),
                "y10": direction_from_return(future_return10),
                "p5": p5,
                "p10": p10,
                "situation": situation,
                "snapshot": snapshot,
            }
        )
    return out[-max_rows:]


def _vector(row: dict[str, Any]) -> np.ndarray:
    s = row["situation"]
    values = [_finite(row["p5"].get(c), 1.0 / 3.0) for c in CLASSES]
    values.extend(_finite(row["p10"].get(c), 1.0 / 3.0) for c in CLASSES)
    values.extend(_finite(s.get(key)) for key in SITUATION_FEATURES)
    for key, allowed in CAT_KEYS:
        actual = str(s.get(key, "UNKNOWN"))
        values.extend(1.0 if actual == value else 0.0 for value in allowed)
    # Historical archive has no trustworthy order-book/taker snapshot.
    # Missingness is explicit rather than filled with invented market values.
    values.append(1.0)  # microstructure_missing
    x = np.asarray(values, dtype=float)
    if not np.isfinite(x).all():
        raise ValueError("situation_vector_nonfinite")
    return x


def _aligned(model: object, rows: list[dict[str, Any]]) -> np.ndarray:
    raw = np.asarray(model.predict_proba(np.stack([_vector(r) for r in rows])), dtype=float)
    out = np.full((len(rows), 3), EPS, dtype=float)
    for index, cls in enumerate(getattr(model, "classes_", [])):
        if str(cls) in CLASSES:
            out[:, CLASSES.index(str(cls))] = raw[:, index]
    out /= out.sum(axis=1, keepdims=True)
    return out


def _metrics(rows: list[dict[str, Any]], probs: np.ndarray) -> dict[str, float]:
    labels = np.asarray([CLASSES.index(r["y"]) for r in rows], dtype=int)
    p = np.asarray(probs, dtype=float)
    picked = np.clip(p[np.arange(len(labels)), labels], EPS, 1.0)
    return {
        "n": int(len(rows)),
        "accuracy": float(np.mean(np.argmax(p, axis=1) == labels)),
        "logloss": float(-np.mean(np.log(picked))),
        "brier": float(np.mean(np.sum((p - np.eye(3)[labels]) ** 2, axis=1))),
    }


def _factories():
    return {
        "logreg": lambda: Pipeline([
            ("scale", StandardScaler()),
            ("model", LogisticRegression(C=0.15, max_iter=3000)),
        ]),
        "extra_trees": lambda: ExtraTreesClassifier(
            n_estimators=280,
            max_depth=8,
            min_samples_leaf=12,
            max_features="sqrt",
            random_state=42,
            n_jobs=-1,
        ),
        "hgb": lambda: HistGradientBoostingClassifier(
            max_iter=220,
            max_leaf_nodes=15,
            learning_rate=0.04,
            l2_regularization=1.0,
            random_state=42,
        ),
    }


def _causal_train(rows: list[dict[str, Any]], test_start: str, horizon: str) -> list[dict[str, Any]]:
    start = _parse_utc(test_start)
    if start is None:
        return []
    cutoff = start - timedelta(minutes=int(EMBARGO_BARS[horizon]))
    return [
        row for row in rows
        if (created := _parse_utc(row["created"])) is not None
        and (target := _parse_utc(row["target"])) is not None
        and created < target
        and target < cutoff
    ]


def _fit_predict(train: list[dict[str, Any]], test: list[dict[str, Any]]) -> np.ndarray:
    if len(train) < MIN_TRAIN or len(test) < 50:
        raise ValueError("insufficient_fit_rows")
    predictions = []
    for factory in _factories().values():
        model = factory()
        model.fit(
            np.stack([_vector(r) for r in train]),
            np.asarray([r["y"] for r in train]),
        )
        predictions.append(_aligned(model, test))
    out = np.mean(np.stack(predictions, axis=0), axis=0)
    out = np.clip(out, EPS, 1.0)
    return out / out.sum(axis=1, keepdims=True)


def evaluate_horizon(rows: list[dict[str, Any]], horizon: str) -> dict[str, Any]:
    # 5m uses its own outcome; 10m uses the dedicated 10m outcome.
    prepared = []
    for row in rows:
        if horizon == "10m":
            item = dict(row)
            item["y"] = direction_from_return(
                math.nan
            ) if False else row["y10"]
        else:
            item = row
        prepared.append(item)
    return _evaluate_rows(prepared, horizon)


def _evaluate_rows(rows: list[dict[str, Any]], horizon: str) -> dict[str, Any]:
    if len(rows) < MIN_TRAIN + TEST_BLOCK * MIN_BLOCKS + 100:
        return {
            "status": "DEFERRED",
            "n": len(rows),
            "reason": "insufficient_archive_rows",
        }
    split = int(len(rows) * (1.0 - FINAL_HOLDOUT_FRAC))
    development, holdout = rows[:split], rows[split:]
    blocks = []
    for end in range(MIN_TRAIN, len(development), TEST_BLOCK):
        test = development[end:min(end + TEST_BLOCK, len(development))]
        train = _causal_train(development[:end], test[0]["created"], horizon)
        if len(train) < MIN_TRAIN or len(test) < 50:
            continue
        try:
            candidate = _fit_predict(train, test)
        except Exception:
            continue
        baseline = np.stack([
            [r["p5"][c] if horizon == "5m" else r["p10"][c] for c in CLASSES]
            for r in test
        ])
        cm = _metrics(test, candidate)
        bm = _metrics(test, baseline)
        blocks.append(
            {
                "n": len(test),
                "candidate": cm,
                "baseline": bm,
                "delta": {
                    "accuracy": cm["accuracy"] - bm["accuracy"],
                    "logloss": cm["logloss"] - bm["logloss"],
                    "brier": cm["brier"] - bm["brier"],
                },
            }
        )
    if len(blocks) < MIN_BLOCKS:
        return {"status": "DEFERRED", "n": len(rows), "reason": "insufficient_valid_oos_blocks"}

    hold_train = _causal_train(development, holdout[0]["created"], horizon)
    hold_candidate = _fit_predict(hold_train, holdout)
    hold_baseline = np.stack([
        [r["p5"][c] if horizon == "5m" else r["p10"][c] for c in CLASSES]
        for r in holdout
    ])
    ll = np.asarray([b["delta"]["logloss"] for b in blocks], dtype=float)
    br = np.asarray([b["delta"]["brier"] for b in blocks], dtype=float)
    ac = np.asarray([b["delta"]["accuracy"] for b in blocks], dtype=float)
    candidate = _metrics(holdout, hold_candidate)
    baseline = _metrics(holdout, hold_baseline)

    return {
        "status": "OK",
        "research_only": True,
        "production_changed": False,
        "promotion_evidence_eligible": False,
        "final_holdout_protected": True,
        "final_holdout_used_for_selection": False,
        "n": len(rows),
        "development_n": len(development),
        "final_holdout_n": len(holdout),
        "summary": {
            "blocks": len(blocks),
            "samples": int(sum(b["n"] for b in blocks)),
            "mean_accuracy_delta": float(ac.mean()),
            "mean_logloss_delta": float(ll.mean()),
            "mean_brier_delta": float(br.mean()),
            "improved_logloss_ratio": float(np.mean(ll < 0)),
            "improved_brier_ratio": float(np.mean(br < 0)),
            "non_worse_accuracy_ratio": float(np.mean(ac >= -0.005)),
        },
        "final_holdout": {
            "candidate": candidate,
            "baseline": baseline,
            "delta": {
                "accuracy": candidate["accuracy"] - baseline["accuracy"],
                "logloss": candidate["logloss"] - baseline["logloss"],
                "brier": candidate["brier"] - baseline["brier"],
            },
        },
        "eligible_pending_live_primary_confirmation": bool(
            float(np.mean(ll < 0)) >= 0.60
            and float(np.mean(br < 0)) >= 0.60
            and float(np.mean(ac >= -0.005)) >= 0.80
            and float(ll.mean()) <= -0.003
            and float(br.mean()) <= -0.0015
        ),
        "blocks": blocks,
    }


def main() -> None:
    rows = _build_rows(MAX_ROWS)
    payload = {
        "schema_version": 1,
        "research_only": True,
        "production_changed": False,
        "production_promotion_allowed": False,
        "archive_pit_policy": "frozen_model_generation_cutoff; closed_candle_only; explicit_microstructure_missingness",
        "rows": len(rows),
        "horizons": {
            "5m": evaluate_horizon(rows, "5m"),
            "10m": evaluate_horizon(rows, "10m"),
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
