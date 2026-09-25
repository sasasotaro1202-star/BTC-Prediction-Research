"""Robust BTC-only historical bootstrap trainer."""
from __future__ import annotations

import json
import math
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from lightgbm import LGBMClassifier
except ImportError:
    LGBMClassifier = None

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None

try:
    from db import DB, init_db
    from binance_history import binance_archive_rows
    from feature_schema import FEATURES
except ModuleNotFoundError:
    from src.db import DB, init_db
    from src.binance_history import binance_archive_rows
    from src.feature_schema import FEATURES

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
DATA_DIR = ROOT / "data" / "historical_research"
CACHE = DATA_DIR / "btc_bootstrap_1m.json"
STATUS = DATA_DIR / "bootstrap_status.json"
from label_policy import CLASSES, NEUTRAL_RETURN
THRESHOLD = NEUTRAL_RETURN
MIN_BOOTSTRAP_ROWS = 10_000
TARGET_ROWS = 30_000
MIN_TRAIN = 1_000
MIN_OOS = 500
UA = "BTC-Prediction-Research/bootstrap/5.2"

def write_status(payload: dict) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    STATUS.write_text(json.dumps({**payload, "updated_at_utc": datetime.now(timezone.utc).isoformat()}, indent=2), encoding="utf-8")

def _get(url: str, attempts: int = 4, timeout: int = 30):
    last = None
    for i in range(attempts):
        try:
            from urllib.request import Request, urlopen
            req = Request(url, headers={"User-Agent": UA, "Accept": "application/json"})
            with urlopen(req, timeout=timeout) as r:
                return json.loads(r.read())
        except Exception as exc:
            last = exc
            if i + 1 < attempts:
                time.sleep(min(4.0, 0.75 * (i + 1)))
    raise last

def _bybit_page(end_ms=None, limit=1000):
    params = {"category": "linear", "symbol": "BTCUSDT", "interval": "1", "limit": min(limit, 1000)}
    if end_ms is not None: params["end"] = int(end_ms)
    return _get("https://api.bybit.com/v5/market/kline?" + urlencode(params))

def _binance_page(end_ms=None, limit=1500):
    params = {"symbol": "BTCUSDT", "interval": "1m", "limit": limit}
    if end_ms is not None: params["endTime"] = int(end_ms)
    return _get("https://fapi.binance.com/fapi/v1/klines?" + urlencode(params))

def fetch_bybit(target: int):
    rows = []
    end = None
    for _ in range(math.ceil(target / 1000) + 8):
        raw = _bybit_page(end, 1000).get("result", {}).get("list", [])
        if not raw: break
        rows.extend([[int(r[0]), float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[5])] for r in raw if len(r) >= 6])
        new_end = min(int(r[0]) for r in raw) - 1
        if end is not None and new_end >= end: break
        end = new_end
        if len(rows) >= target: break
        time.sleep(0.05)
    now_ms = int(time.time() * 1000)
    rows = [r for r in rows if r[0] + 60_000 <= now_ms]
    return sorted({r[0]: r for r in rows}.values(), key=lambda r: r[0])[-target:]

def fetch_binance(target: int):
    rows = []
    end = None
    for _ in range(math.ceil(target / 1500) + 8):
        page = _binance_page(end, 1500)
        if not page: break
        rows.extend([[int(r[0]), float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[5])] for r in page])
        end = min(int(r[0]) for r in page) - 1
        if len(page) < 1500: break
    now_ms = int(time.time() * 1000)
    rows = [r for r in rows if r[0] + 60_000 <= now_ms]
    return sorted({r[0]: r for r in rows}.values(), key=lambda r: r[0])[-target:]

def _contiguous_suffix(rows):
    if not rows: return []
    ordered = sorted({int(r[0]): r for r in rows}.values(), key=lambda r: r[0])
    start = len(ordered) - 1
    while start > 0 and ordered[start][0] - ordered[start - 1][0] == 60_000: start -= 1
    return ordered[start:]

def fetch_history(target: int = TARGET_ROWS):
    errors = []
    best = []
    best_name = "none"
    sources = (("binance_vision_archive", lambda: binance_archive_rows(target)), ("bybit_futures", lambda: fetch_bybit(target)), ("binance_futures", lambda: fetch_binance(target)))
    for name, loader in sources:
        try:
            rows = _contiguous_suffix(loader())
            if len(rows) > len(best): best, best_name = rows, name
            if len(rows) >= target: return rows, name
            errors.append(f"{name}:only_{len(rows)}_contiguous_rows")
        except Exception as exc: errors.append(f"{name}:{type(exc).__name__}:{exc}")
    if best: return best, best_name
    raise RuntimeError("no free BTC historical source available: " + "; ".join(errors))

def ema(values, span):
    a = 2 / (span + 1); e = float(values[0])
    for x in values[1:]: e = a * float(x) + (1 - a) * e
    return e

def make_features(rows):
    c = np.asarray([r[4] for r in rows], float); o = np.asarray([r[1] for r in rows], float); h = np.asarray([r[2] for r in rows], float); l = np.asarray([r[3] for r in rows], float); v = np.asarray([r[5] for r in rows], float); p = c[-1]
    ret = lambda n: c[-1] / c[-1 - n] - 1; r1, r3, r5, r10 = [ret(n) for n in (1, 3, 5, 10)]
    accel = r1 - r3 / 3; rv5 = float(np.std(np.diff(c[-6:]) / c[-6:-1])); rv10 = float(np.std(np.diff(c[-11:]) / c[-11:-1])); hi, lo = max(h[-10:]), min(l[-10:]); rp = (p - lo) / (hi - lo) if hi > lo else 0.5
    body = (p - o[-1]) / p; up = (h[-1] - max(o[-1], p)) / p; low = (min(o[-1], p) - l[-1]) / p; old = float(np.mean(v[-15:-5])); rvol = float(np.mean(v[-5:])) / max(1e-12, old); vtrend = float(np.mean(v[-5:])) / max(1e-12, float(np.mean(v[-10:])))
    return [r1, r3, r5, r10, accel, rv5, rv10, rp, body, up, low, rvol, vtrend, p / ema(c[-20:], 5) - 1, p / ema(c[-30:], 10) - 1]

def build_dataset(rows, horizon):
    X, y = [], []
    for i in range(30, len(rows) - horizon):
        future_return = rows[i + horizon][4] / rows[i][4] - 1
        y.append("UP" if future_return > THRESHOLD else "DOWN" if future_return < -THRESHOLD else "FLAT")
        X.append(make_features(rows[:i + 1]))
    return np.asarray(X, float), np.asarray(y)

def normalize(probs):
    p = np.clip(np.asarray(probs, float), 1e-8, 1); return p / p.sum(axis=1, keepdims=True)

def metrics(y, probs):
    idx = {c: i for i, c in enumerate(CLASSES)}; yi = np.asarray([idx[z] for z in y]); p = normalize(probs); one = np.eye(3)[yi]
    return {"accuracy": float(np.mean(p.argmax(1) == yi)), "logloss": float(log_loss(yi, p, labels=[0, 1, 2])), "brier": float(np.mean(np.sum((p - one) ** 2, axis=1)))}

def fit_temperature(probs, y):
    if len(y) < 200 or len(set(y)) < 3: return 1.0
    p = normalize(probs); yi = np.asarray([{c: i for i, c in enumerate(CLASSES)}[z] for z in y]); split = max(int(len(y) * 0.75), 100); logits = np.log(np.clip(p, 1e-8, 1)); best_t, best_loss = 1.0, float("inf")
    for t in np.linspace(0.7, 2.5, 73):
        z = logits[:split] / t; z -= z.max(axis=1, keepdims=True); q = np.exp(z); q /= q.sum(axis=1, keepdims=True); ll = log_loss(yi[:split], q, labels=[0, 1, 2])
        if ll < best_loss: best_loss, best_t = ll, float(t)
    z = logits[split:] / best_t; z -= z.max(axis=1, keepdims=True); q = np.exp(z); q /= q.sum(axis=1, keepdims=True)
    return best_t if log_loss(yi[split:], q, labels=[0, 1, 2]) < log_loss(yi[split:], normalize(p[split:]), labels=[0, 1, 2]) - 0.001 else 1.0

def development_gate_passes(gate_score, baseline_gate):
    """Require meaningful relative gains on an independent development gate.

    This function never sees the frozen holdout. It intentionally requires
    both proper-scoring improvements before a candidate can be published.
    """
    base_ll = max(abs(float(baseline_gate["logloss"])), 1e-12)
    base_br = max(abs(float(baseline_gate["brier"])), 1e-12)
    ll_rel_gain = (float(baseline_gate["logloss"]) - float(gate_score["logloss"])) / base_ll
    br_rel_gain = (float(baseline_gate["brier"]) - float(gate_score["brier"])) / base_br
    acc_delta = float(gate_score["accuracy"]) - float(baseline_gate["accuracy"])
    return bool(
        ll_rel_gain >= 0.03
        and br_rel_gain >= 0.01
        and acc_delta >= 0.0
    )


def _fit_model(model, X, y):
    model.fit(np.asarray(X, dtype=float), np.asarray(y))
    return model


def candidate_factories():
    """Return the deterministic production candidate set.

    Candidates are evaluated chronologically; adding a candidate never bypasses
    the development gate or the frozen descriptive holdout.
    """
    from ensemble_model import SoftVotingEnsemble
    return [
        (
            "logreg",
            Pipeline([
                ("scale", StandardScaler()),
                ("model", LogisticRegression(C=0.5, max_iter=3000)),
            ]),
        ),
        (
            "rf",
            RandomForestClassifier(
                n_estimators=300,
                max_depth=7,
                min_samples_leaf=12,
                max_features="sqrt",
                random_state=42,
                n_jobs=-1,
            ),
        ),
        (
            # Research-backed deeper RF variant from frozen multi-window replay.
            # It remains subject to the same chronological validation + gate and
            # does not bypass any production safety condition.
            "rf_replay",
            RandomForestClassifier(
                n_estimators=500,
                max_depth=10,
                min_samples_leaf=10,
                max_features="sqrt",
                random_state=42,
                n_jobs=-1,
            ),
        ),
        (
            "rf_balanced",
            RandomForestClassifier(
                n_estimators=500,
                max_depth=10,
                min_samples_leaf=10,
                max_features="sqrt",
                class_weight="balanced_subsample",
                random_state=42,
                n_jobs=-1,
            ),
        ),
        (
            "extra_trees",
            ExtraTreesClassifier(
                n_estimators=350,
                max_depth=10,
                min_samples_leaf=10,
                max_features="sqrt",
                random_state=42,
                n_jobs=-1,
            ),
        ),
        (
            "extra_balanced",
            ExtraTreesClassifier(
                n_estimators=500,
                max_depth=10,
                min_samples_leaf=10,
                max_features="sqrt",
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            ),
        ),
        *(
            [(
                "lightgbm",
                LGBMClassifier(
                    objective="multiclass", num_class=3, n_estimators=350,
                    learning_rate=0.03, num_leaves=31, min_child_samples=30,
                    subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
                    random_state=42, n_jobs=-1, verbosity=-1,
                ),
            )] if LGBMClassifier is not None else []
        ),
        *(
            [(
                "xgboost",
                XGBClassifier(
                    objective="multi:softprob", num_class=3, n_estimators=320,
                    max_depth=5, learning_rate=0.03, min_child_weight=12,
                    subsample=0.9, colsample_bytree=0.9, reg_lambda=2.0,
                    reg_alpha=0.05, eval_metric="mlogloss", random_state=42,
                    n_jobs=-1, verbosity=0,
                ),
            )] if XGBClassifier is not None else []
        ),
        (
            "hgb",
            HistGradientBoostingClassifier(
                max_iter=220,
                max_leaf_nodes=15,
                learning_rate=0.04,
                l2_regularization=1.5,
                random_state=42,
            ),
        ),
        ("soft_ensemble", SoftVotingEnsemble(learn_weights=True)),
    ]


def train_one(X, y, purge_gap=0):
    """Nested chronological development selection with a frozen descriptive holdout.

    Model selection and publication use only train/selection/gate regions.
    The final holdout is never used for selection, gating, thresholding, or
    artifact publication; it is evaluated once for description only.
    """
    n = len(y)
    if n < 3000:
        raise ValueError("bootstrap dataset too small for nested chronological split")

    gap = max(0, int(purge_gap))
    train_end = int(n * 0.55)
    select_end = int(n * 0.70)
    gate_end = int(n * 0.85)

    train_stop = max(0, train_end - gap)
    select_stop = max(train_end, select_end - gap)
    gate_stop = max(select_end, gate_end - gap)

    Xtr, ytr = X[:train_stop], y[:train_stop]
    Xsel, ysel = X[train_end:select_end], y[train_end:select_end]
    Xgate, ygate = X[select_end:gate_end], y[select_end:gate_end]
    Xhold, yhold = X[gate_end:], y[gate_end:]

    if min(len(ytr), len(ysel), len(ygate), len(yhold)) < 100:
        raise ValueError("nested chronological split produced insufficient rows")

    baseline_gate = metrics(
        ygate,
        np.tile(
            np.asarray([np.mean(ytr == c) for c in CLASSES], dtype=float),
            (len(ygate), 1),
        ),
    )

    candidates = candidate_factories()

    validation_results = []
    for name, model in candidates:
        _fit_model(model, Xtr, ytr)
        v = metrics(ysel, normalize(model.predict_proba(Xsel)))
        validation_results.append(
            {
                "model": name,
                "logloss": float(v["logloss"]),
                "brier": float(v["brier"]),
                "accuracy": float(v["accuracy"]),
            }
        )

    validation_results.sort(key=lambda r: (-r["accuracy"], r["logloss"], r["brier"]))
    selected_name = validation_results[0]["model"]
    factories = {name: model for name, model in candidates}
    selected_for_gate = factories[selected_name]
    _fit_model(selected_for_gate, X[:select_stop], y[:select_stop])
    gate_score = metrics(ygate, normalize(selected_for_gate.predict_proba(Xgate)))

    # Publication gate is based only on the independent development gate slice.
    safe_gain = development_gate_passes(gate_score, baseline_gate)

    # Only after the development gate passes is the final model refit on all
    # non-holdout observations. The holdout remains untouched until scoring.
    final_model = factories[selected_name]
    if safe_gain:
        _fit_model(final_model, X[:gate_stop], y[:gate_stop])
    else:
        # Still provide a descriptive holdout result without publishing.
        _fit_model(final_model, X[:gate_stop], y[:gate_stop])

    holdout_score = metrics(
        yhold,
        normalize(final_model.predict_proba(Xhold)),
    )

    return {
        "model_name": selected_name,
        "model": final_model,
        "baseline_gate": baseline_gate,
        "gate_score": gate_score,
        "holdout_score": holdout_score,
        "holdout_n": len(yhold),
        "validation_results": validation_results,
        "promotion_allowed": bool(safe_gain),
        "selection_method": "nested_chronological_selection_gate_holdout",
        "holdout_used_for_selection": False,
        "holdout_used_for_gate": False,
        "holdout_is_descriptive_only": True,
        "train_n": len(ytr),
        "selection_n": len(ysel),
        "gate_n": len(ygate),
    }


def publish(horizon, result):
    name = result["model_name"]
    model = result["model"]
    if not result["promotion_allowed"]:
        return False, {
            "status": "development_gate_rejected",
            "model": name,
            "gate_metrics": result["gate_score"],
            "baseline_gate_metrics": result["baseline_gate"],
            "holdout_metrics": result["holdout_score"],
            "holdout_n": result["holdout_n"],
            "selection": result["validation_results"],
            "selection_method": result["selection_method"],
            "holdout_used_for_selection": False,
            "holdout_used_for_gate": False,
            "holdout_is_descriptive_only": True,
        }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_DIR / f"{horizon}.joblib")
    version = f"bootstrap.{name}.v5.4"
    meta = {
        "model_version": version,
        "horizon": horizon,
        "classes": list(model.classes_),
        "features": FEATURES,
        "artifact": f"{horizon}.joblib",
        "candidate": False,
        "bootstrap": True,
        "selection_method": result["selection_method"],
        "holdout_used_for_selection": False,
        "holdout_used_for_gate": False,
        "holdout_is_descriptive_only": True,
        "train_n": result["train_n"],
        "selection_n": result["selection_n"],
        "gate_n": result["gate_n"],
        "holdout_n": result["holdout_n"],
        "gate_metrics": result["gate_score"],
        "baseline_gate_metrics": result["baseline_gate"],
        "final_holdout_metrics": result["holdout_score"],
        "validation_metrics": result["validation_results"],
        "final_fit_fraction": 0.85,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (MODEL_DIR / f"{horizon}.json").write_text(
        json.dumps(meta, indent=2),
        encoding="utf-8",
    )
    with sqlite3.connect(DB) as con:
        con.execute(
            "INSERT INTO model_registry(horizon,production_version,updated_at_utc) "
            "VALUES(?,?,?) ON CONFLICT(horizon) DO UPDATE SET "
            "production_version=excluded.production_version,updated_at_utc=excluded.updated_at_utc",
            (horizon, version, datetime.now(timezone.utc).isoformat()),
        )
    return True, meta

def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True); DATA_DIR.mkdir(parents=True, exist_ok=True); init_db()
    try:
        rows, source = fetch_history(TARGET_ROWS); write_status({"status": "history_ok", "source": source, "rows": len(rows), "oldest_utc": datetime.fromtimestamp(rows[0][0] / 1000, timezone.utc).isoformat(), "newest_utc": datetime.fromtimestamp(rows[-1][0] / 1000, timezone.utc).isoformat()})
    except Exception as exc: write_status({"status": "history_failed", "error": f"{type(exc).__name__}: {exc}"}); return 0
    if len(rows) < MIN_BOOTSTRAP_ROWS: write_status({"status": "insufficient_history", "rows": len(rows), "minimum": MIN_BOOTSTRAP_ROWS, "source": source}); return 0
    CACHE.write_text(json.dumps({"created_at_utc": datetime.now(timezone.utc).isoformat(), "source": source, "rows": rows}, separators=(",", ":")), encoding="utf-8"); published = []
    for horizon in ("5m", "10m"):
        X, y = build_dataset(rows, int(horizon[:-1]))
        if len(y) < MIN_TRAIN + MIN_OOS or len(set(y)) < 3: published.append({"horizon": horizon, "status": "insufficient_dataset", "rows": len(y)}); continue
        result = train_one(X, y, purge_gap=int(horizon[:-1]))
        ok, meta = publish(horizon, result)
        published.append({
            "horizon": horizon,
            "published": ok,
            "model": result["model_name"],
            "status": meta.get("status") if isinstance(meta, dict) else "published",
            "holdout_used_for_selection": False,
            "holdout_is_descriptive_only": True,
        })
    write_status({"status": "complete", "source": source, "rows": len(rows), "published": published}); return 0

if __name__ == "__main__":
    raise SystemExit(main())
