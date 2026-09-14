"""Robust BTC-only historical bootstrap trainer."""
from __future__ import annotations

import json
import math
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from db import DB, init_db
from binance_history import binance_archive_rows
from market_data import coinbase_rows, bybit_klines, closed_bybit

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"; DATA_DIR = ROOT / "data" / "historical_research"; CACHE = DATA_DIR / "btc_bootstrap_1m.json"; STATUS = DATA_DIR / "bootstrap_status.json"
CLASSES = ["DOWN", "FLAT", "UP"]
FEATURES = ["ret_1m", "ret_3m", "ret_5m", "ret_10m", "acceleration", "volatility_5m", "volatility_10m", "range_position_10m", "body_1m", "upper_wick_1m", "lower_wick_1m", "volume_ratio", "volume_trend", "ema_gap_5m", "ema_gap_10m"]
THRESHOLD = 0.00020; MIN_BOOTSTRAP_ROWS = 10_000; TARGET_ROWS = 30_000; MIN_TRAIN = 1_000; MIN_OOS = 500; UA = "BTC-Prediction-Research/bootstrap/5.0"

def write_status(payload: dict) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True); STATUS.write_text(json.dumps({**payload, "updated_at_utc": datetime.now(timezone.utc).isoformat()}, indent=2), encoding="utf-8")

def _get(url: str, attempts: int = 4, timeout: int = 30):
    last = None
    for i in range(attempts):
        try:
            req = Request(url, headers={"User-Agent": UA, "Accept": "application/json"})
            with urlopen(req, timeout=timeout) as r: return json.loads(r.read())
        except Exception as exc:
            last = exc
            if i + 1 < attempts: time.sleep(min(4.0, 0.75 * (i + 1)))
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
    rows = []; end = None
    for _ in range(math.ceil(target / 1000) + 8):
        raw = _bybit_page(end, 1000).get("result", {}).get("list", [])
        if not raw: break
        rows.extend([[int(r[0]), float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[5])] for r in raw if len(r) >= 6])
        oldest = min(int(r[0]) for r in raw); new_end = oldest - 1
        if end is not None and new_end >= end: break
        end = new_end
        if len(rows) >= target: break
        time.sleep(0.05)
    now_ms = int(time.time() * 1000); rows = [r for r in rows if r[0] + 60_000 <= now_ms]
    return sorted({r[0]: r for r in rows}.values(), key=lambda r: r[0])[-target:]

def fetch_binance(target: int):
    rows = []; end = None
    for _ in range(math.ceil(target / 1500) + 8):
        page = _binance_page(end, 1500)
        if not page: break
        rows.extend([[int(r[0]), float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[5])] for r in page])
        end = min(int(r[0]) for r in page) - 1
        if len(page) < 1500: break
    now_ms = int(time.time() * 1000); rows = [r for r in rows if r[0] + 60_000 <= now_ms]
    return sorted({r[0]: r for r in rows}.values(), key=lambda r: r[0])[-target:]

def _contiguous_suffix(rows):
    """Keep only the latest fully contiguous 1-minute suffix; never bridge gaps."""
    if not rows: return []
    ordered = sorted({int(r[0]): r for r in rows}.values(), key=lambda r: r[0]); start = len(ordered) - 1
    while start > 0 and ordered[start][0] - ordered[start - 1][0] == 60_000: start -= 1
    return ordered[start:]

def fetch_history(target: int = TARGET_ROWS):
    errors = []; best = []; best_name = "none"
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
        future_return = rows[i + horizon][4] / rows[i][4] - 1; y.append("UP" if future_return > THRESHOLD else "DOWN" if future_return < -THRESHOLD else "FLAT"); X.append(make_features(rows[:i + 1]))
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

def train_one(X, y):
    n = len(y); train_end = int(n * 0.65); cal_end = int(n * 0.82); Xtr, Xcal, Xte = X[:train_end], X[train_end:cal_end], X[cal_end:]; ytr, ycal, yte = y[:train_end], y[train_end:cal_end], y[cal_end:]
    baseline = metrics(yte, np.tile(np.asarray([np.mean(ytr == c) for c in CLASSES]), (len(yte), 1))); candidates = [("logreg", Pipeline([("scale", StandardScaler()), ("model", LogisticRegression(C=0.5, max_iter=3000))])), ("rf", RandomForestClassifier(n_estimators=300, max_depth=7, min_samples_leaf=12, max_features="sqrt", random_state=42, n_jobs=-1)), ("hgb", HistGradientBoostingClassifier(max_iter=220, max_leaf_nodes=15, learning_rate=0.04, l2_regularization=1.5, random_state=42))]; results = []
    for name, model in candidates:
        model.fit(Xtr, ytr); temperature = fit_temperature(model.predict_proba(Xcal), ycal); p = normalize(model.predict_proba(Xte))
        if temperature != 1.0: z = np.log(p) / temperature; z -= z.max(axis=1, keepdims=True); p = np.exp(z); p /= p.sum(axis=1, keepdims=True)
        score = metrics(yte, p); results.append((score["logloss"], score["brier"], -score["accuracy"], name, model, temperature, score))
    results.sort(key=lambda r: r[:3]); return results[0], baseline, len(yte)

def publish(horizon, best, baseline, holdout_n):
    _, _, _, name, model, temperature, score = best; safe_gain = score["logloss"] < baseline["logloss"] - 0.01 and score["brier"] < baseline["brier"] - 0.005
    if not safe_gain: return False, {"status": "holdout_rejected", "model": name, "candidate": score, "baseline": baseline, "holdout_n": holdout_n}
    MODEL_DIR.mkdir(parents=True, exist_ok=True); joblib.dump(model, MODEL_DIR / f"{horizon}.joblib"); version = f"bootstrap.{name}.v5"; meta = {"model_version": version, "horizon": horizon, "classes": list(model.classes_), "features": FEATURES, "artifact": f"{horizon}.joblib", "candidate": False, "bootstrap": True, "holdout_n": holdout_n, "holdout_metrics": score, "baseline_metrics": baseline, "temperature": float(temperature), "trained_at_utc": datetime.now(timezone.utc).isoformat()}; (MODEL_DIR / f"{horizon}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    with sqlite3.connect(DB) as con: con.execute("INSERT INTO model_registry(horizon,production_version,updated_at_utc) VALUES(?,?,?) ON CONFLICT(horizon) DO UPDATE SET production_version=excluded.production_version,updated_at_utc=excluded.updated_at_utc", (horizon, version, datetime.now(timezone.utc).isoformat()))
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
        best, baseline, holdout_n = train_one(X, y); ok, meta = publish(horizon, best, baseline, holdout_n); published.append({"horizon": horizon, "published": ok, "model": meta.get("model") if isinstance(meta, dict) else None, "status": meta.get("status") if isinstance(meta, dict) else "published"})
    write_status({"status": "complete", "source": source, "rows": len(rows), "published": published}); return 0

if __name__ == "__main__": raise SystemExit(main())
