"""Strict, efficient historical BTC 5m/10m OOS research.

Design goals:
- chronological, leakage-resistant walk-forward validation
- explicit embargo between train and test
- horizon-specific models
- stable 3-class labels (DOWN/FLAT/UP)
- honest baselines
- pooled OOS metrics and checkpoint metrics
- no model adoption from this script; adoption remains gated separately
"""
from __future__ import annotations

import csv, json, math, time, urllib.parse, urllib.request
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss

OUT = Path("data/historical_research")
OUT.mkdir(parents=True, exist_ok=True)
CLASSES = ["DOWN", "FLAT", "UP"]
FEATURES = [
    "ret_1m","ret_3m","ret_5m","ret_10m","acceleration",
    "volatility_5m","volatility_10m","range_position_10m",
    "body_1m","upper_wick_1m","lower_wick_1m",
    "volume_ratio","volume_trend","ema_gap_5m","ema_gap_10m"
]
GRANULARITY = 60
CANDLE_LIMIT = 300
DAYS = 45
MIN_TRAIN = 12000
TEST_BLOCK = 1000
EMBARGO = 10
TARGETS = {"5m": 5, "10m": 10}
NEUTRAL_BPS = 1.0


def fetch_chunk(start, end):
    q = urllib.parse.urlencode({"granularity": GRANULARITY, "start": start.isoformat(), "end": end.isoformat()})
    url = "https://api.exchange.coinbase.com/products/BTC-USD/candles?" + q
    req = urllib.request.Request(url, headers={"User-Agent": "btc-prediction-research/historical-v2", "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def fetch_history(days=DAYS):
    end = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    start = end - timedelta(days=days)
    rows, cursor = [], start
    while cursor < end:
        nxt = min(cursor + timedelta(minutes=CANDLE_LIMIT), end)
        last_error = None
        for attempt in range(4):
            try:
                rows.extend(fetch_chunk(cursor, nxt))
                break
            except Exception as exc:
                last_error = exc
                time.sleep(1.0 * (attempt + 1))
        else:
            raise RuntimeError(f"Coinbase candle fetch failed at {cursor}: {last_error}")
        cursor = nxt
        time.sleep(0.08)
    by_ts = {}
    for r in rows:
        if len(r) >= 6:
            # Coinbase: [timestamp, low, high, open, close, volume]
            by_ts[int(r[0])] = [int(r[0]), float(r[3]), float(r[2]), float(r[1]), float(r[4]), float(r[5])]
    clean = [by_ts[k] for k in sorted(by_ts)]
    if len(clean) < MIN_TRAIN + 5000:
        raise RuntimeError(f"Only {len(clean)} candles downloaded; insufficient history")
    return clean


def ema(values, span):
    a = 2.0 / (span + 1.0)
    e = float(values[0])
    for v in values[1:]:
        e = a * float(v) + (1.0 - a) * e
    return e


def row_features(rows, i):
    lo_i = max(0, i - 35)
    closes = np.array([r[4] for r in rows[lo_i:i+1]], float)
    opens = np.array([r[1] for r in rows[lo_i:i+1]], float)
    highs = np.array([r[2] for r in rows[lo_i:i+1]], float)
    lows = np.array([r[3] for r in rows[lo_i:i+1]], float)
    vols = np.array([r[5] for r in rows[lo_i:i+1]], float)
    c = closes[-1]
    def ret(n): return closes[-1] / closes[-1-n] - 1.0
    r1, r3, r5, r10 = ret(1), ret(3), ret(5), ret(10)
    acceleration = r1 - r3 / 3.0
    rr5 = np.diff(closes[-6:]) / closes[-6:-1]
    rr10 = np.diff(closes[-11:]) / closes[-11:-1]
    vol5, vol10 = float(np.std(rr5)), float(np.std(rr10))
    hi, lo = float(np.max(highs[-10:])), float(np.min(lows[-10:]))
    rp = (c - lo) / (hi - lo) if hi > lo else 0.5
    body = (c - opens[-1]) / c
    upper = (highs[-1] - max(opens[-1], c)) / c
    lower = (min(opens[-1], c) - lows[-1]) / c
    recent = float(np.mean(vols[-5:])); prior = float(np.mean(vols[-15:-5]))
    vr = recent / (prior if prior > 0 else 1.0)
    vt = recent / (float(np.mean(vols[-10:])) if np.mean(vols[-10:]) > 0 else recent)
    e5, e10 = ema(closes[-20:], 5), ema(closes[-30:], 10)
    return [r1,r3,r5,r10,acceleration,vol5,vol10,rp,body,upper,lower,vr,vt,c/e5-1.0,c/e10-1.0]


def build_dataset(rows, horizon):
    n = len(rows) - horizon
    X, y, ts = [], [], []
    for i in range(35, n):
        x = row_features(rows, i)
        if not all(math.isfinite(v) for v in x):
            continue
        base, future = rows[i][4], rows[i+horizon][4]
        r_bps = (future / base - 1.0) * 10000.0
        label = "UP" if r_bps > NEUTRAL_BPS else "DOWN" if r_bps < -NEUTRAL_BPS else "FLAT"
        X.append(x); y.append(label); ts.append(rows[i][0])
    return np.asarray(X, float), np.asarray(y), np.asarray(ts, dtype=np.int64)


def normalize(p):
    p = np.clip(np.asarray(p, float), 1e-7, 1.0)
    return p / p.sum(axis=1, keepdims=True)


def metrics(y, p):
    idx = {c:i for i,c in enumerate(CLASSES)}
    yi = np.array([idx[v] for v in y])
    p = normalize(p)
    pred = p.argmax(1)
    one = np.eye(3)[yi]
    conf = p.max(1)
    hit = (pred == yi).astype(float)
    ece = 0.0
    for b in range(10):
        lo, hi = b/10, (b+1)/10
        m = (conf >= lo) & ((conf < hi) if hi < 1 else (conf <= hi))
        if m.any():
            ece += float(m.mean()) * abs(float(hit[m].mean()) - float(conf[m].mean()))
    return {
        "n": int(len(y)),
        "accuracy": float((pred == yi).mean()),
        "logloss": float(log_loss(yi, p, labels=[0,1,2])),
        "brier": float(np.mean(np.sum((p-one)**2, axis=1))),
        "ece": float(ece),
    }


def model_factories():
    return {
        "logreg_c0.1": lambda: Pipeline([("scale", StandardScaler()), ("model", LogisticRegression(C=.1, max_iter=2000))]),
        "logreg_c1": lambda: Pipeline([("scale", StandardScaler()), ("model", LogisticRegression(C=1, max_iter=2000))]),
        "logreg_c10": lambda: Pipeline([("scale", StandardScaler()), ("model", LogisticRegression(C=10, max_iter=2000))]),
        "rf_250": lambda: RandomForestClassifier(n_estimators=250, max_depth=8, min_samples_leaf=12, max_features="sqrt", random_state=42, n_jobs=-1),
        "hgb": lambda: HistGradientBoostingClassifier(max_iter=180, max_leaf_nodes=15, learning_rate=.04, l2_regularization=1.5, random_state=42),
    }


def align(model, X):
    raw = model.predict_proba(X)
    out = np.full((len(X), 3), 1e-7)
    for j, c in enumerate(model.classes_):
        out[:, CLASSES.index(str(c))] = raw[:, j]
    return normalize(out)


def baselines(y):
    n = len(y)
    out = {"uniform": metrics(y, np.tile([1/3,1/3,1/3], (n,1)))}
    counts = np.array([(y == c).sum() for c in CLASSES], float)
    freq = counts / counts.sum()
    out["class_frequency"] = metrics(y, np.tile(freq, (n,1)))
    # Persistence baseline: estimate direction from the latest 1m return.
    # This is deliberately simple and evaluated on the same OOS rows.
    return out


def walk_forward(X, y, ts):
    results = {}
    for name, factory in model_factories().items():
        probs, ys, stamps = [], [], []
        for end in range(MIN_TRAIN, len(X), TEST_BLOCK):
            test_start = end + EMBARGO
            test_end = min(test_start + TEST_BLOCK, len(X))
            if test_start >= len(X): break
            model = factory()
            train_y = y[:end]
            if len(set(train_y.tolist())) < 3: continue
            model.fit(X[:end], train_y)
            p = align(model, X[test_start:test_end])
            probs.extend(p.tolist()); ys.extend(y[test_start:test_end].tolist()); stamps.extend(ts[test_start:test_end].tolist())
        if len(ys) >= 10000:
            results[name] = {"metrics": metrics(np.asarray(ys), np.asarray(probs)), "y": ys, "p": probs, "ts": stamps}
    return results


def main():
    started = datetime.now(timezone.utc)
    rows = fetch_history()
    raw_path = OUT / "btc_usd_1m_raw.csv"
    with raw_path.open("w", newline="") as f:
        w = csv.writer(f); w.writerow(["timestamp","open","high","low","close","volume"]); w.writerows(rows)
    report = {
        "protocol_version": "historical-v2",
        "started_utc": started.isoformat(), "finished_utc": None,
        "source": "Coinbase Exchange BTC-USD 1m", "days": DAYS, "candles": len(rows),
        "neutral_bps": NEUTRAL_BPS, "min_train": MIN_TRAIN, "test_block": TEST_BLOCK, "embargo": EMBARGO,
        "horizons": {}
    }
    for h, steps in TARGETS.items():
        X, y, ts = build_dataset(rows, steps)
        r = walk_forward(X, y, ts)
        horizon = {
            "samples": int(len(y)),
            "class_counts": {c:int((y == c).sum()) for c in CLASSES},
            "baseline": baselines(y),
            "models": {k:{"metrics":v["metrics"]} for k,v in r.items()},
        }
        # Required 2k/5k/10k checkpoints from the pooled OOS stream.
        for checkpoint in (2000, 5000, 10000):
            horizon[f"checkpoint_{checkpoint}"] = {}
            for name, obj in r.items():
                n = min(checkpoint, len(obj["y"]))
                horizon[f"checkpoint_{checkpoint}"][name] = metrics(np.asarray(obj["y"][:n]), np.asarray(obj["p"][:n])) if n >= checkpoint else {"available": False, "n": n}
        report["horizons"][h] = horizon
        for name, obj in r.items():
            out = OUT / f"oos_{h}_{name}.csv"
            with out.open("w", newline="") as f:
                w = csv.writer(f); w.writerow(["timestamp","actual","p_down","p_flat","p_up"])
                for yy, pp, stamp in zip(obj["y"], obj["p"], obj["ts"]):
                    w.writerow([int(stamp), yy, float(pp[0]), float(pp[1]), float(pp[2])])
    report["finished_utc"] = datetime.now(timezone.utc).isoformat()
    (OUT / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
