"""Research-only rich-feature challenger for BTC short-horizon production.

Purpose:
- Build the same causal 1m panel used by historical v6, plus the two EMA
  features required by the current 15-feature Champion.
- Score the current Champion, a retrained legacy-feature baseline, and richer
  candidates on the exact same frozen chronological holdout.
- Select models on development data only.
- Persist a candidate artifact only when the frozen holdout and stability gates
  support a meaningful gain.
- Never modify production artifacts or the production registry.
"""
from __future__ import annotations

import json
import math
import sqlite3
from datetime import datetime, timezone, timedelta
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from historical_research import load_market, exists, _window_is_contiguous
except ModuleNotFoundError:
    from src.historical_research import load_market, exists, _window_is_contiguous

try:
    from label_policy import direction_from_return, NEUTRAL_BPS
except ModuleNotFoundError:
    from src.label_policy import direction_from_return, NEUTRAL_BPS

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
OUT_DIR = ROOT / "data" / "historical_research"
OUT = OUT_DIR / "rich_production_challenger.json"

CLASSES = ("DOWN", "FLAT", "UP")
DAYS = 45
DEV_FRACTION = 0.70
MIN_COMMON_ROWS = 20_000
MIN_HOLDOUT_ROWS = 2_000
TEST_BLOCK = 1_000

# Original historical-v6 features (41) + the exact two EMA gaps used by the
# current Champion, so the legacy 15-feature slice is bit-compatible in meaning.
RICH_FEATURES = (
    "ret1","ret3","ret5","ret10","ret15","ret30","accel","rv5","rv10","rv30",
    "rangepos10","rangepos30","body","upper","lower","volratio","voltrend",
    "tradesratio","takerimb","basis","basis_delta","mark_gap","premium",
    "eth_ret5","sol_ret5","eth_ret10","sol_ret10","eth_btc_rel5","sol_btc_rel5",
    "ret5_x_vol","ret10_x_vol","flow_x_vol","range_x_flow","hour_sin","hour_cos",
    "dow_sin","dow_cos","funding","funding_delta","oi_change","oi_z",
    "ema_gap_5m","ema_gap_10m",
)

# Exact current production feature order.
LEGACY_FEATURES = (
    "ret_1m","ret_3m","ret_5m","ret_10m","acceleration","volatility_5m",
    "volatility_10m","range_position_10m","body_1m","upper_wick_1m",
    "lower_wick_1m","volume_ratio","volume_trend","ema_gap_5m","ema_gap_10m",
)
LEGACY_INDICES = (0,1,2,3,6,7,8,10,12,13,14,15,16,41,42)

def _ema(values: np.ndarray, span: int) -> float:
    a = 2.0 / (span + 1.0)
    e = float(values[0])
    for x in values[1:]:
        e = a * float(x) + (1.0 - a) * e
    return e

def _ret(values: np.ndarray, n: int) -> float:
    return float(values[-1] / values[-1-n] - 1.0)

def _metrics(y: list[str], probs: np.ndarray) -> dict[str, float | int]:
    p = np.asarray(probs, dtype=float)
    p = np.clip(p, 1e-8, 1.0)
    p = p / p.sum(axis=1, keepdims=True)
    yi = np.asarray([CLASSES.index(v) for v in y], dtype=int)
    pred = np.argmax(p, axis=1)
    hit = (pred == yi)
    conf = p.max(axis=1)
    ece = 0.0
    for k in range(10):
        lo, hi = k / 10.0, (k + 1) / 10.0
        mask = (conf >= lo) & ((conf < hi) if hi < 1.0 else (conf <= hi))
        if np.any(mask):
            ece += float(mask.mean()) * abs(float(hit[mask].mean()) - float(conf[mask].mean()))
    logloss = -float(np.mean(np.log(np.clip(p[np.arange(len(y)), yi], 1e-8, 1.0))))
    one = np.eye(3)[yi]
    brier = float(np.mean(np.sum((p - one) ** 2, axis=1)))
    return {
        "n": int(len(y)),
        "accuracy": float(hit.mean()),
        "logloss": logloss,
        "brier": brier,
        "ece": float(ece),
        "mean_confidence": float(conf.mean()),
    }

def _predict_aligned(model, X: np.ndarray) -> np.ndarray:
    raw = np.asarray(model.predict_proba(X), dtype=float)
    out = np.full((len(X), 3), 1e-8, dtype=float)
    for j, cls in enumerate(model.classes_):
        cls = str(cls)
        if cls in CLASSES:
            out[:, CLASSES.index(cls)] = raw[:, j]
    out = np.clip(out, 1e-8, 1.0)
    return out / out.sum(axis=1, keepdims=True)

def _factories() -> dict[str, callable]:
    return {
        "logreg": lambda: Pipeline([
            ("scale", StandardScaler()),
            ("model", LogisticRegression(C=0.5, max_iter=3000)),
        ]),
        "rf": lambda: RandomForestClassifier(
            n_estimators=500, max_depth=10, min_samples_leaf=10,
            max_features="sqrt", random_state=42, n_jobs=-1
        ),
        "rf_balanced": lambda: RandomForestClassifier(
            n_estimators=500, max_depth=10, min_samples_leaf=10,
            max_features="sqrt", class_weight="balanced_subsample",
            random_state=42, n_jobs=-1
        ),
        "extra_trees": lambda: ExtraTreesClassifier(
            n_estimators=500, max_depth=10, min_samples_leaf=10,
            max_features="sqrt", random_state=42, n_jobs=-1
        ),
        "extra_balanced": lambda: ExtraTreesClassifier(
            n_estimators=500, max_depth=10, min_samples_leaf=10,
            max_features="sqrt", class_weight="balanced",
            random_state=42, n_jobs=-1
        ),
        "hgb": lambda: HistGradientBoostingClassifier(
            max_iter=300, max_leaf_nodes=31, learning_rate=0.035,
            l2_regularization=1.5, random_state=42
        ),
    }

def build_panel() -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray], np.ndarray]:
    end = (datetime.now(timezone.utc) - timedelta(days=3)).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    start = end - timedelta(days=DAYS)
    raw = load_market(start, end)
    maps = {k: {int(r[0]): r for r in v} for k, v in raw.items() if k not in ("funding", "oi")}
    if not maps.get("btc_fut"):
        raise RuntimeError("missing_btc_futures_history")
    if not maps.get("btc_spot"):
        maps["btc_spot"] = dict(maps["btc_fut"])
    if not maps.get("btc_mark"):
        maps["btc_mark"] = dict(maps["btc_fut"])
    if not maps.get("btc_premium"):
        maps["btc_premium"] = dict(maps["btc_fut"])

    funding = {int(r["fundingTime"]): float(r["fundingRate"]) for r in raw.get("funding", [])}
    oi = {int(r["timestamp"]): float(r["sumOpenInterest"]) for r in raw.get("oi", [])}
    common = sorted(
        set(maps["btc_fut"]) & set(maps["btc_spot"]) &
        set(maps["eth_fut"]) & set(maps["sol_fut"])
    )
    if len(common) < MIN_COMMON_ROWS:
        raise RuntimeError(f"insufficient_common_rows:{len(common)}")

    fk, ok = sorted(funding), sorted(oi)
    rows: list[tuple[int, list[float], float]] = []

    for i, t in enumerate(common):
        if i < 50:
            continue
        w = common[i-50:i+1]
        if not _window_is_contiguous(w):
            continue

        def close(key):
            return np.asarray([float(maps[key][x][4]) for x in w], dtype=float)

        b = close("btc_fut")
        s = close("btc_spot")
        ec = close("eth_fut")
        sc = close("sol_fut")
        bo = np.asarray([float(maps["btc_fut"][x][1]) for x in w])
        bh = np.asarray([float(maps["btc_fut"][x][2]) for x in w])
        bl = np.asarray([float(maps["btc_fut"][x][3]) for x in w])
        bv = np.asarray([float(maps["btc_fut"][x][5]) for x in w])
        bt = np.asarray([float(maps["btc_fut"][x][8]) for x in w])
        tb = np.asarray([float(maps["btc_fut"][x][9]) for x in w])

        mark = close("btc_mark")
        prem = close("btc_premium")
        p = float(b[-1])
        r1,r3,r5,r10,r15,r30 = [_ret(b,n) for n in (1,3,5,10,15,30)]
        accel = r1 - r3 / 3.0
        rv5 = float(np.std(np.diff(b[-6:]) / b[-6:-1]))
        rv10 = float(np.std(np.diff(b[-11:]) / b[-11:-1]))
        rv30 = float(np.std(np.diff(b[-31:]) / b[-31:-1]))
        hi10, lo10 = float(max(bh[-10:])), float(min(bl[-10:]))
        hi30, lo30 = float(max(bh[-30:])), float(min(bl[-30:]))
        rp10 = (p - lo10) / (hi10 - lo10) if hi10 > lo10 else 0.5
        rp30 = (p - lo30) / (hi30 - lo30) if hi30 > lo30 else 0.5
        body = (p - bo[-1]) / p
        upper = (bh[-1] - max(bo[-1], p)) / p
        lower = (min(bo[-1], p) - bl[-1]) / p
        vr = float(np.mean(bv[-5:])) / max(1e-12, float(np.mean(bv[-15:-5])))
        vt = float(np.mean(bv[-5:])) / max(1e-12, float(np.mean(bv[-10:])))
        tr = float(np.mean(bt[-5:])) / max(1e-12, float(np.mean(bt[-15:-5])))
        flow = 2.0 * float(np.sum(tb[-5:])) / max(1e-12, float(np.sum(bv[-5:]))) - 1.0
        basis = p / max(1e-12, float(s[-1])) - 1.0
        bd = basis - (b[-2] / max(1e-12, s[-2]) - 1.0)
        mg = float(mark[-1] / p - 1.0)
        pr = float(prem[-1])
        er5, sr5, er10, sr10 = _ret(ec,5), _ret(sc,5), _ret(ec,10), _ret(sc,10)
        erbtc5, srb5 = er5 - r5, sr5 - r5

        ft = max([k for k in fk if k <= t], default=None)
        ot = max([k for k in ok if k <= t], default=None)
        funding_v = funding.get(ft, 0.0) if ft is not None else 0.0
        prev_f = max([k for k in fk if k < ft], default=None) if ft is not None else None
        funding_delta = funding_v - (funding.get(prev_f, funding_v) if prev_f is not None else funding_v)
        oi_v = oi.get(ot, 0.0) if ot is not None else 0.0
        prev_oi = max([k for k in ok if k < ot], default=None) if ot is not None else None
        oi_change = (
            oi_v / oi.get(prev_oi, oi_v) - 1.0
            if prev_oi is not None and oi.get(prev_oi, 0.0)
            else 0.0
        )
        recent_oi = [oi[k] for k in ok if k <= t][-96:]
        oi_z = (
            (oi_v - np.mean(recent_oi)) / max(1e-12, np.std(recent_oi))
            if recent_oi and np.isfinite(oi_v) else 0.0
        )

        dt = datetime.fromtimestamp(t / 1000.0, timezone.utc)
        hour = dt.hour + dt.minute / 60.0
        hs, hc = math.sin(2*math.pi*hour/24.0), math.cos(2*math.pi*hour/24.0)
        dow = dt.weekday()
        ds, dc = math.sin(2*math.pi*dow/7.0), math.cos(2*math.pi*dow/7.0)
        ema5 = p / _ema(b[-20:], 5) - 1.0
        ema10 = p / _ema(b[-30:], 10) - 1.0

        x = [
            r1,r3,r5,r10,r15,r30,accel,rv5,rv10,rv30,rp10,rp30,body,upper,lower,
            vr,vt,tr,flow,basis,bd,mg,pr,er5,sr5,er10,sr10,erbtc5,srb5,
            r5*rv10,r10*rv10,flow*rv5,rp10*flow,hs,hc,ds,dc,
            funding_v,funding_delta,oi_change,oi_z,ema5,ema10
        ]
        if all(math.isfinite(float(v)) for v in x):
            rows.append((int(t), [float(v) for v in x], p))

    if len(rows) < MIN_COMMON_ROWS:
        raise RuntimeError(f"insufficient_valid_panel_rows:{len(rows)}")

    rows.sort(key=lambda z: z[0])
    # Exact elapsed-time labels. No row-offset stretch across missing minutes.
    return _label_panel(rows, 5), _label_panel(rows, 10), np.asarray([r[0] for r in rows], dtype=np.int64)

def _label_panel(rows: list[tuple[int,list[float],float]], horizon: int) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
    by_ts = {int(r[0]): r for r in rows}
    X, y, t = [], [], []
    for ts, x, base in rows:
        future = by_ts.get(int(ts) + horizon * 60_000)
        if future is None:
            continue
        future_price = float(future[2])
        ret_bps = (future_price / float(base) - 1.0) * 10_000.0
        X.append(x)
        y.append(direction_from_return(ret_bps / 10_000.0))
        t.append(ts)
    Xn = np.asarray(X, dtype=float)
    yn = np.asarray(y, dtype=object)
    tn = np.asarray(t, dtype=np.int64)
    if len(yn) < MIN_COMMON_ROWS // 2:
        raise RuntimeError(f"insufficient_labeled_rows:{horizon}:{len(yn)}")
    return Xn, yn, tn

def _fit_selected(X_dev: np.ndarray, y_dev: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, feature_indices: list[int]) -> tuple[str, object, list[dict]]:
    results = []
    fac = _factories()
    for name, make in fac.items():
        model = make()
        model.fit(X_dev[:, feature_indices], y_dev)
        p = _predict_aligned(model, X_val[:, feature_indices])
        m = _metrics(y_val.tolist(), p)
        results.append({"model": name, **m})
    results.sort(key=lambda r: (-r["accuracy"], r["logloss"], r["brier"]))
    winner_name = str(results[0]["model"])
    winner = fac[winner_name]()
    winner.fit(X_dev[:, feature_indices], y_dev)
    return winner_name, winner, results

def _stability(y: np.ndarray, rich_p: np.ndarray, base_p: np.ndarray) -> dict:
    n = len(y)
    blocks = []
    for a in range(0, n, TEST_BLOCK):
        b = min(n, a + TEST_BLOCK)
        if b - a < 200:
            continue
        rm = _metrics(y[a:b].tolist(), rich_p[a:b])
        bm = _metrics(y[a:b].tolist(), base_p[a:b])
        blocks.append({
            "start": a,
            "n": b-a,
            "rich_accuracy": rm["accuracy"],
            "baseline_accuracy": bm["accuracy"],
            "accuracy_delta": rm["accuracy"] - bm["accuracy"],
            "rich_logloss": rm["logloss"],
            "baseline_logloss": bm["logloss"],
            "logloss_delta": rm["logloss"] - bm["logloss"],
            "brier_delta": rm["brier"] - bm["brier"],
        })
    if not blocks:
        return {"blocks": 0, "accuracy_non_worse_ratio": 0.0, "mean_accuracy_delta": 0.0}
    deltas = np.asarray([b["accuracy_delta"] for b in blocks], dtype=float)
    return {
        "blocks": len(blocks),
        "accuracy_non_worse_ratio": float(np.mean(deltas >= 0.0)),
        "mean_accuracy_delta": float(deltas.mean()),
        "min_accuracy_delta": float(deltas.min()),
        "max_accuracy_delta": float(deltas.max()),
        "blocks_detail": blocks,
    }

def _block_bootstrap_ci(y: np.ndarray, rich_p: np.ndarray, base_p: np.ndarray, seed: int = 42) -> dict:
    n = len(y)
    edges = list(range(0, n, TEST_BLOCK))
    blocks = [(a, min(n, a + TEST_BLOCK)) for a in edges if min(n, a + TEST_BLOCK) - a >= 200]
    if len(blocks) < 5:
        return {"n_blocks": len(blocks), "low": None, "high": None}
    yi = np.asarray([CLASSES.index(v) for v in y.tolist()], dtype=int)
    rich_hit = np.argmax(rich_p, axis=1) == yi
    base_hit = np.argmax(base_p, axis=1) == yi
    block_diffs = np.asarray([
        float(rich_hit[a:b].mean() - base_hit[a:b].mean()) for a,b in blocks
    ])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(block_diffs), size=(1000, len(block_diffs)))
    boot = block_diffs[idx].mean(axis=1)
    return {
        "n_blocks": len(blocks),
        "mean_block_accuracy_delta": float(block_diffs.mean()),
        "ci95_low": float(np.quantile(boot, 0.025)),
        "ci95_high": float(np.quantile(boot, 0.975)),
    }

def _relative_gain(base: float, cand: float, higher_is_better: bool = False) -> float:
    if higher_is_better:
        return (cand - base) / max(abs(base), 1e-12)
    return (base - cand) / max(abs(base), 1e-12)

def evaluate_horizon(horizon: str, X: np.ndarray, y: np.ndarray, t: np.ndarray) -> dict:
    n = len(y)
    order = np.argsort(t)
    X, y, t = X[order], y[order], t[order]
    split = int(n * DEV_FRACTION)
    if split < 12_000 or n - split < MIN_HOLDOUT_ROWS:
        raise RuntimeError(f"insufficient_challenger_split:{horizon}:n={n}:split={split}")
    X_dev, y_dev = X[:split], y[:split]
    X_hold, y_hold = X[split:], y[split:]

    inner = int(len(y_dev) * 0.75)
    X_tr, y_tr = X_dev[:inner], y_dev[:inner]
    X_val, y_val = X_dev[inner:], y_dev[inner:]

    rich_idx = list(range(len(RICH_FEATURES)))
    legacy_idx = list(LEGACY_INDICES)

    rich_name, rich_model, rich_validation = _fit_selected(X_tr, y_tr, X_val, y_val, rich_idx)
    legacy_name, legacy_model, legacy_validation = _fit_selected(X_tr, y_tr, X_val, y_val, legacy_idx)

    # Freeze the model choices after development selection, then refit on all
    # development data. The frozen holdout is never used for model selection.
    rich_model = _factories()[rich_name]()
    rich_model.fit(X_dev[:, rich_idx], y_dev)
    legacy_model = _factories()[legacy_name]()
    legacy_model.fit(X_dev[:, legacy_idx], y_dev)

    rich_hold = _predict_aligned(rich_model, X_hold[:, rich_idx])
    legacy_hold = _predict_aligned(legacy_model, X_hold[:, legacy_idx])

    # Direct current-Champion comparison on the exact same timestamps.
    champion_path = MODEL_DIR / f"{horizon}.joblib"
    champion_meta_path = MODEL_DIR / f"{horizon}.json"
    if not champion_path.is_file() or not champion_meta_path.is_file():
        raise RuntimeError(f"current_champion_missing:{horizon}")
    champion = joblib.load(champion_path)
    champion_hold = _predict_aligned(champion, X_hold[:, legacy_idx])

    rich_m = _metrics(y_hold.tolist(), rich_hold)
    legacy_m = _metrics(y_hold.tolist(), legacy_hold)
    champion_m = _metrics(y_hold.tolist(), champion_hold)

    stability_vs_champion = _stability(y_hold, rich_hold, champion_hold)
    stability_vs_legacy = _stability(y_hold, rich_hold, legacy_hold)
    ci_vs_champion = _block_bootstrap_ci(y_hold, rich_hold, champion_hold)
    ci_vs_legacy = _block_bootstrap_ci(y_hold, rich_hold, legacy_hold)

    acc_rel_vs_champion = _relative_gain(champion_m["accuracy"], rich_m["accuracy"], True)
    acc_rel_vs_legacy = _relative_gain(legacy_m["accuracy"], rich_m["accuracy"], True)
    ll_rel_vs_champion = _relative_gain(champion_m["logloss"], rich_m["logloss"], False)
    br_rel_vs_champion = _relative_gain(champion_m["brier"], rich_m["brier"], False)

    eligible = bool(
        rich_m["n"] >= MIN_HOLDOUT_ROWS
        and acc_rel_vs_champion >= 0.03
        and rich_m["accuracy"] >= champion_m["accuracy"] + 0.005
        and ll_rel_vs_champion >= 0.03
        and br_rel_vs_champion >= 0.01
        and stability_vs_champion["accuracy_non_worse_ratio"] >= 0.70
        and (ci_vs_champion["ci95_low"] is None or ci_vs_champion["ci95_low"] > 0.0)
    )

    candidate_artifact = None
    candidate_meta = None
    if eligible:
        candidate_artifact = MODEL_DIR / f"{horizon}.rich_candidate.joblib"
        candidate_meta = MODEL_DIR / f"{horizon}.rich_candidate.json"
        joblib.dump(rich_model, candidate_artifact)
        candidate_meta.write_text(json.dumps({
            "model_version": f"rich_challenger.{horizon}.dev{split}",
            "horizon": horizon,
            "features": list(RICH_FEATURES),
            "artifact": candidate_artifact.name,
            "candidate": True,
            "research_only": True,
            "production_changed": False,
            "selected_model": rich_name,
            "development_n": int(split),
            "frozen_holdout_n": int(len(y_hold)),
            "holdout_metrics": rich_m,
            "comparison_current_champion": champion_m,
            "comparison_legacy_retrained": legacy_m,
            "stability_vs_champion": stability_vs_champion,
            "bootstrap_ci_vs_champion": ci_vs_champion,
            "pit_policy": "historical_v6_closed_candles_and_exact_elapsed_target",
        }, indent=2, sort_keys=True), encoding="utf-8")

    return {
        "status": "OK",
        "horizon": horizon,
        "panel_rows": int(n),
        "development_n": int(split),
        "frozen_holdout_n": int(len(y_hold)),
        "selected_rich_model": rich_name,
        "selected_legacy_model": legacy_name,
        "validation_rich": rich_validation,
        "validation_legacy": legacy_validation,
        "frozen_holdout": {
            "current_champion": champion_m,
            "legacy_retrained": legacy_m,
            "rich_candidate": rich_m,
            "rich_vs_champion": {
                "accuracy_delta": rich_m["accuracy"] - champion_m["accuracy"],
                "accuracy_relative_gain": acc_rel_vs_champion,
                "logloss_relative_gain": ll_rel_vs_champion,
                "brier_relative_gain": br_rel_vs_champion,
            },
            "rich_vs_legacy": {
                "accuracy_delta": rich_m["accuracy"] - legacy_m["accuracy"],
                "accuracy_relative_gain": acc_rel_vs_legacy,
            },
            "stability_vs_champion": stability_vs_champion,
            "stability_vs_legacy": stability_vs_legacy,
            "bootstrap_ci_vs_champion": ci_vs_champion,
            "bootstrap_ci_vs_legacy": ci_vs_legacy,
        },
        "eligibility": eligible,
        "candidate_artifact": str(candidate_artifact.name) if candidate_artifact else None,
    }

def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (X5, y5, t5), (X10, y10, t10), _ = build_panel()
    # build_panel repeats the cached market-panel construction, so use the
    # already-built 5m panel timestamps only when labels align; otherwise each
    # horizon is evaluated on its own exact target-compatible sample.
    results = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "schema_version": 1,
        "research_only": True,
        "production_changed": False,
        "features": list(RICH_FEATURES),
        "legacy_feature_indices": list(LEGACY_INDICES),
        "horizons": {
            "5m": evaluate_horizon("5m", *X5),
            "10m": evaluate_horizon("10m", *X10),
        },
    }
    OUT.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False))
    # Research workflow should not fail the repository if no candidate clears
    # the promotion evidence threshold; rejection is an expected result.
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
