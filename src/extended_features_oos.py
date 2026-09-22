"""Research-only OOS test for extending the BTC feature schema.

Tests whether already-persisted cutoff features that are currently excluded from
the production schema improve the current production model under chronological
walk-forward evaluation. No production artifact is changed.
"""
from __future__ import annotations

import json
import math
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import joblib

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss
try:
    from lightgbm import LGBMClassifier
except ImportError:
    LGBMClassifier = None

from binance_history import binance_archive_rows
from feature_schema import FEATURES
from label_policy import direction_from_return
from model_compare import _strict_pit_provenance_ok, CLASSES, TEST_BLOCK, MIN_TRAIN, MIN_OOS, PURGE_BARS, EMBARGO_BARS, metrics, aligned, apply_temperature, _temperature

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data" / "predictions.db"
OUT = ROOT / "data" / "historical_research" / "extended_features_oos.json"

EXTRA_FEATURES = ("ret_15m", "ret_30m", "range_position_30m", "trend_alignment")
ALL_FEATURES = tuple(FEATURES) + EXTRA_FEATURES
FINAL_HOLDOUT_FRAC = 0.20


def _norm(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-7, 1.0)
    return p / p.sum(axis=1, keepdims=True)


def _metrics(y, p):
    return metrics(y, p)


def load_rows(horizon: str):
    actual = f"actual_direction_{horizon}"
    pu, pd, pf = f"p_up_{horizon}", f"p_down_{horizon}", f"p_flat_{horizon}"
    with sqlite3.connect(DB) as con:
        raw = con.execute(
            f"""SELECT prediction_id,created_at_utc,feature_json,{actual},{pu},{pd},{pf},scenario_json
                FROM predictions
                WHERE {actual} IS NOT NULL
                ORDER BY created_at_utc,prediction_id"""
        ).fetchall()

    out = []
    for rid, created, feature_json, y, up, down, flat, scenario_json in raw:
        if y not in CLASSES:
            continue
        try:
            scenario = json.loads(scenario_json or "{}")
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        if scenario.get("production_mode") != "binance_primary":
            continue
        if not _strict_pit_provenance_ok(scenario, created):
            continue
        try:
            f = json.loads(feature_json or "{}")
            base = [float(f[k]) for k in FEATURES]
            extra = [float(f[k]) for k in EXTRA_FEATURES]
            production = [float(down), float(flat), float(up)]
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
        vals = base + extra + production
        if not all(math.isfinite(x) for x in vals):
            continue
        if min(production) < 0 or sum(production) <= 0:
            continue
        out.append({
            "id": rid,
            "created": created,
            "base": base,
            "extended": base + extra,
            "y": y,
            "production": _norm([production])[0].tolist(),
        })
    return out


def _ema(values, span):
    a = 2.0 / (float(span) + 1.0)
    e = float(values[0])
    for value in values[1:]:
        e = a * float(value) + (1.0 - a) * e
    return e


def _ret(closes, n):
    if len(closes) <= n:
        raise ValueError(f"insufficient_history_for_return:{n}")
    return float(closes[-1]) / float(closes[-1 - n]) - 1.0


def _feature_snapshot(rows):
    """Mirror the live predictor's causal feature equations exactly."""
    if len(rows) < 31:
        raise ValueError(f"insufficient_history_for_features:{len(rows)}")
    c = np.asarray([float(x[4]) for x in rows], dtype=float)
    o = np.asarray([float(x[1]) for x in rows], dtype=float)
    h = np.asarray([float(x[2]) for x in rows], dtype=float)
    l = np.asarray([float(x[3]) for x in rows], dtype=float)
    v = np.asarray([float(x[5]) for x in rows], dtype=float)
    if not all(np.all(np.isfinite(a)) for a in (c, o, h, l, v)):
        raise ValueError("archive_feature_input_nonfinite")
    if np.any(c <= 0) or np.any(o <= 0) or np.any(h <= 0) or np.any(l <= 0) or np.any(v < 0):
        raise ValueError("archive_feature_input_domain")
    if np.any(h < np.maximum(o, c)) or np.any(l > np.minimum(o, c)):
        raise ValueError("archive_feature_input_ohlc_invalid")
    p = float(c[-1])
    r1, r3, r5, r10, r15, r30 = [_ret(c, n) for n in (1, 3, 5, 10, 15, 30)]
    acceleration = r1 - r3 / 3.0
    rv5 = float(np.std(np.diff(c[-6:]) / c[-6:-1]))
    rv10 = float(np.std(np.diff(c[-11:]) / c[-11:-1]))
    hi10, lo10 = float(np.max(h[-10:])), float(np.min(l[-10:]))
    range_position_10m = (p - lo10) / (hi10 - lo10) if hi10 > lo10 else 0.5
    hi30, lo30 = float(np.max(h[-30:])), float(np.min(l[-30:]))
    range_position_30m = (p - lo30) / (hi30 - lo30) if hi30 > lo30 else 0.5
    body = (p - float(o[-1])) / p
    upper = (float(h[-1]) - max(float(o[-1]), p)) / p
    lower = (min(float(o[-1]), p) - float(l[-1])) / p
    recent_vol = float(np.mean(v[-5:]))
    prior_vol = float(np.mean(v[-15:-5]))
    volume_ratio = recent_vol / prior_vol if prior_vol else 1.0
    volume_trend = recent_vol / max(1e-12, float(np.mean(v[-10:]))) if np.mean(v[-10:]) else 1.0
    return {
        "ret_1m": r1, "ret_3m": r3, "ret_5m": r5, "ret_10m": r10,
        "ret_15m": r15, "ret_30m": r30, "acceleration": acceleration,
        "volatility_5m": rv5, "volatility_10m": rv10,
        "range_position_10m": range_position_10m,
        "range_position_30m": range_position_30m,
        "body_1m": body, "upper_wick_1m": upper, "lower_wick_1m": lower,
        "volume_ratio": volume_ratio, "volume_trend": volume_trend,
        "ema_gap_5m": p / _ema(c[-20:], 5) - 1.0,
        "ema_gap_10m": p / _ema(c[-30:], 10) - 1.0,
        "trend_alignment": 0.50 * r5 + 0.30 * r15 + 0.20 * r30,
    }


def _frozen_champion_probs(model, base):
    raw = np.asarray(model.predict_proba(np.asarray([base], dtype=float))[0], dtype=float)
    out = np.full(3, 1e-7, dtype=float)
    for cls, prob in zip(getattr(model, "classes_", []), raw):
        name = str(cls)
        if name in CLASSES:
            out[CLASSES.index(name)] = float(prob)
    if not np.isfinite(out).all() or np.any(out < 0) or float(out.sum()) <= 0:
        raise ValueError("archive_champion_probability_invalid")
    out /= out.sum()
    return out.tolist()


def load_archive_rows(horizon: str, max_rows: int = 12000):
    """Reconstruct causal rows from free closed Binance Vision candles.

    This cohort is descriptive research only: it is explicitly ineligible for
    production promotion because archive snapshots do not provide source-native
    publication/availability timestamps equivalent to live PIT records.
    """
    horizon_steps = int(horizon[:-1])
    target_rows = int(max_rows)
    model_path = ROOT / "models" / f"{horizon}.joblib"
    meta_path = ROOT / "models" / f"{horizon}.json"
    if not model_path.is_file() or not meta_path.is_file():
        return []
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    trained_raw = meta.get("trained_at_utc")
    if not trained_raw:
        return []
    try:
        trained_at = datetime.fromisoformat(str(trained_raw).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return []
    raw = binance_archive_rows(target_rows + horizon_steps + 40)
    if len(raw) < MIN_TRAIN + MIN_OOS + horizon_steps + 40:
        return []
    champion = joblib.load(model_path)
    out = []
    for i in range(30, len(raw) - horizon_steps):
        created = datetime.fromtimestamp(int(raw[i][0]) / 1000.0, timezone.utc)
        target = datetime.fromtimestamp(int(raw[i + horizon_steps][0]) / 1000.0, timezone.utc)
        # Never replay the frozen Champion on observations at or before the
        # model-generation training timestamp.
        if created <= trained_at:
            continue
        snapshot = _feature_snapshot(raw[: i + 1])
        base = [float(snapshot[k]) for k in FEATURES]
        extra = [float(snapshot[k]) for k in EXTRA_FEATURES]
        if not all(math.isfinite(v) for v in base + extra):
            continue
        future_return = float(raw[i + horizon_steps][4]) / float(raw[i][4]) - 1.0
        out.append({
            "id": f"archive:binance_vision:{int(raw[i][0])}:{horizon}",
            "created": created.isoformat(),
            "target": target.isoformat(),
            "base": base,
            "extended": base + extra,
            "y": direction_from_return(future_return),
            "production": _frozen_champion_probs(champion, base),
        })
    return out[-target_rows:]


def load_research_rows(horizon: str):
    live = load_rows(horizon)
    required = MIN_TRAIN + MIN_OOS
    if len(live) >= required:
        return live, "live_binance_primary", True, "strict_live_primary"
    try:
        archive = load_archive_rows(horizon)
    except Exception as exc:
        archive = []
        print(f"archive fallback unavailable for {horizon}: {type(exc).__name__}: {exc}")
    if len(archive) > len(live):
        return archive, "binance_vision_archive", False, "frozen_champion_archive_replay"
    return live, "live_binance_primary", True, "strict_live_primary_insufficient"


def factories():
    factories = {
        "logreg_extended": lambda: Pipeline([
            ("scale", StandardScaler()),
            ("model", LogisticRegression(C=0.3, max_iter=3000)),
        ]),
        "extra_trees_extended": lambda: ExtraTreesClassifier(
            n_estimators=400,
            max_depth=8,
            min_samples_leaf=10,
            max_features="sqrt",
            random_state=42,
            n_jobs=-1,
        ),
        "hgb_extended": lambda: HistGradientBoostingClassifier(
            max_iter=220,
            max_leaf_nodes=15,
            learning_rate=0.04,
            l2_regularization=1.0,
            random_state=42,
        ),
    }
    if LGBMClassifier is not None:
        factories["lightgbm_extended"] = lambda: LGBMClassifier(
            objective="multiclass",
            num_class=3,
            n_estimators=300,
            learning_rate=0.03,
            num_leaves=31,
            min_child_samples=30,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
            verbosity=-1,
        )
    return factories


def candidate_block(factory, train, test):
    if len(train) < MIN_TRAIN or len(set(r["y"] for r in train)) < 3:
        return None
    split = max(int(len(train) * 0.75), MIN_TRAIN - 100)
    if split < MIN_TRAIN - 200 or len(train) - split < 30:
        return None
    cal_train = train[:split]
    cal_eval = train[split:]
    if len(set(r["y"] for r in cal_train)) < 3:
        return None

    cal_model = factory()
    cal_model.fit(np.asarray([r["extended"] for r in cal_train], dtype=float),
                  np.asarray([r["y"] for r in cal_train]))
    cal_probs = aligned(cal_model, np.asarray([r["extended"] for r in cal_eval], dtype=float))
    temperature = _temperature(cal_probs, [r["y"] for r in cal_eval])

    model = factory()
    model.fit(np.asarray([r["extended"] for r in train], dtype=float),
              np.asarray([r["y"] for r in train]))
    probs = aligned(model, np.asarray([r["extended"] for r in test], dtype=float))
    return apply_temperature(probs, temperature)


def evaluate(horizon: str):
    rows, data_source, promotion_evidence_eligible, baseline_source = load_research_rows(horizon)
    if len(rows) < MIN_TRAIN + MIN_OOS:
        return {
            "status": "DEFERRED",
            "reason": "insufficient_rows",
            "n": len(rows),
            "data_source": data_source,
            "promotion_evidence_eligible": promotion_evidence_eligible,
            "baseline_source": baseline_source,
        }

    split = int(len(rows) * (1.0 - FINAL_HOLDOUT_FRAC))
    development, holdout = rows[:split], rows[split:]
    if len(development) < MIN_TRAIN + MIN_OOS:
        return {
            "status": "DEFERRED",
            "reason": "insufficient_development_rows",
            "n": len(rows),
            "data_source": data_source,
            "promotion_evidence_eligible": promotion_evidence_eligible,
            "baseline_source": baseline_source,
        }

    prod_by_id = {r["id"]: r["production"] for r in rows}
    results = {}

    for name, factory in factories().items():
        ys, prod_probs, cand_probs, blocks = [], [], [], []
        for end in range(MIN_TRAIN, len(development), TEST_BLOCK):
            train_end = max(0, end - PURGE_BARS[horizon] - EMBARGO_BARS[horizon])
            train = development[:train_end]
            test = development[end:min(end + TEST_BLOCK, len(development))]
            if len(train) < MIN_TRAIN or len(test) < max(10, TEST_BLOCK // 2):
                continue
            cp = candidate_block(factory, train, test)
            if cp is None:
                continue
            y = [r["y"] for r in test]
            pp = [prod_by_id[r["id"]] for r in test]
            cm = _metrics(y, cp)
            pm = _metrics(y, pp)
            blocks.append({
                "n": len(y),
                "logloss_delta": cm["logloss"] - pm["logloss"],
                "brier_delta": cm["brier"] - pm["brier"],
                "accuracy_delta": cm["accuracy"] - pm["accuracy"],
            })
            ys.extend(y)
            prod_probs.extend(pp)
            cand_probs.extend(cp.tolist())

        if len(ys) < MIN_OOS:
            continue

        cm = _metrics(ys, cand_probs)
        pm = _metrics(ys, prod_probs)
        ll = np.asarray([b["logloss_delta"] for b in blocks], dtype=float)
        br = np.asarray([b["brier_delta"] for b in blocks], dtype=float)
        ac = np.asarray([b["accuracy_delta"] for b in blocks], dtype=float)
        results[name] = {
            "development": {
                "candidate": cm,
                "production": pm,
                "delta": {
                    "accuracy": cm["accuracy"] - pm["accuracy"],
                    "logloss": cm["logloss"] - pm["logloss"],
                    "brier": cm["brier"] - pm["brier"],
                },
            },
            "block_stability": {
                "blocks": len(blocks),
                "improved_logloss_ratio": float(np.mean(ll < 0)) if len(ll) else 0.0,
                "improved_brier_ratio": float(np.mean(br < 0)) if len(br) else 0.0,
                "non_worse_accuracy_ratio": float(np.mean(ac >= -0.005)) if len(ac) else 0.0,
            },
            "blocks_detail": blocks,
            "eligible_pending_frozen_holdout_confirmation": bool(
                len(blocks) >= 8
                and float(np.mean(ll < 0)) >= 0.60
                and float(np.mean(br < 0)) >= 0.60
                and cm["logloss"] <= pm["logloss"] - 0.003
                and cm["brier"] <= pm["brier"] - 0.0015
                and cm["accuracy"] >= pm["accuracy"] - 0.005
            ),
        }

    holdout_result = {
        "production": _metrics([r["y"] for r in holdout], [r["production"] for r in holdout]),
        "candidate_evaluations": {},
    }
    for name, factory in factories().items():
        if name not in results:
            continue
        cp = candidate_block(factory, development, holdout)
        if cp is None:
            holdout_result["candidate_evaluations"][name] = {"status": "DEFERRED"}
        else:
            holdout_result["candidate_evaluations"][name] = {
                "candidate": _metrics([r["y"] for r in holdout], cp),
                "production": holdout_result["production"],
            }

    return {
        "status": "OK",
        "schema_version": 1,
        "research_only": True,
        "production_changed": False,
        "final_holdout_protected": True,
        "horizon": horizon,
        "n": len(rows),
        "development_n": len(development),
        "final_holdout_n": len(holdout),
        "base_features": list(FEATURES),
        "extra_features": list(EXTRA_FEATURES),
        "data_source": data_source,
        "promotion_evidence_eligible": promotion_evidence_eligible,
        "baseline_source": baseline_source,
        "candidate_results": results,
        "final_holdout": holdout_result,
    }


def main():
    if not DB.exists():
        raise SystemExit("prediction database missing")
    payload = {
        "schema_version": 1,
        "research_only": True,
        "production_changed": False,
        "feature_set_policy": "existing_cutoff_persisted_features_only_no_future_inputs",
        "archive_policy": "free_closed_binance_vision_frozen_champion_replay_research_only",
        "horizons": {h: evaluate(h) for h in ("5m", "10m")},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
