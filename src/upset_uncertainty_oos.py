"""Research-only upset / uncertainty meta-layer for BTC short-horizon OOS.

Goal: estimate the probability that the historical RF champion's argmax is
wrong using only information available at prediction time, then test a
non-reversing probability adjustment. The layer never turns high risk into a
blind opposite-direction bet.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "historical_research" / "upset_uncertainty_oos.json"
CLASSES = ("DOWN", "FLAT", "UP")
HORIZONS = ("5m", "10m")
FILES = {
    "5m": {
        "rf": "oos_5m_rf.csv",
        "extra": "oos_5m_extra.csv",
        "hgb": "oos_5m_hgb.csv",
        "ensemble": "oos_5m_ensemble.csv",
    },
    "10m": {
        "rf": "oos_10m_rf.csv",
        "extra": "oos_10m_extra.csv",
        "hgb": "oos_10m_hgb.csv",
        "ensemble": "oos_10m_ensemble.csv",
    },
}
HOLDOUT_FRAC = 0.20
META_MIN_TRAIN = 2000
META_EMBARGO = 10
META_BLOCK = 500
ROLLING_SHORT = 50
ROLLING_LONG = 250
EPS = 1e-8


def _rows(path: Path) -> dict[int, dict[str, object]]:
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        out = {}
        for r in reader:
            try:
                ts = int(r["timestamp"])
                p = np.asarray([float(r["p_down"]), float(r["p_flat"]), float(r["p_up"])], dtype=float)
                if not np.isfinite(p).all() or np.any(p < 0) or float(p.sum()) <= 0:
                    continue
                p /= p.sum()
                out[ts] = {"actual": str(r["actual"]), "p": p}
            except (KeyError, TypeError, ValueError):
                continue
    return out


def _entropy(p: np.ndarray) -> float:
    return float(-np.sum(p * np.log(np.clip(p, EPS, 1.0))) / math.log(3.0))


def _features(rows: list[dict[str, object]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    errors = np.asarray([int(np.argmax(r["rf"]) != CLASSES.index(r["actual"])) for r in rows], dtype=int)
    x = []
    for i, r in enumerate(rows):
        rf = r["rf"]
        extra = r["extra"]
        hgb = r["hgb"]
        ens = r["ensemble"]
        ps = np.stack([rf, extra, hgb, ens], axis=0)
        order = np.sort(rf)[::-1]
        mean_pair_l1 = float(np.mean(np.abs(ps[:, None, :] - ps[None, :, :]))) / 2.0
        class_dispersion = float(np.mean(np.std(ps, axis=0)))
        nonrf = (extra + hgb) / 2.0
        nonrf /= nonrf.sum()
        consensus = float(np.sum(np.argmax(ps[:3], axis=1) == np.argmax(nonrf)))
        rolling_short = float(np.mean(errors[max(0, i - ROLLING_SHORT):i])) if i else 0.5
        rolling_long = float(np.mean(errors[max(0, i - ROLLING_LONG):i])) if i else 0.5
        rolling_short = rolling_short if i else 0.5
        rolling_long = rolling_long if i else 0.5
        x.append([
            float(order[0]),
            float(order[0] - order[1]),
            _entropy(rf),
            _entropy(ens),
            float(mean_pair_l1),
            float(class_dispersion),
            consensus / 3.0,
            float(np.max(np.abs(rf - ens))),
            float(np.max(np.abs(rf - nonrf))),
            rolling_short,
            rolling_long,
            float(np.mean(np.abs(nonrf - rf))),
        ])
    return np.asarray(x, dtype=float), errors, np.arange(len(rows))


def _adjust(rf: np.ndarray, nonrf: np.ndarray, risk: np.ndarray, mode: str) -> np.ndarray:
    out = []
    for p, nr, r in zip(rf, nonrf, risk):
        r = float(np.clip(r, 0.0, 1.0))
        if mode == "risk_shrink":
            q = (1.0 - r) * p + r * np.full(3, 1.0 / 3.0)
        elif mode == "risk_rescue":
            consensus = np.sum(np.argmax(np.stack([p, nr]), axis=1) == np.argmax(nr))
            if r >= 0.5 and consensus == 2:
                q = 0.2 * p + 0.8 * nr
            else:
                q = (1.0 - r) * p + r * np.full(3, 1.0 / 3.0)
        else:
            raise ValueError(mode)
        q = np.clip(q, EPS, 1.0)
        out.append(q / q.sum())
    return np.asarray(out)


def _metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    yi = np.asarray([CLASSES.index(v) for v in y], dtype=int)
    picked = np.clip(p[np.arange(len(y)), yi], EPS, 1.0)
    one = np.eye(3)[yi]
    conf = p.max(axis=1)
    hit = (p.argmax(axis=1) == yi).astype(float)
    ece = 0.0
    for lo in np.linspace(0.0, 0.9, 10):
        hi = lo + 0.1
        mask = (conf >= lo) & ((conf < hi) if hi < 1.0 else (conf <= hi))
        if np.any(mask):
            ece += float(mask.mean()) * abs(float(hit[mask].mean()) - float(conf[mask].mean()))
    return {
        "n": int(len(y)),
        "accuracy": float(hit.mean()),
        "logloss": float(-np.mean(np.log(picked))),
        "brier": float(np.mean(np.sum((p - one) ** 2, axis=1))),
        "ece": float(ece),
    }


def _bootstrap_delta(y: np.ndarray, base: np.ndarray, cand: np.ndarray, seed: int = 42) -> dict[str, list[float] | float]:
    rng = np.random.default_rng(seed)
    yi = np.asarray([CLASSES.index(v) for v in y], dtype=int)
    a = -np.log(np.clip(cand[np.arange(len(y)), yi], EPS, 1.0)) - -np.log(np.clip(base[np.arange(len(y)), yi], EPS, 1.0))
    obs = float(a.mean())
    vals = np.asarray([rng.choice(a, size=len(a), replace=True).mean() for _ in range(1000)])
    return {"observed": obs, "ci95": [float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))]}


def _load(horizon: str) -> list[dict[str, object]]:
    sources = {k: _rows(ROOT / "data" / "historical_research" / v) for k, v in FILES[horizon].items()}
    common = set.intersection(*(set(v) for v in sources.values()))
    rows = []
    for ts in sorted(common):
        actuals = {sources[k][ts]["actual"] for k in sources}
        if len(actuals) != 1 or next(iter(actuals)) not in CLASSES:
            continue
        row = {"ts": ts, "actual": next(iter(actuals))}
        row.update({k: sources[k][ts]["p"] for k in sources})
        rows.append(row)
    return rows


def _run_meta(rows: list[dict[str, object]], meta_factory) -> tuple[np.ndarray, dict[str, object]]:
    x, errors, _ = _features(rows)
    cut = int(len(rows) * (1.0 - HOLDOUT_FRAC))
    dev, hold = list(range(cut)), list(range(cut, len(rows)))
    risk_dev = np.full(len(dev), 0.5)
    preds_dev = np.zeros((len(dev), 3))
    block_scores = []
    for start in range(META_MIN_TRAIN, len(dev), META_BLOCK):
        train_end = max(META_MIN_TRAIN, start - META_EMBARGO)
        if train_end < META_MIN_TRAIN or start >= len(dev):
            continue
        model = meta_factory()
        model.fit(x[:train_end], errors[:train_end])
        risk = model.predict_proba(x[start:min(start + META_BLOCK, len(dev))])[:, 1]
        e = errors[start:min(start + META_BLOCK, len(dev))]
        risk_dev[start:min(start + META_BLOCK, len(dev))] = risk
        block_scores.append({
            "start": int(start),
            "n": int(len(e)),
            "mean_risk": float(np.mean(risk)),
            "observed_error_rate": float(np.mean(e)),
        })

    selected_dev = np.arange(META_MIN_TRAIN, len(dev))
    if len(selected_dev) == 0:
        raise ValueError("insufficient_meta_development_rows")

    y = np.asarray([rows[i]["actual"] for i in range(len(rows))])
    rf = np.asarray([rows[i]["rf"] for i in range(len(rows))])
    nonrf = np.asarray([(rows[i]["extra"] + rows[i]["hgb"]) / 2.0 for i in range(len(rows))])
    nonrf /= nonrf.sum(axis=1, keepdims=True)

    roc = float(roc_auc_score(errors[selected_dev], risk_dev[selected_dev]))
    ap = float(average_precision_score(errors[selected_dev], risk_dev[selected_dev]))
    shrink_dev = _adjust(rf[selected_dev], nonrf[selected_dev], risk_dev[selected_dev], "risk_shrink")
    rescue_dev = _adjust(rf[selected_dev], nonrf[selected_dev], risk_dev[selected_dev], "risk_rescue")
    base_dev = _metrics(y[selected_dev], rf[selected_dev])

    hold_train_end = max(META_MIN_TRAIN, len(dev) - META_EMBARGO)
    model = meta_factory()
    model.fit(x[:hold_train_end], errors[:hold_train_end])
    risk_hold = model.predict_proba(x[hold])[:, 1]
    base_hold = _metrics(y[hold], rf[hold])
    shrink_hold = _metrics(y[hold], _adjust(rf[hold], nonrf[hold], risk_hold, "risk_shrink"))
    rescue_hold = _metrics(y[hold], _adjust(rf[hold], nonrf[hold], risk_hold, "risk_rescue"))

    high = risk_hold >= np.quantile(risk_hold, 0.80)
    low = risk_hold <= np.quantile(risk_hold, 0.20)
    return risk_hold, {
        "n": len(rows),
        "development_n": len(dev),
        "final_holdout_n": len(hold),
        "development_risk_auc": roc,
        "development_risk_average_precision": ap,
        "development_baseline": base_dev,
        "development_risk_shrink": _metrics(y[selected_dev], shrink_dev),
        "development_risk_rescue": _metrics(y[selected_dev], rescue_dev),
        "development_logloss_delta_ci95": _bootstrap_delta(y[selected_dev], rf[selected_dev], shrink_dev),
        "final_holdout": {
            "baseline": base_hold,
            "risk_shrink": shrink_hold,
            "risk_rescue": rescue_hold,
            "risk_shrink_delta": {
                k: float(shrink_hold[k] - base_hold[k]) for k in ("accuracy", "logloss", "brier", "ece")
            },
            "risk_rescue_delta": {
                k: float(rescue_hold[k] - base_hold[k]) for k in ("accuracy", "logloss", "brier", "ece")
            },
            "high_risk_20pct": _metrics(y[hold][high], rf[hold][high]),
            "low_risk_20pct": _metrics(y[hold][low], rf[hold][low]),
            "high_risk_error_rate": float(np.mean(errors[hold][high])),
            "low_risk_error_rate": float(np.mean(errors[hold][low])),
        },
        "block_scores": block_scores,
        "policy": "predict_then_update; meta-error target uses only prior settled OOS rows; no reverse-direction rule",
    }


def evaluate(horizon: str, meta_name: str, meta_factory) -> dict[str, object]:
    try:
        rows = _load(horizon)
    except FileNotFoundError as exc:
        return {"status": "DEFERRED", "research_only": True, "production_changed": False, "reason": f"missing_oos_file:{exc}", "n": 0}
    if len(rows) < META_MIN_TRAIN + 500:
        return {"status": "DEFERRED", "research_only": True, "production_changed": False, "reason": "insufficient_aligned_oos_rows", "n": len(rows)}
    risk, result = _run_meta(rows, meta_factory)
    result.update({
        "status": "OK",
        "schema_version": 1,
        "research_only": True,
        "production_changed": False,
        "meta_model": meta_name,
        "final_holdout_protected": True,
        "final_holdout_used_for_selection": False,
    })
    return result


def main() -> None:
    factories = {
        "logreg": lambda: LogisticRegression(C=0.5, class_weight="balanced", max_iter=2000),
    }
    payload = {
        "schema_version": 1,
        "research_only": True,
        "production_changed": False,
        "promotion_evidence_eligible": False,
        "horizons": {h: evaluate(h, "logreg", factories["logreg"]) for h in HORIZONS},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
