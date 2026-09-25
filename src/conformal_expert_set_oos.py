"""Research-only conformal expert-set routing for nonstationary BTC direction.

A split-conformal reliability set is built only from an earlier calibration
block. Current-block outcomes never affect current routing. Production is never
mutated or promoted.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from model_compare import CLASSES, PURGE_BARS, EMBARGO_BARS, load_archive_research_rows, metrics

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "historical_research" / "conformal_expert_set_oos.json"
HORIZONS = ("5m", "10m")
EXPERTS = ("logreg", "extra_trees", "hgb", "soft_equal")
MIN_TRAIN = 3000
CAL_BLOCK = 600
TEST_BLOCK = 600
FINAL_HOLDOUT_FRAC = 0.20
MAX_ROWS = 12000
ALPHA = 0.10
EPS = 1e-7

def _align(model, rows):
    X = np.asarray([r["x"] for r in rows], dtype=float)
    raw = np.asarray(model.predict_proba(X), dtype=float)
    out = np.full((len(rows), 3), EPS, dtype=float)
    idx = {c: i for i, c in enumerate(CLASSES)}
    for j, cls in enumerate(model.classes_):
        if str(cls) in idx:
            out[:, idx[str(cls)]] = raw[:, j]
    out = np.clip(out, EPS, 1.0)
    return out / out.sum(axis=1, keepdims=True)

def _factories():
    return {
        "logreg": lambda: Pipeline([("scale", StandardScaler()), ("model", LogisticRegression(C=0.3, max_iter=2500))]),
        "extra_trees": lambda: ExtraTreesClassifier(n_estimators=180, max_depth=10, min_samples_leaf=15, max_features="sqrt", random_state=42, n_jobs=-1),
        "hgb": lambda: HistGradientBoostingClassifier(max_iter=180, max_leaf_nodes=15, learning_rate=0.04, l2_regularization=1.5, random_state=42),
    }

def _fit_experts(train):
    if len(train) < MIN_TRAIN or len(set(r["y"] for r in train)) < 3:
        raise ValueError("insufficient_or_single_class_training_rows")
    X = np.asarray([r["x"] for r in train], dtype=float)
    y = np.asarray([r["y"] for r in train], dtype=str)
    return {name: factory().fit(X, y) for name, factory in _factories().items()}

def _expert_probs(models, rows):
    parts = {name: _align(model, rows) for name, model in models.items()}
    parts["soft_equal"] = np.mean(np.stack([parts["logreg"], parts["extra_trees"], parts["hgb"]], axis=0), axis=0)
    return parts

def _class_scores(probs, rows):
    y = np.asarray([CLASSES.index(r["y"]) for r in rows], dtype=int)
    return 1.0 - np.clip(probs[np.arange(len(rows)), y], EPS, 1.0), y

def conformal_pvalues(cal_probs, cal_rows, test_probs):
    scores, y = _class_scores(cal_probs, cal_rows)
    by_class = {c: scores[y == c] for c in range(3)}
    by_class["global"] = scores
    result = np.zeros(len(test_probs), dtype=float)
    for i, p in enumerate(np.asarray(test_probs, dtype=float)):
        cls = int(np.argmax(p))
        ref = by_class.get(cls)
        if ref is None or len(ref) < 25:
            ref = by_class["global"]
        score = 1.0 - float(np.clip(p[cls], EPS, 1.0))
        result[i] = (1.0 + float(np.sum(ref >= score))) / (len(ref) + 1.0)
    return result

def route_block(cal_parts, cal_rows, test_parts):
    active = np.zeros((len(next(iter(test_parts.values()))), len(EXPERTS)), dtype=bool)
    pvalues = np.zeros_like(active, dtype=float)
    for j, expert in enumerate(EXPERTS):
        pv = conformal_pvalues(cal_parts[expert], cal_rows, test_parts[expert])
        pvalues[:, j] = pv
        active[:, j] = pv >= ALPHA
    out = np.zeros((active.shape[0], 3), dtype=float)
    sizes = active.sum(axis=1)
    for i in range(len(out)):
        names = [EXPERTS[j] for j in range(len(EXPERTS)) if active[i, j]]
        if not names:
            names = ["soft_equal"]
        out[i] = np.mean([test_parts[name][i] for name in names], axis=0)
    out = np.clip(out, EPS, 1.0)
    out /= out.sum(axis=1, keepdims=True)
    return out, sizes, pvalues

def _stats(y, p):
    m = metrics(y, p)
    return {"n": int(len(y)), "accuracy": float(m["accuracy"]), "logloss": float(m["logloss"]),
            "brier": float(m["brier"]), "ece": float(m.get("ece", m.get("calibration_error", np.nan)))}

def evaluate(horizon):
    rows = load_archive_research_rows(horizon, MAX_ROWS)
    if len(rows) < MIN_TRAIN + CAL_BLOCK + TEST_BLOCK:
        return {"status": "DEFERRED", "research_only": True, "production_changed": False,
                "n": len(rows), "reason": "insufficient_archive_rows"}
    split = int(len(rows) * (1.0 - FINAL_HOLDOUT_FRAC))
    dev, holdout = rows[:split], rows[split:]
    gap = int(PURGE_BARS[horizon] + EMBARGO_BARS[horizon])
    blocks = []
    for test_start in range(MIN_TRAIN + CAL_BLOCK + gap, len(dev), TEST_BLOCK):
        test_end = min(test_start + TEST_BLOCK, len(dev))
        test = dev[test_start:test_end]
        cal_end = test_start - gap
        cal = dev[cal_end - CAL_BLOCK:cal_end]
        train = dev[:cal_end - CAL_BLOCK]
        if len(test) < TEST_BLOCK // 2 or len(cal) < CAL_BLOCK or len(train) < MIN_TRAIN:
            continue
        try:
            models = _fit_experts(train)
            cal_parts = _expert_probs(models, cal)
            test_parts = _expert_probs(models, test)
            routed, sizes, pvals = route_block(cal_parts, cal, test_parts)
        except Exception as exc:
            continue
        y = [r["y"] for r in test]
        base = test_parts["soft_equal"]
        bm, cm = _stats(y, base), _stats(y, routed)
        blocks.append({
            "n": len(test),
            "baseline": bm,
            "candidate": cm,
            "delta": {k: cm[k] - bm[k] for k in ("accuracy", "logloss", "brier", "ece")},
            "mean_set_size": float(sizes.mean()),
            "single_expert_ratio": float(np.mean(sizes == 1)),
            "no_valid_expert_ratio": float(np.mean(sizes == 0)),
            "mean_pvalue": float(pvals.mean()),
        })
    if len(blocks) < 8:
        return {"status": "DEFERRED", "research_only": True, "production_changed": False,
                "n": len(rows), "reason": "too_few_valid_blocks", "blocks": len(blocks)}
    total = float(sum(b["n"] for b in blocks))
    def agg(side, metric_name):
        return sum(b["n"] * b[side][metric_name] for b in blocks) / total
    base = {m: agg("baseline", m) for m in ("accuracy", "logloss", "brier", "ece")}
    cand = {m: agg("candidate", m) for m in ("accuracy", "logloss", "brier", "ece")}
    ll = np.asarray([b["delta"]["logloss"] for b in blocks], dtype=float)
    br = np.asarray([b["delta"]["brier"] for b in blocks], dtype=float)
    ac = np.asarray([b["delta"]["accuracy"] for b in blocks], dtype=float)

    final_models = _fit_experts(dev)
    final_cal = dev[-CAL_BLOCK:]
    final_train = dev[:-CAL_BLOCK]
    final_cal_parts = _expert_probs(final_models, final_cal)
    final_hold_parts = _expert_probs(final_models, holdout)
    final_routed, final_sizes, _ = route_block(final_cal_parts, final_cal, final_hold_parts)
    hold_y = [r["y"] for r in holdout]
    hold_base = _stats(hold_y, final_hold_parts["soft_equal"])
    hold_cand = _stats(hold_y, final_routed)

    return {
        "status": "OK", "schema_version": 1, "research_only": True, "production_changed": False,
        "final_holdout_protected": True, "final_holdout_used_for_selection": False,
        "strict_point_in_time_archive_replay": False, "n": len(rows),
        "development_n": len(dev), "final_holdout_n": len(holdout), "alpha": ALPHA, "gap_bars": gap,
        "blocks": len(blocks),
        "aggregate": {"baseline": base, "candidate": cand,
                      "delta": {m: cand[m] - base[m] for m in base}},
        "stability": {
            "improved_logloss_ratio": float(np.mean(ll < 0)),
            "improved_brier_ratio": float(np.mean(br < 0)),
            "non_worse_accuracy_ratio": float(np.mean(ac >= -0.005)),
            "mean_set_size": float(np.mean([b["mean_set_size"] for b in blocks])),
            "single_expert_ratio": float(np.mean([b["single_expert_ratio"] for b in blocks])),
        },
        "final_holdout": {
            "baseline": hold_base, "candidate": hold_cand,
            "delta": {m: hold_cand[m] - hold_base[m] for m in base},
        },
        "eligibility": bool(
            (base["logloss"] - cand["logloss"]) / max(abs(base["logloss"]), EPS) >= 0.03
            and (base["brier"] - cand["brier"]) / max(abs(base["brier"]), EPS) >= 0.01
            and cand["accuracy"] >= base["accuracy"] - 0.005
            and float(np.mean(ac >= -0.005)) >= 0.70
            and hold_cand["logloss"] <= hold_base["logloss"]
            and hold_cand["brier"] <= hold_base["brier"]
            and hold_cand["accuracy"] >= hold_base["accuracy"] - 0.005
        ),
        "blocks_detail": blocks,
    }

def main():
    payload = {"schema_version": 1, "research_only": True, "production_changed": False,
               "horizons": {h: evaluate(h) for h in HORIZONS}}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))

if __name__ == "__main__":
    main()
