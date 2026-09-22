"""Research-only risk-aware dynamic ensemble under concept drift.

Weights are updated only from outcomes of completed prior OOS blocks. The
current block's labels are never used to choose its weights. A bounded softmax,
weight floor, and distress shrinkage keep the router from overreacting.
Production models and artifacts are never modified.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from model_compare import HORIZONS, load_primary_production_strict_rows, load_archive_research_rows, metrics
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier

OUT = Path(__file__).resolve().parents[1] / "data" / "historical_research" / "risk_aware_dynamic_oos.json"
MIN_TRAIN = 2000
TEST_BLOCK = 150
MAX_BLOCKS = 12
MAX_ROWS = 7000
PURGE = {"5m": 5, "10m": 10}
EMBARGO = {"5m": 60, "10m": 60}
FLOOR = 0.10
MAX_WEIGHT = 0.55
SHRINKAGE = 0.20
TEMPERATURE = 0.25
DIVERSITY_PENALTY = 0.05


def _aligned(model, rows):
    X = np.asarray([r["x"] for r in rows], dtype=float)
    raw = np.asarray(model.predict_proba(X), dtype=float)
    out = np.full((len(rows), 3), 1e-7, dtype=float)
    idx = {"DOWN": 0, "FLAT": 1, "UP": 2}
    for j, cls in enumerate(model.classes_):
        if str(cls) in idx:
            out[:, idx[str(cls)]] = raw[:, j]
    out = np.clip(out, 1e-7, 1.0)
    return out / out.sum(axis=1, keepdims=True)


def _weights_from_losses(
    losses,
    briers=None,
    eces=None,
    redundancy=None,
    temperature=TEMPERATURE,
    floor=FLOOR,
    max_weight=MAX_WEIGHT,
):
    """Convert only prior completed-block diagnostics into bounded weights."""
    if losses is None or len(losses) == 0:
        return ()
    ll = np.asarray([float(x) for x in losses], dtype=float)
    n = len(ll)
    uniform = np.full(n, 1.0 / n)
    if n == 0 or not np.all(np.isfinite(ll)):
        return tuple(float(x) for x in uniform)

    def ratio(values):
        if values is None:
            return None
        arr = np.asarray([float(x) for x in values], dtype=float)
        if arr.shape != ll.shape or not np.all(np.isfinite(arr)):
            return None
        arr = np.maximum(arr, 0.0)
        med = float(np.median(arr))
        return np.ones(n, dtype=float) if med <= 1e-12 else arr / med

    llr = ratio(ll)
    if llr is None:
        return tuple(float(x) for x in uniform)
    score = 0.65 * llr

    if briers is not None:
        br = ratio(briers)
        if br is None:
            return tuple(float(x) for x in uniform)
        score += 0.20 * br

    if eces is not None:
        ec = ratio(eces)
        if ec is None:
            return tuple(float(x) for x in uniform)
        score += 0.15 * ec

    if redundancy is not None:
        red = np.asarray([float(x) for x in redundancy], dtype=float)
        if red.shape != ll.shape or not np.all(np.isfinite(red)):
            return tuple(float(x) for x in uniform)
        score += DIVERSITY_PENALTY * np.clip(red, 0.0, 1.0)

    lower = float(floor)
    upper = float(max_weight)
    if n * lower > 1.0 or n * upper < 1.0:
        return tuple(float(x) for x in uniform)

    z = max(float(temperature), 1e-6)
    shifted = score - float(score.min())
    raw = np.exp(-shifted / z)
    raw /= raw.sum()
    w = (1.0 - SHRINKAGE) * raw + SHRINKAGE * uniform
    w = np.clip(w, lower, upper)

    for _ in range(20):
        diff = 1.0 - float(w.sum())
        if abs(diff) <= 1e-12:
            break
        if diff > 0:
            room = upper - w
            active = room > 1e-12
            if not np.any(active):
                break
            w[active] += diff * room[active] / room[active].sum()
        else:
            room = w - lower
            active = room > 1e-12
            if not np.any(active):
                break
            w[active] += diff * room[active] / room[active].sum()
    w = np.clip(w, lower, upper)
    w /= w.sum()
    return tuple(float(x) for x in w)


def _pairwise_redundancy(parts):
    """Return each model's mean absolute probability correlation to peers."""
    if len(parts) < 2:
        return np.zeros(len(parts), dtype=float)
    vectors = [np.asarray(p, dtype=float).reshape(-1) for p in parts]
    out = np.zeros(len(vectors), dtype=float)
    for i in range(len(vectors)):
        pair_values = []
        for j in range(len(vectors)):
            if i == j:
                continue
            a, c = vectors[i], vectors[j]
            if len(a) < 2 or len(c) != len(a):
                return np.ones(len(vectors), dtype=float)
            if float(np.std(a)) <= 1e-12 or float(np.std(c)) <= 1e-12:
                pair_values.append(1.0)
                continue
            corr = float(np.corrcoef(a, c)[0, 1])
            if not np.isfinite(corr):
                return np.ones(len(vectors), dtype=float)
            pair_values.append(abs(corr))
        out[i] = float(np.mean(pair_values)) if pair_values else 0.0
    return np.clip(out, 0.0, 1.0)


def _factories():
    return (
        lambda: Pipeline([
            ("scale", StandardScaler()),
            ("model", LogisticRegression(C=0.25, max_iter=3000)),
        ]),
        lambda: ExtraTreesClassifier(
            n_estimators=240, max_depth=14, min_samples_leaf=12,
            max_features="sqrt", random_state=42, n_jobs=-1,
        ),
        lambda: HistGradientBoostingClassifier(
            max_iter=180, max_leaf_nodes=15, learning_rate=0.04,
            l2_regularization=1.5, random_state=42,
        ),
        lambda: LGBMClassifier(
            n_estimators=220, num_leaves=15, learning_rate=0.04,
            min_child_samples=20, reg_lambda=2.0, random_state=42,
            n_jobs=-1, verbosity=-1,
        ),
    )


def _fit_parts(train, test):
    factories = _factories()
    X = np.asarray([r["x"] for r in train], dtype=float)
    y = np.asarray([r["y"] for r in train])
    models = []
    for factory in factories:
        model = factory()
        model.fit(X, y)
        models.append(model)
    return [_aligned(model, test) for model in models]


def _mix_parts(parts, weights):
    mix = np.zeros((len(parts[0]), 3), dtype=float)
    for w, p in zip(weights, parts):
        mix += float(w) * p
    mix = np.clip(mix, 1e-7, 1.0)
    return mix / mix.sum(axis=1, keepdims=True)


def _block_endpoints(n):
    raw = list(range(MIN_TRAIN, n, TEST_BLOCK))
    if len(raw) <= MAX_BLOCKS:
        return raw
    return sorted(set(int(x) for x in np.linspace(raw[0], raw[-1], MAX_BLOCKS)))


def _block_logloss(y, p):
    names = ("DOWN", "FLAT", "UP")
    yi = np.asarray([names.index(str(v)) for v in y], dtype=int)
    p = np.asarray(p, dtype=float)
    return float(-np.mean(np.log(np.clip(p[np.arange(len(yi)), yi], 1e-12, 1.0))))


def evaluate(horizon: str):
    rows = load_primary_production_strict_rows(horizon)
    data_source = "live_binance_primary"
    if len(rows) < 2100:
        archive = load_archive_research_rows(horizon, MAX_ROWS)
        if len(archive) > len(rows):
            rows = archive
            data_source = "binance_vision_archive"
    if len(rows) > MAX_ROWS:
        rows = rows[-MAX_ROWS:]
    if len(rows) < MIN_TRAIN + TEST_BLOCK + 100:
        return {"status": "DEFERRED", "n": len(rows), "reason": "insufficient_rows"}

    split = int(len(rows) * 0.80)
    development = rows[:split]
    holdout = rows[split:]
    if len(development) < MIN_TRAIN + TEST_BLOCK or len(holdout) < 100:
        return {"status": "DEFERRED", "n": len(rows), "reason": "insufficient_development_or_holdout"}

    blocks = []
    ema_losses = None
    ema_briers = None
    ema_eces = None
    ema_redundancy = None
    alpha = 0.30

    for end in _block_endpoints(len(development)):
        train_end = max(0, end - PURGE[horizon] - EMBARGO[horizon])
        train = development[:train_end]
        test = development[end:min(end + TEST_BLOCK, len(development))]
        if len(train) < MIN_TRAIN or len(test) < 50:
            continue

        parts = _fit_parts(train, test)
        if len(parts) < 3:
            continue
        equal_w = tuple(1.0 / len(parts) for _ in parts)
        risk_w = (
            _weights_from_losses(
                ema_losses,
                briers=ema_briers,
                eces=ema_eces,
                redundancy=ema_redundancy,
            )
            if ema_losses is not None
            else equal_w
        )
        eq = _mix_parts(parts, equal_w)
        risk = _mix_parts(parts, risk_w)
        y = [r["y"] for r in test]
        em = metrics(y, eq)
        rm = metrics(y, risk)

        component_metrics = [metrics(y, p) for p in parts]
        component_losses = [m["logloss"] for m in component_metrics]
        component_briers = [m["brier"] for m in component_metrics]
        component_eces = [m["calibration_error"] for m in component_metrics]
        component_redundancy = _pairwise_redundancy(parts)
        ema_losses = (
            np.asarray(component_losses, dtype=float)
            if ema_losses is None
            else alpha * np.asarray(component_losses, dtype=float)
            + (1.0 - alpha) * ema_losses
        )
        ema_briers = (
            np.asarray(component_briers, dtype=float)
            if ema_briers is None
            else alpha * np.asarray(component_briers, dtype=float)
            + (1.0 - alpha) * ema_briers
        )
        ema_eces = (
            np.asarray(component_eces, dtype=float)
            if ema_eces is None
            else alpha * np.asarray(component_eces, dtype=float)
            + (1.0 - alpha) * ema_eces
        )
        ema_redundancy = (
            np.asarray(component_redundancy, dtype=float)
            if ema_redundancy is None
            else alpha * np.asarray(component_redundancy, dtype=float)
            + (1.0 - alpha) * ema_redundancy
        )

        blocks.append({
            "n": len(y),
            "equal": em,
            "risk_aware": rm,
            "weights_before_block": list(risk_w),
            "component_logloss": component_losses,
            "component_brier": component_briers,
            "component_ece": component_eces,
            "component_redundancy": component_redundancy.tolist(),
            "redundancy_before_block": (
                None if ema_redundancy is None else ema_redundancy.tolist()
            ),
            "delta": {
                "accuracy": rm["accuracy"] - em["accuracy"],
                "logloss": rm["logloss"] - em["logloss"],
                "brier": rm["brier"] - em["brier"],
            },
        })

    if not blocks:
        return {"status": "DEFERRED", "n": len(rows), "reason": "no_valid_development_blocks"}

    ll = np.asarray([b["delta"]["logloss"] for b in blocks], dtype=float)
    br = np.asarray([b["delta"]["brier"] for b in blocks], dtype=float)
    ac = np.asarray([b["delta"]["accuracy"] for b in blocks], dtype=float)
    summary = {
        "blocks": len(blocks),
        "samples": int(sum(b["n"] for b in blocks)),
        "mean_accuracy_delta": float(ac.mean()),
        "mean_logloss_delta": float(ll.mean()),
        "mean_brier_delta": float(br.mean()),
        "improved_logloss_ratio": float(np.mean(ll < 0)),
        "improved_brier_ratio": float(np.mean(br < 0)),
        "non_worse_accuracy_ratio": float(np.mean(ac >= -0.005)),
        "mean_redundancy": float(
            np.mean([
                np.mean(b["component_redundancy"]) for b in blocks
            ])
        ),
    }

    hold_parts = _fit_parts(development, holdout)
    equal_hold_w = tuple(1.0 / len(hold_parts) for _ in hold_parts)
    equal_hold = _mix_parts(hold_parts, equal_hold_w)
    risk_hold_w = (
        _weights_from_losses(
            ema_losses,
            briers=ema_briers,
            eces=ema_eces,
            redundancy=ema_redundancy,
        )
        if ema_losses is not None
        else equal_hold_w
    )
    risk_hold = _mix_parts(hold_parts, risk_hold_w)
    yh = [r["y"] for r in holdout]

    return {
        "status": "OK",
        "schema_version": 1,
        "research_only": True,
        "production_changed": False,
        "final_holdout_protected": True,
        "data_source": data_source,
        "promotion_evidence_eligible": data_source == "live_binance_primary",
        "final_holdout_used_for_selection": False,
        "policy": "causal_blockwise_ema_logloss_brier_ece_plus_redundancy_softmax_with_bounded_weights_and_uniform_shrinkage",
        "summary": summary,
        "development_n": len(development),
        "final_holdout_n": len(holdout),
        "final_holdout": {
            "equal": metrics(yh, equal_hold),
            "risk_aware": metrics(yh, risk_hold),
            "weights": list(risk_hold_w),
        },
        "blocks": blocks,
    }


def main():
    result = {
        "schema_version": 1,
        "research_only": True,
        "production_changed": False,
        "horizons": {h: evaluate(h) for h in HORIZONS},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
