"""Research-only case-conditional gain-aware routing for BTC direction.

For each alternative expert, predict its per-case log-loss advantage over the
current production ensemble using only features available before the case.
Routing is activated only when the predicted advantage clears a conservative
threshold selected on a strictly earlier validation block.

No production artifacts are modified. Final holdout is descriptive only.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "data" / "historical_research"
OUT = RESEARCH / "gain_aware_routing_oos.json"

CLASSES = ("DOWN", "FLAT", "UP")
SOURCE_EXPERTS = ("logreg", "extra", "rf", "hgb", "ensemble")
ALTERNATIVES = ("logreg", "extra", "hgb", "ensemble")
EXPERTS = ("production",) + ALTERNATIVES
HORIZONS = ("5m", "10m")

GAP_BARS = {"5m": 65, "10m": 70}
META_BLOCK = 700
TUNE_BLOCK = 700
TEST_BLOCK = 600
FINAL_HOLDOUT_FRAC = 0.20
MIN_ROWS = 6000
MIN_TEST = 200
EPS = 1e-8

# Conservative, pre-registered search space.
THRESHOLDS = (0.000, 0.003, 0.005, 0.010, 0.020, 0.030)
ALPHAS = (0.15, 0.25, 0.40, 0.60, 1.00)


def _norm(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    if p.ndim == 1:
        p = p[None, :]
    p = np.clip(p, EPS, 1.0)
    s = p.sum(axis=1, keepdims=True)
    if not np.isfinite(p).all() or np.any(s <= 0):
        raise ValueError("invalid_probability_matrix")
    return p / s


def _metrics(y: list[str], p: np.ndarray) -> dict[str, float | int]:
    yi = np.asarray([CLASSES.index(v) for v in y], dtype=int)
    p = _norm(p)
    hit = np.argmax(p, axis=1) == yi
    conf = p.max(axis=1)
    logloss = float(-np.mean(np.log(np.clip(p[np.arange(len(yi)), yi], EPS, 1.0))))
    brier = float(np.mean(np.sum((p - np.eye(3)[yi]) ** 2, axis=1)))
    ece = 0.0
    for k in range(10):
        lo, hi = k / 10.0, (k + 1) / 10.0
        mask = (conf >= lo) & ((conf < hi) if hi < 1.0 else (conf <= hi))
        if np.any(mask):
            ece += float(mask.mean()) * abs(
                float(hit[mask].mean()) - float(conf[mask].mean())
            )
    return {
        "n": int(len(y)),
        "accuracy": float(hit.mean()),
        "logloss": logloss,
        "brier": brier,
        "ece": float(ece),
    }


def _read(path: Path) -> dict[int, tuple[str, np.ndarray]]:
    if not path.is_file():
        raise FileNotFoundError(str(path))
    out: dict[int, tuple[str, np.ndarray]] = {}
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        required = {"timestamp", "actual", "p_down", "p_flat", "p_up"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError(f"schema_missing:{path.name}")
        for row in reader:
            ts = int(row["timestamp"])
            y = row["actual"]
            p = np.asarray(
                [float(row["p_down"]), float(row["p_flat"]), float(row["p_up"])],
                dtype=float,
            )
            if y not in CLASSES:
                continue
            if not np.isfinite(p).all() or np.any(p < 0) or float(p.sum()) <= 0:
                continue
            if ts in out:
                raise ValueError(f"duplicate_timestamp:{path.name}:{ts}")
            out[ts] = (y, _norm(p)[0])
    return out


def load_horizon(horizon: str) -> tuple[list[int], list[str], dict[str, np.ndarray]]:
    data = {
        name: _read(RESEARCH / f"oos_{horizon}_{name}.csv")
        for name in SOURCE_EXPERTS
    }
    common = sorted(set.intersection(*(set(v) for v in data.values())))
    if len(common) < MIN_ROWS:
        raise ValueError(f"insufficient_common_oos_rows:{horizon}:{len(common)}")
    ys: list[str] = []
    probs: dict[str, list[np.ndarray]] = {name: [] for name in EXPERTS}
    for ts in common:
        labels = {data[name][ts][0] for name in SOURCE_EXPERTS}
        if len(labels) != 1:
            raise ValueError(f"label_mismatch:{horizon}:{ts}")
        ys.append(next(iter(labels)))
        probs["production"].append(data["rf"][ts][1])
        for name in ALTERNATIVES:
            probs[name].append(data[name][ts][1])
    arrays = {name: np.asarray(value, dtype=float) for name, value in probs.items()}
    if not np.all(np.diff(np.asarray(common, dtype=np.int64)) > 0):
        raise ValueError(f"timestamps_not_strictly_increasing:{horizon}")
    return common, ys, arrays


def causal_state_features(
    y: list[str], probs: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    """Build row-wise features from prior settled outcomes only."""
    n = len(y)
    states = {name: np.full(n, 1.05, dtype=float) for name in EXPERTS}
    prev_gain = {
        name: np.zeros(n, dtype=float) for name in ALTERNATIVES
    }
    alpha = 0.08
    state = {name: 1.05 for name in EXPERTS}
    gain_state = {name: 0.0 for name in ALTERNATIVES}
    for i in range(n):
        for name in EXPERTS:
            states[name][i] = state[name]
        for name in ALTERNATIVES:
            prev_gain[name][i] = gain_state[name]
        yi = CLASSES.index(y[i])
        prod_loss = -math.log(max(float(probs["production"][i, yi]), EPS))
        for name in EXPERTS:
            loss = -math.log(max(float(probs[name][i, yi]), EPS))
            state[name] = (1.0 - alpha) * state[name] + alpha * loss
        for name in ALTERNATIVES:
            gain = prod_loss - (
                -math.log(max(float(probs[name][i, yi]), EPS))
            )
            gain_state[name] = (1.0 - alpha) * gain_state[name] + alpha * gain
    return {
        **{f"loss_{name}": values for name, values in states.items()},
        **{f"gain_{name}": values for name, values in prev_gain.items()},
    }


def build_features(
    probs: dict[str, np.ndarray],
    causal: dict[str, np.ndarray],
) -> np.ndarray:
    prod = _norm(probs["production"])
    expert_stack = np.stack([_norm(probs[name]) for name in EXPERTS], axis=0)

    pieces = [prod]
    for name in ALTERNATIVES:
        p = _norm(probs[name])
        pieces.append(p)
        pieces.append(p - prod)

    entropy = -np.sum(prod * np.log(np.clip(prod, EPS, 1.0)), axis=1)
    ordered = np.sort(prod, axis=1)[:, ::-1]
    margin = ordered[:, 0] - ordered[:, 1]
    disagreement = np.mean(
        np.sum((expert_stack - prod[None, :, :]) ** 2, axis=2), axis=0
    )
    vote_disagreement = 1.0 - np.mean(
        np.argmax(expert_stack, axis=2) == np.argmax(prod, axis=1)[None, :],
        axis=0,
    )

    pieces.extend(
        [
            entropy[:, None],
            margin[:, None],
            disagreement[:, None],
            vote_disagreement[:, None],
        ]
    )

    causal_matrix = np.column_stack(
        [causal[k] for k in sorted(causal.keys())]
    )
    pieces.append(causal_matrix)
    out = np.column_stack(pieces).astype(float)
    if not np.isfinite(out).all():
        raise ValueError("feature_matrix_nonfinite")
    return out


def _fit_gain_model(x: np.ndarray, target: np.ndarray) -> HistGradientBoostingRegressor:
    y = np.clip(np.asarray(target, dtype=float), -2.0, 2.0)
    model = HistGradientBoostingRegressor(
        loss="huber",
        max_iter=180,
        max_leaf_nodes=15,
        learning_rate=0.04,
        min_samples_leaf=40,
        l2_regularization=3.0,
        random_state=42,
    )
    model.fit(x, y)
    return model


def _predict_gains(
    models: dict[str, HistGradientBoostingRegressor],
    x: np.ndarray,
) -> np.ndarray:
    out = []
    for name in ALTERNATIVES:
        pred = np.asarray(models[name].predict(x), dtype=float)
        if not np.isfinite(pred).all():
            raise ValueError(f"predicted_gain_nonfinite:{name}")
        out.append(np.clip(pred, -2.0, 2.0))
    return np.column_stack(out)


def _route(
    production: np.ndarray,
    alternatives: dict[str, np.ndarray],
    gains: np.ndarray,
    threshold: float,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    names = list(ALTERNATIVES)
    best = np.argmax(gains, axis=1)
    best_gain = gains[np.arange(len(gains)), best]
    choose = best_gain >= float(threshold)
    out = production.copy()
    chosen = np.full(len(gains), "production", dtype=object)
    for i in np.flatnonzero(choose):
        expert = names[int(best[i])]
        out[i] = (1.0 - alpha) * production[i] + alpha * alternatives[expert][i]
        chosen[i] = expert
    return _norm(out), choose


def _choose_policy(
    y: list[str],
    production: np.ndarray,
    alternatives: dict[str, np.ndarray],
    gains: np.ndarray,
) -> tuple[float, float, dict[str, Any]]:
    best: tuple[tuple[float, float, float], float, float, dict[str, Any]] | None = None
    scored = []
    for threshold in THRESHOLDS:
        for alpha in ALPHAS:
            candidate, choose = _route(
                production, alternatives, gains, threshold, alpha
            )
            m = _metrics(y, candidate)
            p = m["logloss"]
            coverage = float(choose.mean())
            # Prefer lower logloss; among near-ties prefer less intervention.
            key = (float(p), float(-m["accuracy"]), coverage)
            item = {
                "threshold": threshold,
                "alpha": alpha,
                "logloss": float(p),
                "brier": float(m["brier"]),
                "accuracy": float(m["accuracy"]),
                "coverage": coverage,
            }
            scored.append(item)
            if best is None or key < best[0]:
                best = (key, threshold, alpha, item)
    if best is None:
        raise RuntimeError("policy_search_empty")
    return float(best[1]), float(best[2]), {"selected": best[3], "grid": scored}


def _aggregate(metrics_list: list[dict[str, Any]]) -> dict[str, float | int]:
    total = sum(int(m["n"]) for m in metrics_list)
    if total <= 0:
        raise ValueError("empty_aggregate")
    keys = ("accuracy", "logloss", "brier", "ece")
    return {
        key: float(
            sum(int(m["n"]) * float(m[key]) for m in metrics_list) / total
        )
        for key in keys
    } | {"n": total}


def evaluate(horizon: str) -> dict[str, Any]:
    timestamps, y, probs = load_horizon(horizon)
    causal = causal_state_features(y, probs)
    features = build_features(probs, causal)
    n = len(y)
    holdout_start = int(round(n * (1.0 - FINAL_HOLDOUT_FRAC)))
    if holdout_start < MIN_ROWS or n - holdout_start < 400:
        return {"status": "DEFERRED", "reason": "insufficient_rows", "n": n}

    blocks: list[dict[str, Any]] = []
    for test_start in range(
        META_BLOCK + TUNE_BLOCK + GAP_BARS[horizon],
        holdout_start,
        TEST_BLOCK,
    ):
        test_end = min(test_start + TEST_BLOCK, holdout_start)
        if test_end - test_start < MIN_TEST:
            continue

        meta_end = test_start - GAP_BARS[horizon] - TUNE_BLOCK
        meta_start = meta_end - META_BLOCK
        tune_start = test_start - GAP_BARS[horizon] - TUNE_BLOCK
        tune_end = tune_start + TUNE_BLOCK
        meta_idx = np.arange(meta_start, meta_end, dtype=int)
        tune_idx = np.arange(tune_start, tune_end, dtype=int)
        test_idx = np.arange(test_start, test_end, dtype=int)
        if len(meta_idx) < META_BLOCK or len(tune_idx) < TUNE_BLOCK:
            continue

        models: dict[str, HistGradientBoostingRegressor] = {}
        meta_x = features[meta_idx]
        tune_x = features[tune_idx]
        test_x = features[test_idx]
        yi_meta = np.asarray([CLASSES.index(v) for v in y], dtype=int)[meta_idx]

        prod_meta = probs["production"][meta_idx]
        prod_loss = -np.log(np.clip(prod_meta[np.arange(len(meta_idx)), yi_meta], EPS, 1.0))

        for j, name in enumerate(ALTERNATIVES):
            p = probs[name][meta_idx]
            alt_loss = -np.log(np.clip(p[np.arange(len(meta_idx)), yi_meta], EPS, 1.0))
            target = np.clip(prod_loss - alt_loss, -2.0, 2.0)
            models[name] = _fit_gain_model(meta_x, target)

        tune_gains = _predict_gains(models, tune_x)
        tune_alt = {name: probs[name][tune_idx] for name in ALTERNATIVES}
        tune_prod = probs["production"][tune_idx]
        tune_y = [y[i] for i in tune_idx]
        threshold, alpha, policy = _choose_policy(
            tune_y, tune_prod, tune_alt, tune_gains
        )

        test_gains = _predict_gains(models, test_x)
        test_alt = {name: probs[name][test_idx] for name in ALTERNATIVES}
        test_prod = probs["production"][test_idx]
        candidate, choose = _route(
            test_prod, test_alt, test_gains, threshold, alpha
        )
        test_y = [y[i] for i in test_idx]
        bm = _metrics(test_y, test_prod)
        cm = _metrics(test_y, candidate)

        selected_names = np.argmax(test_gains, axis=1)
        chosen_name = np.full(len(test_idx), "production", dtype=object)
        chosen_name[choose] = np.asarray(ALTERNATIVES, dtype=object)[selected_names[choose]]
        blocks.append(
            {
                "start_index": int(test_start),
                "end_index": int(test_end),
                "start_timestamp": int(timestamps[test_start]),
                "end_timestamp": int(timestamps[test_end - 1]),
                "n": len(test_idx),
                "threshold": float(threshold),
                "alpha": float(alpha),
                "coverage": float(choose.mean()),
                "chosen_expert_rate": {
                    name: float(np.mean(chosen_name == name))
                    for name in ("production",) + ALTERNATIVES
                },
                "baseline": bm,
                "candidate": cm,
                "delta": {
                    key: float(cm[key] - bm[key])
                    for key in ("accuracy", "logloss", "brier", "ece")
                },
                "tuning": policy["selected"],
            }
        )

    if len(blocks) < 8:
        return {
            "status": "DEFERRED",
            "reason": "insufficient_oos_blocks",
            "n": n,
            "blocks": len(blocks),
        }

    base_agg = _aggregate([b["baseline"] for b in blocks])
    cand_agg = _aggregate([b["candidate"] for b in blocks])
    ll_d = np.asarray([b["delta"]["logloss"] for b in blocks], dtype=float)
    br_d = np.asarray([b["delta"]["brier"] for b in blocks], dtype=float)
    ac_d = np.asarray([b["delta"]["accuracy"] for b in blocks], dtype=float)
    ece_d = np.asarray([b["delta"]["ece"] for b in blocks], dtype=float)

    ll_rel = (base_agg["logloss"] - cand_agg["logloss"]) / max(
        abs(base_agg["logloss"]), EPS
    )
    br_rel = (base_agg["brier"] - cand_agg["brier"]) / max(
        abs(base_agg["brier"]), EPS
    )

    eligible = bool(
        ll_rel >= 0.03
        and br_rel >= 0.01
        and float(np.mean(ll_d <= 0.0)) >= 0.70
        and float(np.mean(br_d <= 0.0)) >= 0.70
        and float(np.mean(ac_d >= -0.005)) >= 0.70
        and float(np.mean(ece_d <= 0.0)) >= 0.70
    )

    # Final holdout is never used to tune. Refit on development only.
    dev_idx = np.arange(holdout_start, dtype=int)
    final_train_end = holdout_start - TUNE_BLOCK - GAP_BARS[horizon]
    final_meta_idx = np.arange(0, final_train_end, dtype=int)
    final_tune_idx = np.arange(
        final_train_end, holdout_start - GAP_BARS[horizon], dtype=int
    )
    final_holdout_idx = np.arange(holdout_start, n, dtype=int)
    if len(final_meta_idx) < META_BLOCK or len(final_tune_idx) < TUNE_BLOCK:
        return {
            "status": "DEFERRED",
            "reason": "insufficient_final_development_split",
            "n": n,
            "blocks": len(blocks),
        }

    final_models: dict[str, HistGradientBoostingRegressor] = {}
    yi_final_meta = np.asarray([CLASSES.index(v) for v in y], dtype=int)[final_meta_idx]
    prod_final_meta = probs["production"][final_meta_idx]
    prod_final_loss = -np.log(
        np.clip(prod_final_meta[np.arange(len(final_meta_idx)), yi_final_meta], EPS, 1.0)
    )
    for name in ALTERNATIVES:
        alt_meta = probs[name][final_meta_idx]
        alt_loss = -np.log(
            np.clip(alt_meta[np.arange(len(final_meta_idx)), yi_final_meta], EPS, 1.0)
        )
        final_models[name] = _fit_gain_model(
            features[final_meta_idx], np.clip(prod_final_loss - alt_loss, -2.0, 2.0)
        )

    final_tune_gains = _predict_gains(final_models, features[final_tune_idx])
    final_threshold, final_alpha, final_policy = _choose_policy(
        [y[i] for i in final_tune_idx],
        probs["production"][final_tune_idx],
        {name: probs[name][final_tune_idx] for name in ALTERNATIVES},
        final_tune_gains,
    )
    hold_gains = _predict_gains(final_models, features[final_holdout_idx])
    hold_candidate, hold_choose = _route(
        probs["production"][final_holdout_idx],
        {name: probs[name][final_holdout_idx] for name in ALTERNATIVES},
        hold_gains,
        final_threshold,
        final_alpha,
    )
    hold_y = [y[i] for i in final_holdout_idx]
    hold_base = _metrics(hold_y, probs["production"][final_holdout_idx])
    hold_cand = _metrics(hold_y, hold_candidate)
    return {
        "status": "OK",
        "schema_version": 1,
        "research_only": True,
        "production_changed": False,
        "strict_common_oos_csv": True,
        "chronological": True,
        "final_holdout_protected": True,
        "final_holdout_used_for_selection": False,
        "horizon": horizon,
        "n": n,
        "development_n": len(dev_idx),
        "final_holdout_n": len(final_holdout_idx),
        "development": {
            "blocks": len(blocks),
            "baseline": base_agg,
            "candidate": cand_agg,
            "delta": {
                key: float(cand_agg[key] - base_agg[key])
                for key in ("accuracy", "logloss", "brier", "ece")
            },
            "relative_improvement": {
                "logloss": float(ll_rel),
                "brier": float(br_rel),
                "accuracy": float(
                    (cand_agg["accuracy"] - base_agg["accuracy"])
                    / max(abs(base_agg["accuracy"]), EPS)
                ),
            },
            "stability": {
                "non_worse_logloss_ratio": float(np.mean(ll_d <= 0.0)),
                "non_worse_brier_ratio": float(np.mean(br_d <= 0.0)),
                "non_worse_accuracy_ratio": float(np.mean(ac_d >= -0.005)),
                "non_worse_ece_ratio": float(np.mean(ece_d <= 0.0)),
            },
            "eligible": eligible,
        },
        "final_holdout": {
            "baseline": hold_base,
            "candidate": hold_cand,
            "delta": {
                key: float(hold_cand[key] - hold_base[key])
                for key in ("accuracy", "logloss", "brier", "ece")
            },
            "threshold": float(final_threshold),
            "alpha": float(final_alpha),
            "coverage": float(hold_choose.mean()),
            "policy_validation": final_policy["selected"],
        },
        "blocks_detail": blocks,
    }


def main() -> None:
    payload = {
        "schema_version": 1,
        "research_only": True,
        "production_changed": False,
        "horizons": {h: evaluate(h) for h in HORIZONS},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
