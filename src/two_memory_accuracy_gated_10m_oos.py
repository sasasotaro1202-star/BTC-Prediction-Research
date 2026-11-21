"""Research-only accuracy-gated two-memory BTC 10m blend.

The two-memory candidate improves probability scores but slightly reduced blind
Accuracy in its first evaluation. This challenger keeps the two-memory blend
available only for cases where a meta-model, trained strictly on earlier
settled blocks, predicts a positive case-level Accuracy benefit.

No current-block or blind labels are used before prediction. The gate threshold
is fixed (not tuned on the protected blind set), and production artifacts are
never modified.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from two_memory_blend_10m_oos import (
    ADAPT_HOLDOUT_FRAC,
    BLIND_FRAC,
    DEV_FRAC,
    EPS,
    GAP_BARS,
    HORIZON,
    MODEL_DIR,
    OUT as TWO_MEMORY_OUT,
    RF_TREES,
    TEST_BLOCK,
    TRAIN_WINDOW,
    _align,
    _blend,
    _dataset,
    _dt,
    _metrics,
    _rf,
    _weight,
)
from binance_history import binance_archive_rows
from feature_schema import FEATURES
from label_policy import CLASSES

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "historical_research" / "two_memory_accuracy_gated_10m_oos.json"

MIN_BLOCKS = 8
MIN_DEV_ROWS = 10_000
GATE_THRESHOLD = 0.65
MIN_GATE_ROWS = 800
EPS_METRIC = 1e-12


def _entropy(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), EPS, 1.0)
    return -np.sum(p * np.log(p), axis=1)


def _margin(p: np.ndarray) -> np.ndarray:
    p = np.sort(np.asarray(p, dtype=float), axis=1)
    return p[:, -1] - p[:, -2]


def _gate_features(
    frozen: np.ndarray,
    recent: np.ndarray,
    blend: np.ndarray,
    recent_weight: float,
) -> np.ndarray:
    frozen = np.asarray(frozen, dtype=float)
    recent = np.asarray(recent, dtype=float)
    blend = np.asarray(blend, dtype=float)
    top_f = np.argmax(frozen, axis=1)
    top_r = np.argmax(recent, axis=1)
    top_b = np.argmax(blend, axis=1)
    l1 = np.sum(np.abs(frozen - recent), axis=1)
    l2 = np.sqrt(np.sum((frozen - recent) ** 2, axis=1))
    same_fr = (top_f == top_r).astype(float)
    return np.column_stack(
        [
            frozen,
            recent,
            blend,
            np.max(frozen, axis=1),
            np.max(recent, axis=1),
            np.max(blend, axis=1),
            _margin(frozen),
            _margin(recent),
            _margin(blend),
            _entropy(frozen),
            _entropy(recent),
            _entropy(blend),
            l1,
            l2,
            same_fr,
            (top_f != top_b).astype(float),
            (top_r != top_b).astype(float),
            np.full(len(frozen), float(recent_weight)),
        ]
    )


def _fit_gate(x: list[np.ndarray], y: list[int]):
    if not x or not y or len(y) < MIN_GATE_ROWS:
        return None
    yy = np.asarray(y, dtype=int)
    if len(np.unique(yy)) < 2:
        return None
    xx = np.asarray(x, dtype=float)
    if xx.ndim != 2 or not np.isfinite(xx).all():
        return None
    model = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    C=0.35,
                    class_weight="balanced",
                    max_iter=3000,
                    random_state=42,
                ),
            ),
        ]
    )
    model.fit(xx, yy)
    return model


def _apply_gate(
    gate,
    frozen: np.ndarray,
    recent: np.ndarray,
    blend: np.ndarray,
    recent_weight: float,
):
    feats = _gate_features(frozen, recent, blend, recent_weight)
    disagree = np.argmax(frozen, axis=1) != np.argmax(blend, axis=1)
    if gate is None:
        prob = np.zeros(len(feats), dtype=float)
        use = np.zeros(len(feats), dtype=bool)
    else:
        prob = gate.predict_proba(feats)[:, 1]
        use = disagree & (prob >= GATE_THRESHOLD)
    out = frozen.copy()
    out[use] = blend[use]
    return out, use, prob


def _delta(candidate: dict, baseline: dict) -> dict[str, float]:
    return {
        "accuracy": float(candidate["accuracy"] - baseline["accuracy"]),
        "logloss": float(candidate["logloss"] - baseline["logloss"]),
        "brier": float(candidate["brier"] - baseline["brier"]),
        "ece": float(candidate["ece"] - baseline["ece"]),
    }


def _aggregate(blocks: list[dict], key: str) -> dict:
    if not blocks:
        return {}
    total = sum(int(b["n"]) for b in blocks)
    out = {"n": total}
    for m in ("accuracy", "logloss", "brier", "ece"):
        out[m] = float(
            sum(int(b["n"]) * float(b[key][m]) for b in blocks) / total
        )
    return out


def _ratio(values, predicate) -> float:
    if not values:
        return 0.0
    return float(np.mean([bool(predicate(v)) for v in values]))


def _relative_gain(baseline: float, candidate: float, lower_is_better: bool) -> float:
    base = max(abs(float(baseline)), EPS_METRIC)
    if lower_is_better:
        return float((baseline - candidate) / base)
    return float((candidate - baseline) / base)


def evaluate():
    meta = json.loads((MODEL_DIR / f"{HORIZON}.json").read_text(encoding="utf-8"))
    trained_at = _dt(meta["trained_at_utc"])
    rows = binance_archive_rows(50_000)
    x, y, timestamps = _dataset(rows, trained_at)

    if len(y) < MIN_DEV_ROWS + 2_000:
        return {
            "status": "DEFERRED",
            "reason": "insufficient_future_rows",
            "n": int(len(y)),
            "research_only": True,
            "production_changed": False,
            "strict_pit": False,
            "promotion_evidence_eligible": False,
        }

    dev_end = int(round(len(y) * DEV_FRAC))
    adapt_end = int(round(len(y) * (DEV_FRAC + ADAPT_HOLDOUT_FRAC)))
    blind_start = min(adapt_end, len(y) - 1)
    if dev_end < MIN_DEV_ROWS or blind_start <= dev_end:
        raise ValueError("invalid_chronological_split")

    champion = joblib.load(MODEL_DIR / f"{HORIZON}.joblib")
    if champion is None:
        raise ValueError("missing_champion_model")

    state_frozen = None
    state_recent = None
    gate_x: list[np.ndarray] = []
    gate_y: list[int] = []
    dev_blocks = []

    def process_range(start: int, end: int, allow_updates: bool, first_test: int):
        nonlocal state_frozen, state_recent, gate_x, gate_y
        blocks = []
        for test_start in range(first_test, end, TEST_BLOCK):
            test_end = min(test_start + TEST_BLOCK, end)
            if test_end - test_start < TEST_BLOCK:
                continue
            train_end = test_start - GAP_BARS
            train_start = max(0, train_end - TRAIN_WINDOW)
            if train_end - train_start < TRAIN_WINDOW:
                continue
            train_y = np.asarray(y[train_start:train_end])
            if len(set(train_y)) < 3:
                continue

            recent_model = _rf()
            recent_model.fit(x[train_start:train_end], train_y)
            frozen = _align(champion, x[test_start:test_end])
            recent = _align(recent_model, x[test_start:test_end])

            w = _weight(state_frozen, state_recent)
            raw_blend = _blend(frozen, recent, w)
            gate = _fit_gate(gate_x, gate_y)
            gated, use, gate_prob = _apply_gate(
                gate, frozen, recent, raw_blend, w
            )

            yy = y[test_start:test_end]
            base_m = _metrics(yy, frozen)
            raw_m = _metrics(yy, raw_blend)
            gate_m = _metrics(yy, gated)

            base_pred = np.argmax(frozen, axis=1)
            blend_pred = np.argmax(raw_blend, axis=1)
            yi = np.asarray([CLASSES.index(v) for v in yy], dtype=int)
            gated_pred = np.argmax(gated, axis=1)
            chosen = use & (blend_pred != base_pred)
            benefit = (blend_pred == yi) & (base_pred != yi)
            route_benefit_rate = (
                float(np.mean(benefit[chosen])) if np.any(chosen) else None
            )

            blocks.append(
                {
                    "start": timestamps[test_start].isoformat(),
                    "end": timestamps[test_end - 1].isoformat(),
                    "n": int(test_end - test_start),
                    "recent_weight": float(w),
                    "gate_available": gate is not None,
                    "gate_rate": float(use.mean()),
                    "decision_change_rate": float(chosen.mean()),
                    "routed_benefit_rate": route_benefit_rate,
                    "baseline": base_m,
                    "raw_candidate": raw_m,
                    "gated_candidate": gate_m,
                    "delta": {
                        "raw": _delta(raw_m, base_m),
                        "gated": _delta(gate_m, base_m),
                    },
                }
            )

            if allow_updates:
                gate_feats = _gate_features(frozen, recent, raw_blend, w)
                # Training label is strictly prior by construction: this block
                # is appended only after its predictions have been scored.
                gain_label = (blend_pred == yi) & (base_pred != yi)
                gate_x.extend(gate_feats.tolist())
                gate_y.extend(gain_label.astype(int).tolist())
                if len(gate_y) > 12_000:
                    gate_x = gate_x[-12_000:]
                    gate_y = gate_y[-12_000:]

                base_loss = base_m["logloss"]
                recent_loss = raw_m["logloss"] if raw_m["n"] else base_loss
                state_frozen = (
                    0.20 * base_loss
                    + 0.80 * (state_frozen if state_frozen is not None else base_loss)
                )
                state_recent = (
                    0.20 * recent_loss
                    + 0.80 * (state_recent if state_recent is not None else recent_loss)
                )
        return blocks

    first_dev_test = max(TRAIN_WINDOW + GAP_BARS, 8_500)
    dev_blocks = process_range(dev_end - (dev_end - first_dev_test), dev_end, True, first_dev_test)
    if len(dev_blocks) < MIN_BLOCKS:
        return {
            "status": "DEFERRED",
            "reason": "insufficient_development_blocks",
            "n": int(len(y)),
            "development_blocks": len(dev_blocks),
            "research_only": True,
            "production_changed": False,
            "strict_pit": False,
            "promotion_evidence_eligible": False,
        }

    adapt_blocks = process_range(dev_end, blind_start, True, dev_end)

    # Freeze all gate state, recent RF and blend weight before blind.
    blind_train_end = blind_start - GAP_BARS
    blind_train_start = max(0, blind_train_end - TRAIN_WINDOW)
    if blind_train_end - blind_train_start < TRAIN_WINDOW:
        raise ValueError("blind_training_window_too_short")
    blind_recent_model = _rf()
    blind_recent_model.fit(
        x[blind_train_start:blind_train_end],
        np.asarray(y[blind_train_start:blind_train_end]),
    )
    blind_frozen = _align(champion, x[blind_start:])
    blind_recent = _align(blind_recent_model, x[blind_start:])
    blind_weight = _weight(state_frozen, state_recent)
    blind_raw = _blend(blind_frozen, blind_recent, blind_weight)
    blind_gate = _fit_gate(gate_x, gate_y)
    blind_gated, blind_use, blind_prob = _apply_gate(
        blind_gate, blind_frozen, blind_recent, blind_raw, blind_weight
    )
    blind_y = y[blind_start:]

    dev_base = _aggregate(dev_blocks, "baseline")
    dev_raw = _aggregate(dev_blocks, "raw_candidate")
    dev_gated = _aggregate(dev_blocks, "gated_candidate")
    adapt_base = _aggregate(adapt_blocks, "baseline")
    adapt_gated = _aggregate(adapt_blocks, "gated_candidate")
    blind_base = _metrics(blind_y, blind_frozen)
    blind_raw_m = _metrics(blind_y, blind_raw)
    blind_gated_m = _metrics(blind_y, blind_gated)

    gated_acc_delta = [float(b["delta"]["gated"]["accuracy"]) for b in dev_blocks]
    gated_ll_delta = [float(b["delta"]["gated"]["logloss"]) for b in dev_blocks]
    gated_br_delta = [float(b["delta"]["gated"]["brier"]) for b in dev_blocks]
    gated_ece_delta = [float(b["delta"]["gated"]["ece"]) for b in dev_blocks]
    route_benefits = [
        b["routed_benefit_rate"] for b in dev_blocks
        if b["routed_benefit_rate"] is not None
    ]

    dev_relative = {
        "accuracy": _relative_gain(
            dev_base["accuracy"], dev_gated["accuracy"], False
        ),
        "brier": _relative_gain(
            dev_base["brier"], dev_gated["brier"], True
        ),
        "logloss": _relative_gain(
            dev_base["logloss"], dev_gated["logloss"], True
        ),
    }

    blind_relative = {
        "accuracy": _relative_gain(
            blind_base["accuracy"], blind_gated_m["accuracy"], False
        ),
        "brier": _relative_gain(
            blind_base["brier"], blind_gated_m["brier"], True
        ),
        "logloss": _relative_gain(
            blind_base["logloss"], blind_gated_m["logloss"], True
        ),
    }

    eligibility = bool(
        dev_relative["accuracy"] >= 0.03
        and dev_relative["brier"] >= 0.01
        and dev_relative["logloss"] >= 0.03
        and _ratio(gated_acc_delta, lambda v: v >= -0.005) >= 0.70
        and _ratio(gated_br_delta, lambda v: v <= 0.0) >= 0.70
        and _ratio(gated_ll_delta, lambda v: v <= 0.0) >= 0.70
        and _ratio(gated_ece_delta, lambda v: v <= 0.0) >= 0.70
        and blind_gated_m["accuracy"] >= blind_base["accuracy"]
        and blind_gated_m["brier"] <= blind_base["brier"]
        and blind_gated_m["logloss"] <= blind_base["logloss"]
    )

    return {
        "status": "OK",
        "schema_version": 1,
        "research_only": True,
        "production_changed": False,
        "strict_pit": False,
        "promotion_evidence_eligible": False,
        "archive_publication_time_unknown": True,
        "horizon": HORIZON,
        "production_model_version": meta.get("model_version"),
        "model_version_under_test": "two_memory_accuracy_gated",
        "trained_at_utc": meta.get("trained_at_utc"),
        "features": list(FEATURES),
        "config": {
            "train_window": TRAIN_WINDOW,
            "test_block": TEST_BLOCK,
            "gap_bars": GAP_BARS,
            "rf_trees": RF_TREES,
            "ewma_alpha": 0.20,
            "eta": 3.0,
            "gate_threshold": GATE_THRESHOLD,
            "gate_min_rows": MIN_GATE_ROWS,
            "dev_frac": DEV_FRAC,
            "adaptive_holdout_frac": ADAPT_HOLDOUT_FRAC,
            "blind_frac": BLIND_FRAC,
        },
        "n": int(len(y)),
        "development_n": int(dev_end),
        "adaptive_holdout_n": int(blind_start - dev_end),
        "final_blind_n": int(len(blind_y)),
        "development": {
            "blocks": len(dev_blocks),
            "baseline": dev_base,
            "raw_two_memory": dev_raw,
            "gated_candidate": dev_gated,
            "delta": _delta(dev_gated, dev_base),
            "relative_improvement": dev_relative,
            "stability": {
                "non_worse_accuracy_ratio": _ratio(
                    gated_acc_delta, lambda v: v >= -0.005
                ),
                "improved_brier_ratio": _ratio(
                    gated_br_delta, lambda v: v < 0.0
                ),
                "improved_logloss_ratio": _ratio(
                    gated_ll_delta, lambda v: v < 0.0
                ),
                "improved_ece_ratio": _ratio(
                    gated_ece_delta, lambda v: v < 0.0
                ),
            },
            "mean_routed_benefit_rate": (
                float(np.mean(route_benefits)) if route_benefits else None
            ),
        },
        "adaptive_holdout": {
            "blocks": len(adapt_blocks),
            "baseline": adapt_base,
            "gated_candidate": adapt_gated,
            "delta": (
                _delta(adapt_gated, adapt_base) if adapt_gated else None
            ),
            "used_for_selection": False,
        },
        "final_blind": {
            "protected": True,
            "used_for_selection": False,
            "gate_frozen": True,
            "gate_available": blind_gate is not None,
            "gate_rate": float(blind_use.mean()),
            "weight_frozen": float(blind_weight),
            "baseline": blind_base,
            "raw_two_memory": blind_raw_m,
            "gated_candidate": blind_gated_m,
            "delta_raw": _delta(blind_raw_m, blind_base),
            "delta_gated": _delta(blind_gated_m, blind_base),
            "relative_improvement": blind_relative,
            "mean_gate_probability": float(np.mean(blind_prob)) if len(blind_prob) else 0.0,
        },
        "eligibility": eligibility,
        "source_two_memory_artifact": str(TWO_MEMORY_OUT),
    }


def main():
    payload = {
        "schema_version": 1,
        "research_only": True,
        "production_changed": False,
        "strict_pit": False,
        "promotion_evidence_eligible": False,
        "horizons": {HORIZON: evaluate()},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
