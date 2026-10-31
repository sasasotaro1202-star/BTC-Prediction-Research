"""Research-only two-memory adaptive blend for BTC 10m.

Combines the frozen production RF (full-history memory) with a rolling recent RF
(short-memory) and updates their mixing weight only from previously settled
block losses. No current outcome affects the current prediction.

Evaluation has three chronological regions:
  1. development: adaptive state is learned prequentially;
  2. adaptive holdout: no tuning/selection, state continues forward;
  3. final blind: one fixed recent model and one fixed blend weight are frozen
     before the blind window and never updated with blind outcomes.

Binance Vision archive timing is not equivalent to live PIT metadata, so this
cannot authorize production promotion.
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from binance_history import binance_archive_rows
from bootstrap_train import make_features
from feature_schema import FEATURES
from label_policy import CLASSES, NEUTRAL_RETURN

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
OUT = ROOT / "data" / "historical_research" / "two_memory_blend_10m_oos.json"

HORIZON_STEPS = 10
HORIZON = "10m"
TARGET_ROWS = 50_000
TRAIN_WINDOW = 8_000
TEST_BLOCK = 500
GAP_BARS = 10
DEV_FRAC = 0.75
ADAPT_HOLDOUT_FRAC = 0.15
BLIND_FRAC = 0.10
MIN_BLOCKS = 8
MIN_DEV_ROWS = 10_000
RF_TREES = 200
EWMA_ALPHA = 0.20
ETA = 3.0
WEIGHT_MIN = 0.15
WEIGHT_MAX = 0.85
EPS = 1e-8


def _dt(value: str) -> datetime:
    return datetime.fromisoformat(str(value).replace("Z", "+00:00"))


def _norm(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    if p.ndim == 1:
        p = p[None, :]
    p = np.clip(p, EPS, 1.0)
    s = p.sum(axis=1, keepdims=True)
    if not np.isfinite(p).all() or np.any(s <= 0):
        raise ValueError("invalid_probability_matrix")
    return p / s


def _align(model, x: np.ndarray) -> np.ndarray:
    raw = np.asarray(model.predict_proba(x), dtype=float)
    out = np.full((len(x), 3), EPS, dtype=float)
    for i, cls in enumerate(model.classes_):
        name = str(cls)
        if name in CLASSES:
            out[:, CLASSES.index(name)] = raw[:, i]
    return _norm(out)


def _metrics(y: list[str], p: np.ndarray) -> dict[str, float | int]:
    yi = np.asarray([CLASSES.index(v) for v in y], dtype=int)
    p = _norm(p)
    pred = p.argmax(axis=1)
    hit = pred == yi
    conf = p.max(axis=1)
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
        "logloss": float(-np.mean(np.log(np.clip(p[np.arange(len(y)), yi], EPS, 1.0)))),
        "brier": float(np.mean(np.sum((p - np.eye(3)[yi]) ** 2, axis=1))),
        "ece": float(ece),
    }


def _rf() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=RF_TREES,
        max_depth=10,
        min_samples_leaf=10,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1,
    )


def _dataset(rows: list[list[float]], trained_at: datetime):
    x, y, ts = [], [], []
    for i in range(30, len(rows) - HORIZON_STEPS):
        created = datetime.fromtimestamp(int(rows[i][0]) / 1000.0, timezone.utc)
        if created <= trained_at:
            continue
        try:
            feat = make_features(rows[i - 29 : i + 1])
        except Exception:
            continue
        if len(feat) != len(FEATURES) or not all(math.isfinite(float(v)) for v in feat):
            continue
        future_return = float(rows[i + HORIZON_STEPS][4]) / float(rows[i][4]) - 1.0
        label = (
            "UP" if future_return > NEUTRAL_RETURN
            else "DOWN" if future_return < -NEUTRAL_RETURN
            else "FLAT"
        )
        x.append(feat)
        y.append(label)
        ts.append(created)
    if not x:
        return np.empty((0, len(FEATURES))), [], []
    return np.asarray(x, dtype=float), y, ts


def _loss(p: np.ndarray, y: list[str]) -> float:
    yi = np.asarray([CLASSES.index(v) for v in y], dtype=int)
    return float(-np.mean(np.log(np.clip(p[np.arange(len(y)), yi], EPS, 1.0))))


def _weight(frozen_loss: float | None, recent_loss: float | None) -> float:
    if frozen_loss is None or recent_loss is None:
        return 0.50
    # positive score => recent memory has lower loss => larger recent weight
    score = ETA * (frozen_loss - recent_loss)
    score = float(np.clip(score, -8.0, 8.0))
    w = 1.0 / (1.0 + math.exp(-score))
    return float(np.clip(w, WEIGHT_MIN, WEIGHT_MAX))


def _blend(frozen: np.ndarray, recent: np.ndarray, recent_weight: float) -> np.ndarray:
    return _norm((1.0 - recent_weight) * frozen + recent_weight * recent)


def _aggregate(blocks: list[dict], key: str) -> dict:
    total = sum(int(b["n"]) for b in blocks)
    result = {"n": total}
    for metric in ("accuracy", "logloss", "brier", "ece"):
        result[metric] = float(
            sum(int(b["n"]) * float(b[key][metric]) for b in blocks) / total
        )
    return result


def _ratio(values, predicate):
    a = np.asarray(values, dtype=float)
    return float(np.mean([bool(predicate(v)) for v in a])) if len(a) else 0.0


def _bootstrap(values: list[float], seed: int = 42, n: int = 2000):
    a = np.asarray(values, dtype=float)
    if len(a) < 2:
        return {"mean": float(a.mean()) if len(a) else float("nan"),
                "low": float("nan"), "high": float("nan")}
    rng = np.random.default_rng(seed)
    sample = a[rng.integers(0, len(a), size=(n, len(a)))]
    means = sample.mean(axis=1)
    return {
        "mean": float(a.mean()),
        "low": float(np.quantile(means, 0.025)),
        "high": float(np.quantile(means, 0.975)),
    }


def evaluate():
    meta = json.loads((MODEL_DIR / f"{HORIZON}.json").read_text(encoding="utf-8"))
    trained_at = _dt(meta["trained_at_utc"])
    rows = binance_archive_rows(TARGET_ROWS)
    x, y, timestamps = _dataset(rows, trained_at)

    if len(y) < MIN_DEV_ROWS + 2000:
        return {
            "status": "DEFERRED",
            "reason": "insufficient_future_rows",
            "n": int(len(y)),
            "research_only": True,
            "production_changed": False,
            "strict_pit": False,
            "promotion_evidence_eligible": False,
        }

    blind_start = int(round(len(y) * DEV_FRAC + len(y) * ADAPT_HOLDOUT_FRAC))
    dev_end = int(round(len(y) * DEV_FRAC))
    adapt_start, adapt_end = dev_end, blind_start
    blind_start = min(blind_start, len(y) - 1)

    if dev_end < MIN_DEV_ROWS or adapt_end <= adapt_start:
        raise ValueError("invalid_chronological_split")

    champion = joblib.load(MODEL_DIR / f"{HORIZON}.joblib")

    def run_prequential(start: int, end: int, frozen_loss, recent_loss, first_test):
        blocks = []
        state = {"frozen_loss": frozen_loss, "recent_loss": recent_loss}
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
            model = _rf()
            model.fit(x[train_start:train_end], train_y)
            recent = _align(model, x[test_start:test_end])
            frozen = _align(champion, x[test_start:test_end])
            w = _weight(state["frozen_loss"], state["recent_loss"])
            blend = _blend(frozen, recent, w)
            yy = y[test_start:test_end]
            bm = _metrics(yy, frozen)
            rm = _metrics(yy, recent)
            cm = _metrics(yy, blend)
            blocks.append({
                "start": timestamps[test_start].isoformat(),
                "end": timestamps[test_end - 1].isoformat(),
                "n": test_end - test_start,
                "recent_weight": w,
                "baseline": bm,
                "recent": rm,
                "candidate": cm,
                "delta": {
                    "accuracy": cm["accuracy"] - bm["accuracy"],
                    "logloss": cm["logloss"] - bm["logloss"],
                    "brier": cm["brier"] - bm["brier"],
                    "ece": cm["ece"] - bm["ece"],
                },
            })
            # Feedback arrives only after the current block is scored.
            state["frozen_loss"] = (
                EWMA_ALPHA * bm["logloss"] + (1.0 - EWMA_ALPHA) * (
                    state["frozen_loss"] if state["frozen_loss"] is not None else bm["logloss"]
                )
            )
            state["recent_loss"] = (
                EWMA_ALPHA * rm["logloss"] + (1.0 - EWMA_ALPHA) * (
                    state["recent_loss"] if state["recent_loss"] is not None else rm["logloss"]
                )
            )
        return blocks, state

    first_dev_test = max(TRAIN_WINDOW + GAP_BARS, 8_500)
    dev_blocks, state = run_prequential(
        0, dev_end, None, None, first_dev_test
    )
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

    # Adaptive holdout continues the state learned on development, but makes
    # no hyperparameter or threshold choice.
    adapt_blocks, state_after_adapt = run_prequential(
        adapt_start, adapt_end, state["frozen_loss"], state["recent_loss"], adapt_start
    )

    # Final blind: freeze one recent RF and one mixing weight before any blind
    # outcome is consumed. The blind window is never used for model updates.
    train_end = blind_start - GAP_BARS
    train_start = max(0, train_end - TRAIN_WINDOW)
    if train_end - train_start < TRAIN_WINDOW:
        raise ValueError("blind_training_window_too_short")
    blind_model = _rf()
    blind_model.fit(x[train_start:train_end], np.asarray(y[train_start:train_end]))
    blind_recent = _align(blind_model, x[blind_start:])
    blind_frozen = _align(champion, x[blind_start:])
    blind_weight = _weight(state_after_adapt["frozen_loss"], state_after_adapt["recent_loss"])
    blind_candidate = _blend(blind_frozen, blind_recent, blind_weight)
    blind_y = y[blind_start:]

    dev_baseline = _aggregate(dev_blocks, "baseline")
    dev_candidate = _aggregate(dev_blocks, "candidate")
    adapt_baseline = _aggregate(adapt_blocks, "baseline") if adapt_blocks else None
    adapt_candidate = _aggregate(adapt_blocks, "candidate") if adapt_blocks else None
    blind_baseline = _metrics(blind_y, blind_frozen)
    blind_candidate_m = _metrics(blind_y, blind_candidate)

    ll_delta = [float(b["delta"]["logloss"]) for b in dev_blocks]
    br_delta = [float(b["delta"]["brier"]) for b in dev_blocks]
    acc_delta = [float(b["delta"]["accuracy"]) for b in dev_blocks]
    ece_delta = [float(b["delta"]["ece"]) for b in dev_blocks]
    ll_rel_gain = (dev_baseline["logloss"] - dev_candidate["logloss"]) / max(abs(dev_baseline["logloss"]), EPS)
    br_rel_gain = (dev_baseline["brier"] - dev_candidate["brier"]) / max(abs(dev_baseline["brier"]), EPS)

    eligible = bool(
        ll_rel_gain >= 0.03
        and br_rel_gain >= 0.01
        and _ratio(ll_delta, lambda v: v <= 0) >= 0.70
        and _ratio(br_delta, lambda v: v <= 0) >= 0.70
        and _ratio(acc_delta, lambda v: v >= -0.005) >= 0.70
        and _ratio(ece_delta, lambda v: v <= 0) >= 0.70
        and blind_candidate_m["accuracy"] >= blind_baseline["accuracy"] - 0.005
        and blind_candidate_m["logloss"] <= blind_baseline["logloss"]
        and blind_candidate_m["brier"] <= blind_baseline["brier"]
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
        "model_version_under_test": "two_memory_ewma_blend",
        "trained_at_utc": meta.get("trained_at_utc"),
        "features": list(FEATURES),
        "config": {
            "train_window": TRAIN_WINDOW,
            "test_block": TEST_BLOCK,
            "gap_bars": GAP_BARS,
            "rf_trees": RF_TREES,
            "ewma_alpha": EWMA_ALPHA,
            "eta": ETA,
            "weight_bounds": [WEIGHT_MIN, WEIGHT_MAX],
            "dev_frac": DEV_FRAC,
            "adaptive_holdout_frac": ADAPT_HOLDOUT_FRAC,
            "blind_frac": BLIND_FRAC,
        },
        "n": int(len(y)),
        "development_n": int(dev_end),
        "adaptive_holdout_n": int(adapt_end - adapt_start),
        "final_blind_n": int(len(blind_y)),
        "development": {
            "blocks": len(dev_blocks),
            "baseline": dev_baseline,
            "candidate": dev_candidate,
            "delta": _block_delta_aggregate(dev_candidate, dev_baseline),
            "block_stability": {
                "improved_logloss_ratio": _ratio(ll_delta, lambda v: v < 0),
                "improved_brier_ratio": _ratio(br_delta, lambda v: v < 0),
                "non_worse_accuracy_ratio": _ratio(acc_delta, lambda v: v >= -0.005),
                "non_worse_ece_ratio": _ratio(ece_delta, lambda v: v <= 0),
                "logloss_delta_bootstrap_ci": _bootstrap(ll_delta),
                "brier_delta_bootstrap_ci": _bootstrap(br_delta),
            },
            "recent_weight_summary": {
                "mean": float(np.mean([b["recent_weight"] for b in dev_blocks])),
                "min": float(np.min([b["recent_weight"] for b in dev_blocks])),
                "max": float(np.max([b["recent_weight"] for b in dev_blocks])),
            },
        },
        "adaptive_holdout": {
            "blocks": len(adapt_blocks),
            "baseline": adapt_baseline,
            "candidate": adapt_candidate,
            "delta": _block_delta_aggregate(adapt_candidate, adapt_baseline) if adapt_candidate else None,
            "used_for_selection": False,
        },
        "final_blind": {
            "protected": True,
            "used_for_selection": False,
            "weight_frozen": float(blind_weight),
            "baseline": blind_baseline,
            "candidate": blind_candidate_m,
            "delta": _block_delta_aggregate(blind_candidate_m, blind_baseline),
        },
        "eligibility": eligible,
    }


def _block_delta_aggregate(candidate: dict, baseline: dict) -> dict[str, float]:
    return {
        "accuracy": float(candidate["accuracy"] - baseline["accuracy"]),
        "logloss": float(candidate["logloss"] - baseline["logloss"]),
        "brier": float(candidate["brier"] - baseline["brier"]),
        "ece": float(candidate["ece"] - baseline["ece"]),
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
