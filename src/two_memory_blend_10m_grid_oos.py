"""Research-only nested two-memory adaptive blend for BTC 10m.

One frozen production RF is the long-memory expert. A fixed-window rolling RF is
the short-memory expert. On each chronological block, the short-memory weight
is updated from *previously settled block losses only*. Candidate response speed
(ETA) is selected from a small preregistered grid using development OOS only.

Evaluation:
- development: nested chronological OOS used for ETA selection;
- adaptive holdout: no tuning, selected ETA carried forward;
- final blind: one recent RF and one weight are frozen before the blind period.

No production artifact/model is changed. Binance Vision archive publication
timing is not equivalent to live PIT metadata, so promotion remains prohibited.
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss

from binance_history import binance_archive_rows
from bootstrap_train import make_features
from feature_schema import FEATURES
from label_policy import CLASSES, NEUTRAL_RETURN

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
OUT = ROOT / "data" / "historical_research" / "two_memory_blend_10m_grid_oos.json"

HORIZON = "10m"
STEPS = 10
TARGET_ROWS = 50_000
TRAIN_WINDOW = 8_000
TEST_BLOCK = 500
GAP_BARS = 10

DEV_FRAC = 0.75
ADAPT_HOLDOUT_FRAC = 0.15
BLIND_FRAC = 0.10

ETA_GRID = (0.0, 1.0, 3.0, 6.0, 10.0)
EWMA_ALPHA = 0.20
WEIGHT_MIN = 0.15
WEIGHT_MAX = 0.85
RF_TREES = 200
MIN_DEV_ROWS = 10_000
MIN_BLOCKS = 8
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
        "logloss": float(log_loss(yi, p, labels=[0, 1, 2])),
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
    for i in range(30, len(rows) - STEPS):
        created = datetime.fromtimestamp(int(rows[i][0]) / 1000.0, timezone.utc)
        if created <= trained_at:
            continue
        try:
            feat = make_features(rows[i - 29 : i + 1])
        except Exception:
            continue
        if len(feat) != len(FEATURES) or not all(math.isfinite(float(v)) for v in feat):
            continue
        future_return = float(rows[i + STEPS][4]) / float(rows[i][4]) - 1.0
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


def _weight(frozen_loss: float | None, recent_loss: float | None, eta: float) -> float:
    if frozen_loss is None or recent_loss is None or eta <= 0.0:
        return 0.50
    score = float(np.clip(eta * (frozen_loss - recent_loss), -8.0, 8.0))
    w = 1.0 / (1.0 + math.exp(-score))
    return float(np.clip(w, WEIGHT_MIN, WEIGHT_MAX))


def _blend(frozen: np.ndarray, recent: np.ndarray, weight: float) -> np.ndarray:
    return _norm((1.0 - weight) * frozen + weight * recent)


def _loss(p: np.ndarray, y: list[str]) -> float:
    yi = np.asarray([CLASSES.index(v) for v in y], dtype=int)
    return float(-np.mean(np.log(np.clip(p[np.arange(len(y)), yi], EPS, 1.0))))


def _bootstrap(values: list[float], seed: int = 42, n: int = 2000) -> dict[str, float]:
    a = np.asarray(values, dtype=float)
    if len(a) < 2:
        return {
            "mean": float(a.mean()) if len(a) else float("nan"),
            "low": float("nan"),
            "high": float("nan"),
        }
    rng = np.random.default_rng(seed)
    sample = a[rng.integers(0, len(a), size=(n, len(a)))]
    means = sample.mean(axis=1)
    return {
        "mean": float(a.mean()),
        "low": float(np.quantile(means, 0.025)),
        "high": float(np.quantile(means, 0.975)),
    }


def _generate_blocks(
    x: np.ndarray,
    y: list[str],
    ts: list[datetime],
    champion,
    start: int,
    end: int,
) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    first = max(start, TRAIN_WINDOW + GAP_BARS)
    for test_start in range(first, end, TEST_BLOCK):
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

        blocks.append({
            "test_start": int(test_start),
            "test_end": int(test_end),
            "timestamp_start": ts[test_start].isoformat(),
            "timestamp_end": ts[test_end - 1].isoformat(),
            "n": int(test_end - test_start),
            "y": y[test_start:test_end],
            "frozen": frozen,
            "recent": recent,
        })
    return blocks


def _simulate(
    blocks: list[dict[str, Any]],
    eta: float,
    initial_state: tuple[float | None, float | None] = (None, None),
) -> tuple[dict[str, Any], tuple[float | None, float | None]]:
    frozen_loss, recent_loss = initial_state
    results = []
    for block in blocks:
        weight = _weight(frozen_loss, recent_loss, eta)
        candidate = _blend(block["frozen"], block["recent"], weight)
        base_m = _metrics(block["y"], block["frozen"])
        recent_m = _metrics(block["y"], block["recent"])
        cand_m = _metrics(block["y"], candidate)

        results.append({
            "start": block["timestamp_start"],
            "end": block["timestamp_end"],
            "n": block["n"],
            "recent_weight": weight,
            "baseline": base_m,
            "recent": recent_m,
            "candidate": cand_m,
            "delta": {
                "accuracy": cand_m["accuracy"] - base_m["accuracy"],
                "logloss": cand_m["logloss"] - base_m["logloss"],
                "brier": cand_m["brier"] - base_m["brier"],
                "ece": cand_m["ece"] - base_m["ece"],
            },
        })

        frozen_loss = (
            EWMA_ALPHA * base_m["logloss"]
            + (1.0 - EWMA_ALPHA) * (
                frozen_loss if frozen_loss is not None else base_m["logloss"]
            )
        )
        recent_loss = (
            EWMA_ALPHA * recent_m["logloss"]
            + (1.0 - EWMA_ALPHA) * (
                recent_loss if recent_loss is not None else recent_m["logloss"]
            )
        )

    if not results:
        raise ValueError("no_blocks")

    total = sum(r["n"] for r in results)
    aggregate = {"n": total}
    for name in ("accuracy", "logloss", "brier", "ece"):
        aggregate[name] = float(
            sum(r["n"] * r["candidate"][name] for r in results) / total
        )
    baseline = {"n": total}
    for name in ("accuracy", "logloss", "brier", "ece"):
        baseline[name] = float(
            sum(r["n"] * r["baseline"][name] for r in results) / total
        )

    return {
        "blocks": len(results),
        "baseline": baseline,
        "candidate": aggregate,
        "delta": {
            name: float(aggregate[name] - baseline[name])
            for name in ("accuracy", "logloss", "brier", "ece")
        },
        "block_results": results,
    }, (frozen_loss, recent_loss)


def evaluate():
    meta = json.loads((MODEL_DIR / "10m.json").read_text(encoding="utf-8"))
    trained_at = _dt(meta["trained_at_utc"])
    rows = binance_archive_rows(TARGET_ROWS)
    x, y, ts = _dataset(rows, trained_at)

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

    n = len(y)
    dev_end = int(round(n * DEV_FRAC))
    adapt_end = int(round(n * (DEV_FRAC + ADAPT_HOLDOUT_FRAC)))
    blind_start = min(adapt_end, n - 1)

    champion = joblib.load(MODEL_DIR / "10m.joblib")

    # Development models are trained once per block and replayed across the
    # ETA grid, so hyperparameter selection does not multiply model fitting cost.
    dev_blocks = _generate_blocks(x, y, ts, champion, 0, dev_end)
    if len(dev_blocks) < MIN_BLOCKS:
        return {
            "status": "DEFERRED",
            "reason": "insufficient_development_blocks",
            "n": int(n),
            "development_blocks": len(dev_blocks),
            "research_only": True,
            "production_changed": False,
            "strict_pit": False,
            "promotion_evidence_eligible": False,
        }

    eta_trials: dict[str, Any] = {}
    for eta in ETA_GRID:
        trial, state = _simulate(dev_blocks, eta)
        eta_trials[str(eta)] = {
            "eta": eta,
            "result": trial,
            "final_state": {
                "frozen_loss": state[0],
                "recent_loss": state[1],
            },
        }

    # Selection uses development OOS only; ties resolve to the smaller ETA.
    selected_eta = min(
        ETA_GRID,
        key=lambda eta: (
            float(eta_trials[str(eta)]["result"]["candidate"]["logloss"]),
            float(eta_trials[str(eta)]["result"]["candidate"]["brier"]),
            float(eta),
        ),
    )
    selected_dev = eta_trials[str(selected_eta)]["result"]
    dev_state = (
        eta_trials[str(selected_eta)]["final_state"]["frozen_loss"],
        eta_trials[str(selected_eta)]["final_state"]["recent_loss"],
    )

    # Adaptive holdout: selected ETA is frozen; only its prequential state is
    # carried forward, with no additional model/parameter selection.
    adapt_blocks = _generate_blocks(x, y, ts, champion, adapt_end - max(0, adapt_end - n), adapt_end)
    # The generator's minimum first index is train_window+gap. For this holdout,
    # explicitly request from adapt_start so each block has a causal training set.
    adapt_start = dev_end
    adapt_blocks = _generate_blocks(x, y, ts, champion, adapt_start, adapt_end)
    adapt_result, adapt_state = _simulate(adapt_blocks, selected_eta, dev_state) if adapt_blocks else ({}, dev_state)

    # Final blind: model and weight frozen before any blind outcome is used.
    train_end = blind_start - GAP_BARS
    train_start = max(0, train_end - TRAIN_WINDOW)
    if train_end - train_start < TRAIN_WINDOW or len(set(y[train_start:train_end])) < 3:
        raise ValueError("blind_training_window_invalid")
    blind_model = _rf()
    blind_model.fit(x[train_start:train_end], np.asarray(y[train_start:train_end]))
    blind_recent = _align(blind_model, x[blind_start:])
    blind_frozen = _align(champion, x[blind_start:])
    blind_weight = _weight(adapt_state[0], adapt_state[1], selected_eta)
    blind_candidate = _blend(blind_frozen, blind_recent, blind_weight)
    blind_y = y[blind_start:]

    blind_baseline = _metrics(blind_y, blind_frozen)
    blind_candidate_m = _metrics(blind_y, blind_candidate)

    dev_ll_deltas = [float(b["delta"]["logloss"]) for b in selected_dev["block_results"]]
    dev_br_deltas = [float(b["delta"]["brier"]) for b in selected_dev["block_results"]]
    dev_acc_deltas = [float(b["delta"]["accuracy"]) for b in selected_dev["block_results"]]
    dev_ece_deltas = [float(b["delta"]["ece"]) for b in selected_dev["block_results"]]

    ll_rel_gain = (
        selected_dev["baseline"]["logloss"] - selected_dev["candidate"]["logloss"]
    ) / max(abs(selected_dev["baseline"]["logloss"]), EPS)
    br_rel_gain = (
        selected_dev["baseline"]["brier"] - selected_dev["candidate"]["brier"]
    ) / max(abs(selected_dev["baseline"]["brier"]), EPS)

    eligible = bool(
        ll_rel_gain >= 0.03
        and br_rel_gain >= 0.01
        and float(np.mean(np.asarray(dev_ll_deltas) < 0)) >= 0.70
        and float(np.mean(np.asarray(dev_br_deltas) < 0)) >= 0.70
        and float(np.mean(np.asarray(dev_acc_deltas) >= -0.005)) >= 0.70
        and float(np.mean(np.asarray(dev_ece_deltas) <= 0)) >= 0.70
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
        "model_version_under_test": "two_memory_ewma_eta_grid",
        "trained_at_utc": meta.get("trained_at_utc"),
        "features": list(FEATURES),
        "config": {
            "train_window": TRAIN_WINDOW,
            "test_block": TEST_BLOCK,
            "gap_bars": GAP_BARS,
            "rf_trees": RF_TREES,
            "ewma_alpha": EWMA_ALPHA,
            "eta_grid": list(ETA_GRID),
            "weight_bounds": [WEIGHT_MIN, WEIGHT_MAX],
            "dev_frac": DEV_FRAC,
            "adaptive_holdout_frac": ADAPT_HOLDOUT_FRAC,
            "blind_frac": BLIND_FRAC,
        },
        "n": int(n),
        "development_n": int(dev_end),
        "adaptive_holdout_n": int(adapt_end - adapt_start),
        "final_blind_n": int(len(blind_y)),
        "eta_selection": {
            "selected_eta": float(selected_eta),
            "selected_from": "development_oos_only",
            "trials": {
                k: {
                    "eta": v["eta"],
                    "blocks": v["result"]["blocks"],
                    "logloss": v["result"]["candidate"]["logloss"],
                    "brier": v["result"]["candidate"]["brier"],
                    "accuracy": v["result"]["candidate"]["accuracy"],
                }
                for k, v in eta_trials.items()
            },
        },
        "development": {
            "blocks": selected_dev["blocks"],
            "baseline": selected_dev["baseline"],
            "candidate": selected_dev["candidate"],
            "delta": selected_dev["delta"],
            "block_stability": {
                "improved_logloss_ratio": float(np.mean(np.asarray(dev_ll_deltas) < 0)),
                "improved_brier_ratio": float(np.mean(np.asarray(dev_br_deltas) < 0)),
                "non_worse_accuracy_ratio": float(np.mean(np.asarray(dev_acc_deltas) >= -0.005)),
                "non_worse_ece_ratio": float(np.mean(np.asarray(dev_ece_deltas) <= 0)),
                "logloss_delta_bootstrap_ci": _bootstrap(dev_ll_deltas),
                "brier_delta_bootstrap_ci": _bootstrap(dev_br_deltas),
            },
            "mean_recent_weight": float(np.mean([b["recent_weight"] for b in selected_dev["block_results"])),
            "min_recent_weight": float(np.min([b["recent_weight"] for b in selected_dev["block_results"])),
            "max_recent_weight": float(np.max([b["recent_weight"] for b in selected_dev["block_results"])),
        },
        "adaptive_holdout": {
            "blocks": int(adapt_result.get("blocks", 0)),
            "baseline": adapt_result.get("baseline"),
            "candidate": adapt_result.get("candidate"),
            "delta": adapt_result.get("delta"),
            "used_for_selection": False,
        },
        "final_blind": {
            "protected": True,
            "used_for_selection": False,
            "weight_frozen": float(blind_weight),
            "baseline": blind_baseline,
            "candidate": blind_candidate_m,
            "delta": {
                name: float(blind_candidate_m[name] - blind_baseline[name])
                for name in ("accuracy", "logloss", "brier", "ece")
            },
        },
        "eligibility": eligible,
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
