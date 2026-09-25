"""Research-only case-level gate for the 5m historical situation ensemble.

The situation ensemble showed a positive protected 5m holdout delta but weak
development-average accuracy. This challenger keeps production probabilities as
the default and applies the situation ensemble only when a strictly-prior
meta-model predicts a case-level accuracy rescue (candidate correct, baseline
wrong). The gate is fixed at >=0.65 and never trained on current or blind labels.
Archive publication timing is non-strict PIT, so promotion remains disabled.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from historical_situation_meta_oos import (
    _build_rows,
    _causal_train,
    _fit_predict,
    _metrics,
    _parse_utc,
    _vector,
)
from model_compare import CLASSES

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "historical_research" / "situation_benefit_gate_5m_oos.json"

HORIZON = "5m"
TEST_BLOCK = 500
FINAL_HOLDOUT_FRAC = 0.20
MIN_TRAIN = 3000
MIN_BLOCKS = 8
MAX_ROWS = 20000
MIN_GATE_ROWS = 800
GATE_THRESHOLD = 0.65
EPS = 1e-7


def _entropy(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), EPS, 1.0)
    return -np.sum(p * np.log(p), axis=1)


def _margin(p: np.ndarray) -> np.ndarray:
    q = np.sort(np.asarray(p, dtype=float), axis=1)
    return q[:, -1] - q[:, -2]


def _features(
    rows: list[dict[str, Any]],
    baseline: np.ndarray,
    candidate: np.ndarray,
) -> np.ndarray:
    base = np.asarray(baseline, dtype=float)
    cand = np.asarray(candidate, dtype=float)
    disagreement = np.sum(np.abs(base - cand), axis=1)
    top_diff = (
        np.argmax(base, axis=1) != np.argmax(cand, axis=1)
    ).astype(float)
    x = np.column_stack(
        [
            np.stack([_vector(r) for r in rows]),
            cand,
            _entropy(base),
            _entropy(cand),
            _margin(base),
            _margin(cand),
            disagreement,
            top_diff,
        ]
    )
    if not np.isfinite(x).all():
        raise ValueError("gate_feature_nonfinite")
    return x


def _fit_gate(x: list[np.ndarray], y: list[int]):
    if len(y) < MIN_GATE_ROWS:
        return None
    yy = np.asarray(y, dtype=int)
    if len(np.unique(yy)) < 2:
        return None
    xx = np.vstack(x)
    if len(xx) != len(yy) or not np.isfinite(xx).all():
        raise ValueError("invalid_gate_training_data")
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


def _apply(
    gate,
    rows: list[dict[str, Any]],
    baseline: np.ndarray,
    candidate: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = _features(rows, baseline, candidate)
    if gate is None:
        prob = np.zeros(len(rows), dtype=float)
        use = np.zeros(len(rows), dtype=bool)
    else:
        prob = gate.predict_proba(x)[:, 1]
        use = (
            (np.argmax(baseline, axis=1) != np.argmax(candidate, axis=1))
            & (prob >= GATE_THRESHOLD)
        )
    out = np.asarray(baseline, dtype=float).copy()
    out[use] = candidate[use]
    return out, use, prob


def _delta(a: dict[str, Any], b: dict[str, Any]) -> dict[str, float]:
    return {
        k: float(a[k] - b[k])
        for k in ("accuracy", "logloss", "brier")
    }


def evaluate() -> dict[str, Any]:
    rows = _build_rows(MAX_ROWS)
    if len(rows) < MIN_TRAIN + TEST_BLOCK * MIN_BLOCKS + 100:
        return {
            "status": "DEFERRED",
            "reason": "insufficient_archive_rows",
            "n": len(rows),
            "research_only": True,
            "production_changed": False,
            "promotion_evidence_eligible": False,
        }

    split = int(len(rows) * (1.0 - FINAL_HOLDOUT_FRAC))
    development = rows[:split]
    holdout = rows[split:]
    gate_x: list[np.ndarray] = []
    gate_y: list[int] = []
    blocks: list[dict[str, Any]] = []

    for end in range(MIN_TRAIN, len(development), TEST_BLOCK):
        test = development[end : min(end + TEST_BLOCK, len(development))]
        if len(test) < TEST_BLOCK:
            continue
        train = _causal_train(development[:end], test[0]["created"], HORIZON)
        if len(train) < MIN_TRAIN:
            continue

        candidate = _fit_predict(train, test)
        baseline = np.asarray(
            [[r["p5"][c] for c in CLASSES] for r in test],
            dtype=float,
        )
        baseline /= baseline.sum(axis=1, keepdims=True)
        gate = _fit_gate(gate_x, gate_y)
        gated, used, gate_prob = _apply(gate, test, baseline, candidate)

        bm = _metrics(test, baseline)
        rm = _metrics(test, candidate)
        gm = _metrics(test, gated)

        blocks.append(
            {
                "n": len(test),
                "baseline": bm,
                "raw_candidate": rm,
                "candidate": gm,
                "delta": _delta(gm, bm),
                "raw_delta": _delta(rm, bm),
                "coverage": float(used.mean()),
                "gate_training_rows": len(gate_y),
                "gate_positive_rate": float(np.mean(gate_y)) if gate_y else 0.0,
                "gate_probability_mean": float(np.mean(gate_prob)),
            }
        )

        # Current labels become available only after the block is scored.
        cand_label = np.argmax(candidate, axis=1)
        base_label = np.argmax(baseline, axis=1)
        true_label = np.asarray([CLASSES.index(r["y"]) for r in test], dtype=int)
        rescue = (cand_label == true_label) & (base_label != true_label)
        gate_x.append(_features(test, baseline, candidate))
        gate_y.extend(rescue.astype(int).tolist())

    if len(blocks) < MIN_BLOCKS:
        return {
            "status": "DEFERRED",
            "reason": "insufficient_valid_oos_blocks",
            "n": len(rows),
            "blocks": len(blocks),
            "research_only": True,
            "production_changed": False,
            "promotion_evidence_eligible": False,
        }

    final_train = _causal_train(development, holdout[0]["created"], HORIZON)
    final_candidate = _fit_predict(final_train, holdout)
    final_baseline = np.asarray(
        [[r["p5"][c] for c in CLASSES] for r in holdout],
        dtype=float,
    )
    final_baseline /= final_baseline.sum(axis=1, keepdims=True)
    final_gate = _fit_gate(gate_x, gate_y)
    final_gated, final_used, final_prob = _apply(
        final_gate, holdout, final_baseline, final_candidate
    )

    hold_base = _metrics(holdout, final_baseline)
    hold_raw = _metrics(holdout, final_candidate)
    hold_gate = _metrics(holdout, final_gated)

    ll = np.asarray([b["delta"]["logloss"] for b in blocks], dtype=float)
    br = np.asarray([b["delta"]["brier"] for b in blocks], dtype=float)
    ac = np.asarray([b["delta"]["accuracy"] for b in blocks], dtype=float)

    return {
        "status": "OK",
        "schema_version": 1,
        "research_only": True,
        "production_changed": False,
        "strict_pit": False,
        "archive_publication_time_unknown": True,
        "promotion_evidence_eligible": False,
        "horizon": HORIZON,
        "n": len(rows),
        "development_n": len(development),
        "final_holdout_n": len(holdout),
        "gate": {
            "threshold": GATE_THRESHOLD,
            "min_training_rows": MIN_GATE_ROWS,
            "final_training_rows": len(gate_y),
            "final_positive_rate": float(np.mean(gate_y)) if gate_y else 0.0,
            "final_coverage": float(final_used.mean()),
            "final_probability_mean": float(np.mean(final_prob)),
            "strictly_prior_labels": True,
        },
        "development": {
            "blocks": len(blocks),
            "baseline": _metrics(
                development,
                np.vstack(
                    [
                        np.asarray([[r["p5"][c] for c in CLASSES] for r in development])
                    ]
                ),
            ) if False else None,
            "gated_aggregate": {
                "accuracy": float(
                    sum(b["n"] * b["candidate"]["accuracy"] for b in blocks)
                    / sum(b["n"] for b in blocks)
                ),
                "logloss": float(
                    sum(b["n"] * b["candidate"]["logloss"] for b in blocks)
                    / sum(b["n"] for b in blocks)
                ),
                "brier": float(
                    sum(b["n"] * b["candidate"]["brier"] for b in blocks)
                    / sum(b["n"] for b in blocks)
                ),
            },
            "mean_delta": {
                "accuracy": float(ac.mean()),
                "logloss": float(ll.mean()),
                "brier": float(br.mean()),
            },
            "stability": {
                "non_worse_accuracy_ratio": float(np.mean(ac >= -0.005)),
                "improved_logloss_ratio": float(np.mean(ll < 0)),
                "improved_brier_ratio": float(np.mean(br < 0)),
            },
        },
        "final_holdout": {
            "protected": True,
            "used_for_selection": False,
            "baseline": hold_base,
            "raw_candidate": hold_raw,
            "candidate": hold_gate,
            "delta": _delta(hold_gate, hold_base),
            "raw_delta": _delta(hold_raw, hold_base),
        },
        "blocks": blocks,
        "eligibility": False,
    }


def main() -> None:
    payload = {
        "schema_version": 1,
        "research_only": True,
        "production_changed": False,
        "strict_pit": False,
        "promotion_evidence_eligible": False,
        "horizons": {HORIZON: evaluate()},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
