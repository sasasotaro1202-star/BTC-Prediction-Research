"""Research-only hierarchical 3-class BTC direction OOS.

The model first estimates FLAT versus MOVE, then estimates DOWN versus UP
conditional on MOVE. This can reduce the burden on a single multiclass
classifier when neutral/flat outcomes dominate short horizons.

All evaluation is chronological and purge/embargoed. Final holdout is
descriptive only. Production artifacts are never modified.
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from model_compare import (
    HORIZONS,
    CLASSES,
    MIN_TRAIN,
    MIN_OOS,
    TEST_BLOCK,
    PURGE_BARS,
    EMBARGO_BARS,
    load_primary_production_strict_rows,
    load_archive_research_rows,
    _research_archive_rows_from_raw,
    metrics,
    walk_forward,
    aligned,
    _temperature,
    apply_temperature,
    loss_arrays,
    hac_test,
    _adjusted_alpha,
)

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
OUT = ROOT / "data" / "historical_research" / "hierarchical_oos.json"

MIN_ROWS = 2200
FINAL_HOLDOUT_FRAC = 0.20
ARCHIVE_MAX_ROWS = 12000
VARIANTS = ("logistic", "extra_trees", "hgb")


class HierarchicalClassifier:
    """Two-stage probabilistic classifier with canonical DOWN/FLAT/UP output."""

    def __init__(self, stage1_factory, stage2_factory):
        self.stage1_factory = stage1_factory
        self.stage2_factory = stage2_factory
        self.stage1 = None
        self.stage2 = None
        self.classes_ = np.asarray(CLASSES)

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        if set(y.tolist()) != set(CLASSES):
            if not {"FLAT"} <= set(y.tolist()) or not ({"DOWN", "UP"} & set(y.tolist())):
                raise ValueError("hierarchical model requires valid class labels")
        move_mask = y != "FLAT"
        if move_mask.sum() < 30 or len(set(y[move_mask].tolist())) < 2:
            raise ValueError("insufficient move rows for hierarchical direction stage")
        move = move_mask.astype(int)
        self.stage1 = self.stage1_factory()
        self.stage1.fit(X, move)
        self.stage2 = self.stage2_factory()
        self.stage2.fit(X[move_mask], y[move_mask])
        if set(self.stage2.classes_.tolist()) != {"DOWN", "UP"}:
            raise ValueError("hierarchical direction stage must contain DOWN and UP")
        return self

    def predict_proba(self, X):
        if self.stage1 is None or self.stage2 is None:
            raise RuntimeError("hierarchical model is not fitted")
        X = np.asarray(X, dtype=float)
        p_move = np.asarray(self.stage1.predict_proba(X), dtype=float)
        # stage1 classes are [0,1] for FLAT/MOVE; be explicit instead of assuming
        # ordering so a future sklearn implementation cannot silently invert them.
        move_index = {int(cls): idx for idx, cls in enumerate(self.stage1.classes_)}
        p_flat = p_move[:, move_index[0]]
        p_movement = p_move[:, move_index[1]]
        p_dir = np.asarray(self.stage2.predict_proba(X), dtype=float)
        dir_index = {str(cls): idx for idx, cls in enumerate(self.stage2.classes_)}
        p_down_move = p_dir[:, dir_index["DOWN"]]
        p_up_move = p_dir[:, dir_index["UP"]]
        out = np.column_stack([
            p_movement * p_down_move,
            p_flat,
            p_movement * p_up_move,
        ])
        out = np.clip(out, 1e-8, 1.0)
        return out / out.sum(axis=1, keepdims=True)


def _base_factories():
    return {
        "logistic": lambda: Pipeline([
            ("scale", StandardScaler()),
            ("model", LogisticRegression(C=0.25, max_iter=3000)),
        ]),
        "extra_trees": lambda: ExtraTreesClassifier(
            n_estimators=320,
            max_depth=10,
            min_samples_leaf=10,
            max_features="sqrt",
            random_state=42,
            n_jobs=-1,
        ),
        "hgb": lambda: HistGradientBoostingClassifier(
            max_iter=220,
            max_leaf_nodes=15,
            learning_rate=0.04,
            l2_regularization=1.5,
            random_state=42,
        ),
    }


def factories():
    base = _base_factories()
    return {
        name: (lambda a=base[name], b=base[name]: HierarchicalClassifier(a, b))
        for name in VARIANTS
    }


def _score_frozen_champion(horizon, rows):
    model_path = MODEL_DIR / f"{horizon}.joblib"
    meta_path = MODEL_DIR / f"{horizon}.json"
    if not model_path.is_file() or not meta_path.is_file():
        return [], None
    try:
        model = joblib.load(model_path)
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        trained_raw = meta.get("trained_at_utc")
        trained_at = (
            datetime.fromisoformat(str(trained_raw).replace("Z", "+00:00"))
            if trained_raw else None
        )
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return [], None
    out = []
    for row in rows:
        try:
            created = datetime.fromisoformat(str(row["created"]).replace("Z", "+00:00"))
        except (TypeError, ValueError):
            continue
        if trained_at is not None and created <= trained_at:
            continue
        x = np.asarray([row["x"]], dtype=float)
        if not np.isfinite(x).all():
            continue
        try:
            p = aligned(model, x)[0].tolist()
        except Exception:
            continue
        out.append({**row, "production": p})
    return out, str(meta.get("model_version", ""))


def _fresh_post_training_fallback(horizon):
    """Build research rows from fresh non-Binance venues after Champion training.
    
    This is a research-only recovery path. It preserves causal ordering by
    requiring every evaluation timestamp to be strictly after the frozen
    Champion's recorded training time, and it never becomes promotion evidence.
    """
    meta_path = MODEL_DIR / f"{horizon}.json"
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        trained_raw = meta.get("trained_at_utc")
        trained_at = (
            datetime.fromisoformat(str(trained_raw).replace("Z", "+00:00"))
            if trained_raw else None
        )
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return [], None
    if trained_at is None:
        return [], None

    loaders = []
    try:
        try:
            from src.bootstrap_train import fetch_bybit
        except ModuleNotFoundError:
            from bootstrap_train import fetch_bybit
        loaders.append(("bybit", fetch_bybit))
    except Exception:
        pass
    try:
        try:
            from src.coinbase_fallback_train import fetch_coinbase
        except ModuleNotFoundError:
            from coinbase_fallback_train import fetch_coinbase
        loaders.append(("coinbase", fetch_coinbase))
    except Exception:
        pass

    target = max(ARCHIVE_MAX_ROWS + 40, 12000)
    best = []
    best_source = None
    for source_name, loader in loaders:
        try:
            raw = loader(target)
            rows = _research_archive_rows_from_raw(
                raw, horizon, ARCHIVE_MAX_ROWS, source_name
            )
            fresh = []
            for row in rows:
                created = datetime.fromisoformat(
                    str(row["created"]).replace("Z", "+00:00")
                )
                if created > trained_at:
                    fresh.append(row)
            scored, _ = _score_frozen_champion(horizon, fresh)
            if len(scored) > len(best):
                best, best_source = scored, source_name
            if len(scored) >= MIN_ROWS:
                return (
                    scored[-ARCHIVE_MAX_ROWS:],
                    f"{source_name}_fresh_archive_frozen_champion",
                )
        except Exception:
            continue
    return (
        best[-ARCHIVE_MAX_ROWS:],
        f"{best_source}_fresh_archive_frozen_champion" if best_source else None,
    )


def load_rows(horizon):
    live = load_primary_production_strict_rows(horizon)
    if len(live) >= MIN_ROWS:
        return live, "live_binance_primary", True
    archive = load_archive_research_rows(horizon, ARCHIVE_MAX_ROWS)
    scored, _ = _score_frozen_champion(horizon, archive)
    if len(scored) >= MIN_ROWS:
        return scored[-ARCHIVE_MAX_ROWS:], "binance_vision_archive_frozen_champion", False

    fresh, fresh_source = _fresh_post_training_fallback(horizon)
    if fresh:
        return fresh, fresh_source or "fresh_venue_archive_frozen_champion", False
    return [], "binance_vision_archive_frozen_champion", False


def _candidate_holdout(factory, train, test, horizon):
    if len(train) < MIN_TRAIN or len(test) < 100:
        return None
    try:
        split = max(int(len(train) * 0.75), MIN_TRAIN - 100)
        if len(train) - split < 50:
            return None
        cal_model = factory()
        cal_model.fit(
            np.asarray([r["x"] for r in train[:split]], dtype=float),
            np.asarray([r["y"] for r in train[:split]]),
        )
        cal_probs = aligned(cal_model, np.asarray([r["x"] for r in train[split:]], dtype=float))
        temp = _temperature(
            cal_probs,
            [r["y"] for r in train[split:]],
        )
        model = factory()
        model.fit(
            np.asarray([r["x"] for r in train], dtype=float),
            np.asarray([r["y"] for r in train]),
        )
        raw = aligned(model, np.asarray([r["x"] for r in test], dtype=float))
        return apply_temperature(raw, temp)
    except (ValueError, RuntimeError, TypeError):
        return None


def evaluate(horizon):
    rows, source, evidence_eligible = load_rows(horizon)
    if len(rows) < MIN_ROWS:
        return {
            "status": "DEFERRED",
            "reason": "insufficient_rows",
            "n": len(rows),
            "data_source": source,
            "promotion_evidence_eligible": evidence_eligible,
        }
    rows = sorted(rows, key=lambda r: (str(r["created"]), int(r["id"])))
    split = int(len(rows) * (1.0 - FINAL_HOLDOUT_FRAC))
    development, holdout = rows[:split], rows[split:]
    if len(development) < MIN_TRAIN + MIN_OOS or len(holdout) < 100:
        return {"status": "DEFERRED", "reason": "insufficient_split", "n": len(rows)}

    production_by_id = {r["id"]: r["production"] for r in rows}
    corrected_alpha = _adjusted_alpha(0.05, len(VARIANTS))
    results = {}
    for name, factory in factories().items():
        wf = walk_forward(development, factory, horizon)
        if wf is None:
            continue
        prod = [production_by_id[i] for i in wf["ids"]]
        cm = metrics(wf["ys"], wf["probs"])
        pm = metrics(wf["ys"], prod)
        diff = loss_arrays(wf["ys"], prod, wf["probs"])
        stats = {
            k: hac_test(v, PURGE_BARS[horizon], alpha=corrected_alpha)
            for k, v in diff.items()
        }
        block_ll, block_br, block_ac = [], [], []
        for start in range(0, len(wf["ys"]), TEST_BLOCK):
            ys = wf["ys"][start:start + TEST_BLOCK]
            if len(ys) < max(10, TEST_BLOCK // 2):
                continue
            a = metrics(ys, wf["probs"][start:start + TEST_BLOCK])
            p = metrics(ys, prod[start:start + TEST_BLOCK])
            block_ll.append(a["logloss"] - p["logloss"])
            block_br.append(a["brier"] - p["brier"])
            block_ac.append(a["accuracy"] - p["accuracy"])
        candidate_results = {
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
                "blocks": len(block_ll),
                "improved_logloss_ratio": float(np.mean(np.asarray(block_ll) < 0)) if block_ll else 0.0,
                "improved_brier_ratio": float(np.mean(np.asarray(block_br) < 0)) if block_br else 0.0,
                "non_worse_accuracy_ratio": float(np.mean(np.asarray(block_ac) >= -0.005)) if block_ac else 0.0,
            },
            "statistical_tests": stats,
        }
        hp = _candidate_holdout(factory, development, holdout, horizon)
        candidate_results["final_holdout"] = (
            {
                "candidate": metrics([r["y"] for r in holdout], hp),
                "production": metrics([r["y"] for r in holdout], [r["production"] for r in holdout]),
            }
            if hp is not None else {"status": "DEFERRED"}
        )
        candidate_results["eligible_pending_frozen_holdout_confirmation"] = bool(
            evidence_eligible
            and len(block_ll) >= 8
            and np.mean(np.asarray(block_ll) < 0) >= 0.60
            and np.mean(np.asarray(block_br) < 0) >= 0.60
            and cm["logloss"] <= pm["logloss"] - 0.003
            and cm["brier"] <= pm["brier"] - 0.0015
            and cm["accuracy"] >= pm["accuracy"] - 0.005
            and stats["logloss"].get("significant") is True
            and stats["brier"].get("significant") is True
        )
        results[name] = candidate_results

    return {
        "status": "OK",
        "schema_version": 1,
        "research_only": True,
        "production_changed": False,
        "final_holdout_protected": True,
        "final_holdout_used_for_selection": False,
        "data_source": source,
        "promotion_evidence_eligible": evidence_eligible,
        "n": len(rows),
        "development_n": len(development),
        "final_holdout_n": len(holdout),
        "corrected_alpha": corrected_alpha,
        "variants": results,
    }


def main():
    payload = {
        "schema_version": 1,
        "research_only": True,
        "production_changed": False,
        "horizons": {h: evaluate(h) for h in HORIZONS},
        "finished_utc": datetime.now(timezone.utc).isoformat(),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
