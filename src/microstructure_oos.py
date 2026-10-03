"""Research-only OOS evaluation of stored PIT microstructure signals.

This lane tests whether microstructure fields already captured at prediction
time improve the current production Champion. It never mutates production
artifacts or substitutes missing values.

Feature variants:
- binance_micro: depth/taker/funding/OI from required Binance-primary sources.
- market_flow_v2: binance_micro plus deterministic closed-bar VWAP/volume/range features
  and windowed taker-flow features when a fresh Binance WebSocket cache is present.
- cross_venue: binance_micro plus spot/futures and Bybit snapshot divergence when
  their source-native availability is explicitly valid under the strict PIT policy.

All model selection uses chronological development OOS with the repository's
purge/embargo evaluator. The final 20% is descriptive only.
"""
from __future__ import annotations

import json
import math
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from feature_schema import FEATURES as BASE_FEATURES
from model_compare import (
    HORIZONS,
    CLASSES,
    PURGE_BARS,
    EMBARGO_BARS,
    aligned,
    metrics,
    walk_forward,
    hac_test,
    _adjusted_alpha,
    loss_arrays,
    load_primary_production_strict_rows,
)

try:
    from lightgbm import LGBMClassifier
except ImportError:
    LGBMClassifier = None

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None

DB = ROOT / "data" / "predictions.db"
OUT = ROOT / "data" / "historical_research" / "microstructure_oos.json"
FINAL_HOLDOUT_FRAC = 0.20
MIN_ROWS = 2200
MIN_OOS = 500
TEST_BLOCK = 100

BINANCE_MICRO = (
    "book_imbalance",
    "taker_imbalance",
    "funding_binance",
    "oi_log1p",
)
BINANCE_MICRO_CORE = (
    "book_imbalance",
    "taker_imbalance",
    "funding_binance",
)
MARKET_FLOW_V2 = (
    "vwap_distance_5m",
    "vwap_distance_15m",
    "vwap_distance_30m",
    "volume_burst_5m",
    "volume_burst_15m",
    "range_compression_5m",
    "range_compression_15m",
    "taker_imbalance_5m",
    "taker_imbalance_15m",
    "taker_imbalance_delta_5m_15m",
)

EXTENDED_FEATURES = (
    "ret_15m",
    "ret_30m",
    "range_position_30m",
    "trend_alignment",
)

CROSS_VENUE = (
    "cross_exchange_gap",
    "spot_futures_gap",
    "bybit_book_imbalance",
)

BYBIT_OI = ("bybit_oi_log1p",)

FEATURE_VARIANT_NAMES = (
    "binance_core",
    "binance_micro",
    "market_flow_v2_no_oi",
    "market_flow_v2",
    "full_stack_no_oi",
    "full_stack",
    "cross_venue_no_oi",
    "cross_venue",
    "binance_core_bybit_oi",
    "market_flow_v2_no_oi_bybit_oi",
    "full_stack_no_oi_bybit_oi",
)

PIT_PRIMARY_SOURCES = (
    "binance_futures",
    "binance_depth",
    "binance_taker",
    "binance_premium",
)


def _safe_json(value):
    try:
        obj = json.loads(value or "{}")
        return obj if isinstance(obj, dict) else {}
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}


def _finite(value):
    try:
        x = float(value)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def _strict_primary_sources_ok(scenario):
    if not isinstance(scenario, dict):
        return False
    provenance = scenario.get("provenance")
    if not isinstance(provenance, dict):
        return False
    sources = provenance.get("sources")
    if not isinstance(sources, dict):
        return False
    for name in PIT_PRIMARY_SOURCES:
        info = sources.get(name)
        if not isinstance(info, dict):
            return False
        status = str(info.get("status", ""))
        if status != "ok":
            return False
        for key in ("available_at", "retrieved_at", "prediction_cutoff"):
            if _finite_datetime(info.get(key)) is None:
                return False
    return True


def _finite_datetime(value):
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None


def _millisecond_datetime(value):
    try:
        return datetime.fromtimestamp(int(value) / 1000, timezone.utc)
    except (TypeError, ValueError, OverflowError, OSError):
        return None


def _micro_from_scenario(scenario, *, cross_venue=False):
    if not isinstance(scenario, dict):
        return None
    m = scenario.get("microstructure")
    if not isinstance(m, dict):
        return None
    primary = {}
    for key in ("book_imbalance", "taker_imbalance", "funding_binance", "oi"):
        value = _finite(m.get(key))
        if value is None:
            return None
        primary[key] = value
    values = {
        "book_imbalance": primary["book_imbalance"],
        "taker_imbalance": primary["taker_imbalance"],
        "funding_binance": primary["funding_binance"],
        "oi_log1p": math.log1p(max(primary["oi"], 0.0)),
    }

    if not cross_venue:
        return values

    bybit_status = scenario.get("data_quality", {}).get("bybit_depth")
    bybit_price_status = scenario.get("data_quality", {}).get("bybit_futures")
    if bybit_status != "ok" or bybit_price_status not in {"ok", "ok_current_only"}:
        return None

    for key in CROSS_VENUE:
        raw_key = key
        value = _finite(m.get(raw_key))
        if value is None:
            return None
        values[key] = value
    return values


def _bybit_oi_from_scenario(scenario, *, created_at=None):
    """Return a strict-PIT Bybit OI snapshot only when its own provenance passes."""
    if not isinstance(scenario, dict):
        return None
    micro = scenario.get("microstructure")
    dq = scenario.get("data_quality")
    provenance = scenario.get("provenance")
    sources = provenance.get("sources") if isinstance(provenance, dict) else None
    source = sources.get("bybit_ticker") if isinstance(sources, dict) else None
    if not isinstance(micro, dict) or not isinstance(dq, dict) or not isinstance(source, dict):
        return None
    if dq.get("bybit_oi") != "ok" or source.get("status") != "ok":
        return None
    fields = source.get("fields")
    if not isinstance(fields, list) or "openInterest" not in fields:
        return None
    oi = _finite(micro.get("bybit_oi"))
    if oi is None or oi <= 0:
        return None

    created = _finite_datetime(created_at) if created_at is not None else None
    source_available = _finite_datetime(source.get("available_at"))
    source_retrieved = _finite_datetime(source.get("retrieved_at"))
    source_cutoff = _finite_datetime(source.get("prediction_cutoff"))
    if None in (created, source_available, source_retrieved, source_cutoff):
        return None
    if not (source_available <= source_retrieved <= source_cutoff <= created):
        return None
    return {"bybit_oi_log1p": math.log1p(oi)}


def _extended_from_feature_json(feature_json):
    try:
        parsed = json.loads(feature_json or "{}")
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    if not isinstance(parsed, dict):
        return None
    values = {}
    for key in EXTENDED_FEATURES:
        value = _finite(parsed.get(key))
        if value is None:
            return None
        values[key] = value
    return values


def _market_flow_from_scenario(scenario, created_at=None):
    if not isinstance(scenario, dict):
        return None
    m = scenario.get("microstructure")
    if not isinstance(m, dict):
        return None
    values = {}
    for key in MARKET_FLOW_V2:
        value = _finite(m.get(key))
        if value is None:
            return None
        values[key] = value

    # Windowed taker features are accepted only with an explicit persisted
    # provenance envelope. The audit rechecks timing instead of trusting a
    # boolean freshness flag written by the producer.
    dq = scenario.get("data_quality")
    if not isinstance(dq, dict) or dq.get("binance_taker_window_transport") != "websocket_closed_klines":
        return None
    provenance = scenario.get("provenance")
    sources = provenance.get("sources") if isinstance(provenance, dict) else None
    source = sources.get("binance_taker_window") if isinstance(sources, dict) else None
    if not isinstance(source, dict) or source.get("status") != "ok":
        return None

    created = _finite_datetime(created_at) if created_at is not None else None
    source_available = _finite_datetime(source.get("available_at"))
    source_retrieved = _finite_datetime(source.get("retrieved_at"))
    source_cutoff = _finite_datetime(source.get("prediction_cutoff"))
    source_event = _finite_datetime(source.get("event_time"))
    if None in (created, source_available, source_retrieved, source_cutoff, source_event):
        return None
    if not (source_event <= source_available <= source_retrieved <= source_cutoff <= created):
        return None
    if (created - source_retrieved).total_seconds() > 180:
        return None

    # Keep the runtime millisecond marker as a redundant audit anchor.
    try:
        marker = int(dq.get("binance_taker_window_retrieved_at_ms"))
    except (TypeError, ValueError):
        return None
    marker_dt = _millisecond_datetime(marker)
    if marker_dt is None:
        return None
    if abs((marker_dt - source_retrieved).total_seconds()) > 2:
        return None
    return values


def load_variants(horizon: str):
    """Return complete-case strict-primary cohorts without imputation."""
    base = load_primary_production_strict_rows(horizon)
    if not base:
        return {
            "base": [],
            "binance_core": [],
            "binance_micro": [],
            "market_flow_v2_no_oi": [],
            "market_flow_v2": [],
            "full_stack_no_oi": [],
            "full_stack": [],
            "cross_venue_no_oi": [],
            "cross_venue": [],
            "binance_core_bybit_oi": [],
            "market_flow_v2_no_oi_bybit_oi": [],
            "full_stack_no_oi_bybit_oi": [],
        }

    scenario_by_id = {}
    with sqlite3.connect(DB) as con:
        ids = [int(r["id"]) for r in base if str(r["id"]).isdigit()]
        if ids:
            placeholders = ",".join("?" for _ in ids)
            rows = con.execute(
                f"SELECT prediction_id, feature_json, scenario_json FROM predictions WHERE prediction_id IN ({placeholders})",
                ids,
            ).fetchall()
            scenario_by_id = {
                int(r[0]): {"feature_json": r[1], "scenario": _safe_json(r[2])}
                for r in rows
            }

    variants = {
        "base": [],
        "binance_core": [],
        "binance_micro": [],
        "market_flow_v2_no_oi": [],
        "market_flow_v2": [],
        "full_stack_no_oi": [],
        "full_stack": [],
        "cross_venue_no_oi": [],
        "cross_venue": [],
        "binance_core_bybit_oi": [],
        "market_flow_v2_no_oi_bybit_oi": [],
        "full_stack_no_oi_bybit_oi": [],
    }
    for row in base:
        record = scenario_by_id.get(int(row["id"]))
        scenario = record.get("scenario") if isinstance(record, dict) else None
        feature_json = record.get("feature_json") if isinstance(record, dict) else None
        if not _strict_primary_sources_ok(scenario):
            continue
        common = {
            "id": row["id"],
            "created": row["created"],
            "target": row["target"],
            "y": row["y"],
            "production": row["production"],
        }
        base_row = {**common, "x": list(row["x"])}
        variants["base"].append(base_row)

        micro_obj = scenario.get("microstructure") if isinstance(scenario, dict) else None
        core = None
        if isinstance(micro_obj, dict):
            candidate = {k: _finite(micro_obj.get(k)) for k in BINANCE_MICRO_CORE}
            if all(v is not None for v in candidate.values()):
                core = candidate
                variants["binance_core"].append(
                    {**common, "x": list(row["x"]) + [core[k] for k in BINANCE_MICRO_CORE]}
                )

        micro = _micro_from_scenario(scenario, cross_venue=False)
        if micro is not None:
            variants["binance_micro"].append(
                {**common, "x": list(row["x"]) + [micro[k] for k in BINANCE_MICRO]}
            )

        flow = _market_flow_from_scenario(scenario, created_at=row["created"])
        bybit_oi = _bybit_oi_from_scenario(scenario, created_at=row["created"])
        if core is not None and flow is not None:
            variants["market_flow_v2_no_oi"].append(
                {
                    **common,
                    "x": list(row["x"])
                    + [core[k] for k in BINANCE_MICRO_CORE]
                    + [flow[k] for k in MARKET_FLOW_V2],
                }
            )
        if micro is not None and flow is not None:
            variants["market_flow_v2"].append(
                {
                    **common,
                    "x": list(row["x"])
                    + [micro[k] for k in BINANCE_MICRO]
                    + [flow[k] for k in MARKET_FLOW_V2],
                }
            )

        extra = _extended_from_feature_json(feature_json)

        if core is not None and flow is not None and extra is not None:
            variants["full_stack_no_oi"].append(
                {
                    **common,
                    "x": list(row["x"])
                    + [extra[k] for k in EXTENDED_FEATURES]
                    + [core[k] for k in BINANCE_MICRO_CORE]
                    + [flow[k] for k in MARKET_FLOW_V2],
                }
            )
        if micro is not None and flow is not None and extra is not None:
            variants["full_stack"].append(
                {
                    **common,
                    "x": list(row["x"])
                    + [extra[k] for k in EXTENDED_FEATURES]
                    + [micro[k] for k in BINANCE_MICRO]
                    + [flow[k] for k in MARKET_FLOW_V2],
                }
            )

        cross = _micro_from_scenario(scenario, cross_venue=True)
        if core is not None and bybit_oi is not None:
            variants["binance_core_bybit_oi"].append(
                {**common, "x": list(row["x"]) + [core[k] for k in BINANCE_MICRO_CORE] + [bybit_oi[k] for k in BYBIT_OI]}
            )
        if core is not None and flow is not None and bybit_oi is not None:
            variants["market_flow_v2_no_oi_bybit_oi"].append(
                {
                    **common,
                    "x": list(row["x"])
                    + [core[k] for k in BINANCE_MICRO_CORE]
                    + [flow[k] for k in MARKET_FLOW_V2]
                    + [bybit_oi[k] for k in BYBIT_OI],
                }
            )
        if core is not None and flow is not None and extra is not None and bybit_oi is not None:
            variants["full_stack_no_oi_bybit_oi"].append(
                {
                    **common,
                    "x": list(row["x"])
                    + [extra[k] for k in EXTENDED_FEATURES]
                    + [core[k] for k in BINANCE_MICRO_CORE]
                    + [flow[k] for k in MARKET_FLOW_V2]
                    + [bybit_oi[k] for k in BYBIT_OI],
                }
            )
        if core is not None and cross is not None:
            variants["cross_venue_no_oi"].append(
                {**common, "x": list(row["x"]) + [core[k] for k in BINANCE_MICRO_CORE] + [cross[k] for k in CROSS_VENUE]}
            )
        if cross is not None:
            variants["cross_venue"].append(
                {**common, "x": list(row["x"]) + [cross[k] for k in BINANCE_MICRO + CROSS_VENUE]}
            )
    for key in variants:
        variants[key] = sorted(variants[key], key=lambda r: (str(r["created"]), int(r["id"])))
    return variants


def coverage_diagnostics(horizon: str) -> dict:
    """Explain strict-PIT microstructure coverage without changing eligibility."""
    base = load_primary_production_strict_rows(horizon)
    diagnostics = {
        "strict_primary_rows": len(base),
        "strict_primary_first_created": min((str(r["created"]) for r in base), default=None),
        "strict_primary_last_created": max((str(r["created"]) for r in base), default=None),
        "field_presence": {
            name: 0 for name in (*BINANCE_MICRO, *BYBIT_OI, *MARKET_FLOW_V2, *EXTENDED_FEATURES, *CROSS_VENUE)
        },
        "variant_coverage": {},
    }

    scenario_by_id = {}
    ids = [int(r["id"]) for r in base if str(r["id"]).isdigit()]
    if ids:
        with sqlite3.connect(DB) as con:
            placeholders = ",".join("?" for _ in ids)
            rows = con.execute(
                f"SELECT prediction_id, feature_json, scenario_json FROM predictions "
                f"WHERE prediction_id IN ({placeholders})",
                ids,
            ).fetchall()
            scenario_by_id = {
                int(r[0]): {"feature_json": r[1], "scenario": _safe_json(r[2])}
                for r in rows
            }

    for row in base:
        record = scenario_by_id.get(int(row["id"]), {})
        scenario = record.get("scenario") if isinstance(record, dict) else {}
        feature_json = record.get("feature_json") if isinstance(record, dict) else "{}"
        micro = scenario.get("microstructure") if isinstance(scenario, dict) else {}
        if isinstance(micro, dict):
            for key in (*BINANCE_MICRO, *BYBIT_OI, *MARKET_FLOW_V2, *CROSS_VENUE):
                raw_key = "oi" if key == "oi_log1p" else ("bybit_oi" if key == "bybit_oi_log1p" else key)
                if _finite(micro.get(raw_key)) is not None:
                    diagnostics["field_presence"][key] += 1
        extra = _extended_from_feature_json(feature_json)
        if extra is not None:
            for key in EXTENDED_FEATURES:
                diagnostics["field_presence"][key] += 1

    total = len(base)
    counts = {
        "binance_core": 0,
        "binance_micro": 0,
        "market_flow_v2_no_oi": 0,
        "market_flow_v2": 0,
        "full_stack_no_oi": 0,
        "full_stack": 0,
        "cross_venue_no_oi": 0,
        "cross_venue": 0,
        "binance_core_bybit_oi": 0,
        "market_flow_v2_no_oi_bybit_oi": 0,
        "full_stack_no_oi_bybit_oi": 0,
    }
    for row in base:
        record = scenario_by_id.get(int(row["id"]), {})
        scenario = record.get("scenario") if isinstance(record, dict) else {}
        feature_json = record.get("feature_json") if isinstance(record, dict) else "{}"
        micro_obj = scenario.get("microstructure") if isinstance(scenario, dict) else None
        core_ok = bool(
            isinstance(micro_obj, dict)
            and all(_finite(micro_obj.get(k)) is not None for k in BINANCE_MICRO_CORE)
        )
        micro_ok = _micro_from_scenario(scenario) is not None
        flow_ok = _market_flow_from_scenario(scenario, created_at=row["created"]) is not None
        extra_ok = _extended_from_feature_json(feature_json) is not None
        cross_ok = _micro_from_scenario(scenario, cross_venue=True) is not None
        cross_core_ok = bool(core_ok and cross_ok)
        bybit_oi_ok = _bybit_oi_from_scenario(scenario, created_at=row["created"]) is not None
        counts["binance_core"] += int(core_ok)
        counts["binance_micro"] += int(micro_ok)
        counts["market_flow_v2_no_oi"] += int(core_ok and flow_ok)
        counts["market_flow_v2"] += int(micro_ok and flow_ok)
        counts["full_stack_no_oi"] += int(core_ok and flow_ok and extra_ok)
        counts["full_stack"] += int(micro_ok and flow_ok and extra_ok)
        counts["cross_venue_no_oi"] += int(cross_core_ok)
        counts["cross_venue"] += int(cross_ok)
        counts["binance_core_bybit_oi"] += int(core_ok and bybit_oi_ok)
        counts["market_flow_v2_no_oi_bybit_oi"] += int(core_ok and flow_ok and bybit_oi_ok)
        counts["full_stack_no_oi_bybit_oi"] += int(core_ok and flow_ok and extra_ok and bybit_oi_ok)

    for name, count in counts.items():
        diagnostics["variant_coverage"][name] = {
            "complete_rows": count,
            "missing_rows": max(0, total - count),
            "coverage_ratio": float(count / total) if total else 0.0,
        }
    return diagnostics


def factories():
    out = {
        "logreg": lambda: Pipeline([
            ("scale", StandardScaler()),
            ("model", LogisticRegression(C=0.25, max_iter=3000)),
        ]),
        "extra_trees": lambda: ExtraTreesClassifier(
            n_estimators=360,
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
    if LGBMClassifier is not None:
        out["lightgbm"] = lambda: LGBMClassifier(
            objective="multiclass",
            num_class=3,
            n_estimators=260,
            num_leaves=15,
            learning_rate=0.035,
            min_child_samples=24,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=2.0,
            random_state=42,
            n_jobs=-1,
            verbosity=-1,
        )
    if XGBClassifier is not None:
        out["xgboost"] = lambda: XGBClassifier(
            objective="multi:softprob",
            num_class=3,
            n_estimators=220,
            max_depth=4,
            learning_rate=0.035,
            min_child_weight=10,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=2.0,
            reg_alpha=0.05,
            eval_metric="mlogloss",
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        )
    return out


def _candidate_holdout(train, test, factory):
    if len(train) < 1000 or len(test) < 100:
        return None
    split = max(int(len(train) * 0.75), 800)
    if split >= len(train) - 50:
        return None
    cal_train = train[:split]
    cal_eval = train[split:]
    if len(set(r["y"] for r in cal_train)) < 3:
        return None

    cal = factory()
    y_cal = np.asarray([r["y"] for r in cal_train])
    cal.fit(np.asarray([r["x"] for r in cal_train], dtype=float), y_cal)
    temp = 1.0
    try:
        from model_compare import _temperature, apply_temperature

        temp = _temperature(
            aligned(cal, np.asarray([r["x"] for r in cal_eval], dtype=float)),
            [r["y"] for r in cal_eval],
        )
    except (TypeError, ValueError, RuntimeError):
        temp = 1.0

    model = factory()
    model.fit(
        np.asarray([r["x"] for r in train], dtype=float),
        np.asarray([r["y"] for r in train]),
    )
    raw = aligned(model, np.asarray([r["x"] for r in test], dtype=float))
    from model_compare import apply_temperature

    return apply_temperature(raw, temp)


def _block_details(ids, ys, production, candidate, block_size=TEST_BLOCK):
    details = []
    for start in range(0, len(ys), block_size):
        end = min(start + block_size, len(ys))
        if end - start < max(20, block_size // 2):
            continue
        pm = metrics(ys[start:end], production[start:end])
        cm = metrics(ys[start:end], candidate[start:end])
        details.append({
            "n": end - start,
            "logloss_delta": cm["logloss"] - pm["logloss"],
            "brier_delta": cm["brier"] - pm["brier"],
            "accuracy_delta": cm["accuracy"] - pm["accuracy"],
            "start_id": ids[start],
            "end_id": ids[end - 1],
        })
    return details


def evaluate_variant(horizon: str, rows, *, corrected_alpha):
    if len(rows) < MIN_ROWS:
        return {"status": "DEFERRED", "reason": "insufficient_strict_primary_rows", "n": len(rows)}
    split = int(len(rows) * (1.0 - FINAL_HOLDOUT_FRAC))
    development = rows[:split]
    holdout = rows[split:]
    if len(development) < 1500 or len(holdout) < 100:
        return {"status": "DEFERRED", "reason": "insufficient_development_or_holdout", "n": len(rows)}

    production_by_id = {r["id"]: r["production"] for r in rows}
    candidate_results = {}
    for name, factory in factories().items():
        wf = walk_forward(development, factory, horizon)
        if wf is None:
            continue
        aligned_prod = [production_by_id[i] for i in wf["ids"]]
        dev_m = metrics(wf["ys"], wf["probs"])
        prod_m = metrics(wf["ys"], aligned_prod)
        blocks = _block_details(wf["ids"], wf["ys"], aligned_prod, wf["probs"])
        diff_ll = np.asarray([b["logloss_delta"] for b in blocks], dtype=float)
        diff_br = np.asarray([b["brier_delta"] for b in blocks], dtype=float)
        diff_ac = np.asarray([b["accuracy_delta"] for b in blocks], dtype=float)
        row_diffs = loss_arrays(
            wf["ys"],
            aligned_prod,
            wf["probs"],
        )
        stats = {
            "logloss": hac_test(
                row_diffs["logloss"],
                PURGE_BARS[horizon],
                alpha=corrected_alpha,
            ),
            "brier": hac_test(
                row_diffs["brier"],
                PURGE_BARS[horizon],
                alpha=corrected_alpha,
            ),
        }
        eligible = (
            len(blocks) >= 8
            and float(np.mean(diff_ll < 0)) >= 0.60
            and float(np.mean(diff_br < 0)) >= 0.60
            and dev_m["logloss"] <= prod_m["logloss"] - 0.003
            and dev_m["brier"] <= prod_m["brier"] - 0.0015
            and dev_m["accuracy"] >= prod_m["accuracy"] - 0.005
            and bool(stats["logloss"]["significant"])
            and bool(stats["brier"]["significant"])
        )
        candidate_results[name] = {
            "development": {
                "candidate": dev_m,
                "production": prod_m,
                "delta": {
                    "accuracy": dev_m["accuracy"] - prod_m["accuracy"],
                    "logloss": dev_m["logloss"] - prod_m["logloss"],
                    "brier": dev_m["brier"] - prod_m["brier"],
                },
            },
            "block_stability": {
                "blocks": len(blocks),
                "improved_logloss_ratio": float(np.mean(diff_ll < 0)) if len(diff_ll) else 0.0,
                "improved_brier_ratio": float(np.mean(diff_br < 0)) if len(diff_br) else 0.0,
                "non_worse_accuracy_ratio": float(np.mean(diff_ac >= -0.005)) if len(diff_ac) else 0.0,
            },
            "statistical_tests": stats,
            "eligible_pending_frozen_holdout_confirmation": bool(eligible),
        }

        hold_pred = _candidate_holdout(development, holdout, factory)
        if hold_pred is not None:
            y_hold = [r["y"] for r in holdout]
            hold_prod = [r["production"] for r in holdout]
            candidate_results[name]["final_holdout"] = {
                "candidate": metrics(y_hold, hold_pred),
                "production": metrics(y_hold, hold_prod),
                "delta": {
                    "accuracy": metrics(y_hold, hold_pred)["accuracy"] - metrics(y_hold, hold_prod)["accuracy"],
                    "logloss": metrics(y_hold, hold_pred)["logloss"] - metrics(y_hold, hold_prod)["logloss"],
                    "brier": metrics(y_hold, hold_pred)["brier"] - metrics(y_hold, hold_prod)["brier"],
                },
            }
        else:
            candidate_results[name]["final_holdout"] = {"status": "DEFERRED"}

    return {
        "status": "OK",
        "research_only": True,
        "production_changed": False,
        "final_holdout_protected": True,
        "promotion_evidence_eligible": True,
        "n": len(rows),
        "development_n": len(development),
        "final_holdout_n": len(holdout),
        "candidates": candidate_results,
    }


def main():
    result = {
        "schema_version": 1,
        "research_only": True,
        "production_changed": False,
        "final_holdout_protected": True,
        "policy": "strict_primary_pit_microstructure_complete_case_no_imputation_purged_embargoed_chronological_oos",
        "feature_variants": {
            "binance_core": list(BINANCE_MICRO_CORE),
            "binance_micro": list(BINANCE_MICRO),
            "market_flow_v2_no_oi": list(BINANCE_MICRO_CORE + MARKET_FLOW_V2),
            "market_flow_v2": list(BINANCE_MICRO + MARKET_FLOW_V2),
            "full_stack_no_oi": list(BASE_FEATURES + EXTENDED_FEATURES + BINANCE_MICRO_CORE + MARKET_FLOW_V2),
            "full_stack": list(
                BASE_FEATURES + EXTENDED_FEATURES + BINANCE_MICRO + MARKET_FLOW_V2
            ),
            "cross_venue_no_oi": list(BINANCE_MICRO_CORE + CROSS_VENUE),
            "cross_venue": list(BINANCE_MICRO + CROSS_VENUE),
            "binance_core_bybit_oi": list(BASE_FEATURES + BINANCE_MICRO_CORE + BYBIT_OI),
            "market_flow_v2_no_oi_bybit_oi": list(BASE_FEATURES + BINANCE_MICRO_CORE + MARKET_FLOW_V2 + BYBIT_OI),
            "full_stack_no_oi_bybit_oi": list(BASE_FEATURES + EXTENDED_FEATURES + BINANCE_MICRO_CORE + MARKET_FLOW_V2 + BYBIT_OI),
        },
        "horizons": {},
    }
    for h in HORIZONS:
        variants = load_variants(h)
        coverage = coverage_diagnostics(h)
        family_count = max(1, len(factories()) * len(FEATURE_VARIANT_NAMES))
        corrected_alpha = _adjusted_alpha(0.05, family_count)
        result["horizons"][h] = {
            "base_strict_primary_rows": len(variants["base"]),
            "coverage_diagnostics": coverage,
            "binance_core_rows": len(variants["binance_core"]),
            "binance_micro_rows": len(variants["binance_micro"]),
            "market_flow_v2_no_oi_rows": len(variants["market_flow_v2_no_oi"]),
            "market_flow_v2_rows": len(variants["market_flow_v2"]),
            "full_stack_no_oi_rows": len(variants["full_stack_no_oi"]),
            "full_stack_rows": len(variants["full_stack"]),
            "cross_venue_no_oi_rows": len(variants["cross_venue_no_oi"]),
            "cross_venue_rows": len(variants["cross_venue"]),
            "binance_core_bybit_oi_rows": len(variants["binance_core_bybit_oi"]),
            "market_flow_v2_no_oi_bybit_oi_rows": len(variants["market_flow_v2_no_oi_bybit_oi"]),
            "full_stack_no_oi_bybit_oi_rows": len(variants["full_stack_no_oi_bybit_oi"]),
            "binance_core": evaluate_variant(
                h, variants["binance_core"], corrected_alpha=corrected_alpha
            ),
            "binance_micro": evaluate_variant(
                h, variants["binance_micro"], corrected_alpha=corrected_alpha
            ),
            "market_flow_v2_no_oi": evaluate_variant(
                h, variants["market_flow_v2_no_oi"], corrected_alpha=corrected_alpha
            ),
            "market_flow_v2": evaluate_variant(
                h, variants["market_flow_v2"], corrected_alpha=corrected_alpha
            ),
            "full_stack_no_oi": evaluate_variant(
                h, variants["full_stack_no_oi"], corrected_alpha=corrected_alpha
            ),
            "full_stack": evaluate_variant(
                h, variants["full_stack"], corrected_alpha=corrected_alpha
            ),
            "cross_venue_no_oi": evaluate_variant(
                h, variants["cross_venue_no_oi"], corrected_alpha=corrected_alpha
            ),
            "cross_venue": evaluate_variant(
                h, variants["cross_venue"], corrected_alpha=corrected_alpha
            ),
            "binance_core_bybit_oi": evaluate_variant(
                h, variants["binance_core_bybit_oi"], corrected_alpha=corrected_alpha
            ),
            "market_flow_v2_no_oi_bybit_oi": evaluate_variant(
                h, variants["market_flow_v2_no_oi_bybit_oi"], corrected_alpha=corrected_alpha
            ),
            "full_stack_no_oi_bybit_oi": evaluate_variant(
                h, variants["full_stack_no_oi_bybit_oi"], corrected_alpha=corrected_alpha
            ),
            "corrected_alpha": corrected_alpha,
        }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
