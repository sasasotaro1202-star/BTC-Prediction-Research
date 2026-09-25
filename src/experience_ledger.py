"""Build an idempotent post-settlement BTC experience ledger.

Only settled outcomes are admitted. Each experience row is immutable with
respect to the original prediction and contains prediction-time context plus
the realized label. Aggregates are recomputed from the ledger so later research
can target weak cases without using future outcomes for the original prediction.
"""
from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path

from db import DB, init_db

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "experience"
SUMMARY = OUT_DIR / "experience_summary.json"
CLASSES = ("DOWN", "FLAT", "UP")
WINDOWS = (100, 300, 1000, 5000)


def _parse(value: str) -> datetime:
    return datetime.fromisoformat(str(value).replace("Z", "+00:00"))


def _entropy(p: list[float]) -> float:
    return float(-sum(max(x, 1e-12) * math.log(max(x, 1e-12)) for x in p) / math.log(3.0))


def _prob_tuple(row, horizon: str) -> list[float]:
    if horizon == "5m":
        vals = [row["p_down_5m"], row["p_flat_5m"], row["p_up_5m"]]
    else:
        vals = [row["p_down_10m"], row["p_flat_10m"], row["p_up_10m"]]
    p = [float(x) for x in vals]
    if not all(math.isfinite(x) and x >= 0.0 for x in p):
        raise ValueError("invalid_probability")
    s = sum(p)
    if s <= 0 or not math.isfinite(s):
        raise ValueError("invalid_probability_sum")
    return [x / s for x in p]


def _flags(value, *, include_status_prefix=False) -> list[str]:
    out = []
    if isinstance(value, list):
        for item in value:
            if item is not None:
                out.append(str(item))
    elif isinstance(value, dict):
        for key, item in sorted(value.items()):
            if include_status_prefix and isinstance(item, str) and (
                item.startswith("error:") or item in {"missing", "unknown", "unavailable"}
            ):
                out.append(f"{key}:{item}")
    return sorted(set(out))


def _feature_hash(features) -> str | None:
    if not isinstance(features, dict):
        return None
    raw = json.dumps(features, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _experience_from_row(row, horizon: str):
    scenario = json.loads(row["scenario_json"] or "{}")
    actual = row[f"actual_direction_{horizon}"]
    settled = row[f"settled_{horizon}_at_utc"]
    if actual not in CLASSES or not settled:
        return None
    p = _prob_tuple(row, horizon)
    predicted = CLASSES[max(range(3), key=lambda i: p[i])]
    confidence = max(p)
    sorted_p = sorted(p)
    margin = sorted_p[-1] - sorted_p[-2]
    regime = scenario.get("regime")
    mode = str(scenario.get("production_mode", ""))
    warnings = _flags(scenario.get("warnings"))
    quality = _flags(scenario.get("data_quality"), include_status_prefix=True)
    source = row[f"settlement_source_{horizon}"]
    created = str(row["created_at_utc"])
    target = str(row[f"target_{horizon}"])
    return (
        int(row["prediction_id"]), horizon, created, target, str(row["model_version"]),
        mode, str(regime) if regime is not None else None, predicted, actual,
        int(predicted == actual), float(confidence), float(_entropy(p)), float(margin),
        json.dumps(warnings, separators=(",", ":")),
        json.dumps(quality, separators=(",", ":")),
        str(source) if source is not None else None,
        float(row["base_price"]), float(row[f"actual_price_{horizon}"]),
        json.dumps({"DOWN": p[0], "FLAT": p[1], "UP": p[2]}, separators=(",", ":")),
        _feature_hash(scenario.get("features")),
        str(settled),
    )


def _stats(rows):
    if not rows:
        return {"n": 0, "accuracy": None, "by_predicted": {}, "by_actual": {}}
    n = len(rows)
    accuracy = sum(int(r[9]) for r in rows) / n
    by_pred = {}
    by_actual = {}
    for cls in CLASSES:
        pred_rows = [r for r in rows if r[7] == cls]
        actual_rows = [r for r in rows if r[8] == cls]
        by_pred[cls] = {
            "n": len(pred_rows),
            "accuracy": (sum(int(r[9]) for r in pred_rows) / len(pred_rows)) if pred_rows else None,
        }
        by_actual[cls] = {
            "n": len(actual_rows),
            "hit_rate": (sum(int(r[9]) for r in actual_rows) / len(actual_rows)) if actual_rows else None,
        }
    return {"n": n, "accuracy": float(accuracy), "by_predicted": by_pred, "by_actual": by_actual}


def _case_stats(rows, key_fn):
    groups = defaultdict(list)
    for row in rows:
        key = key_fn(row)
        if key is not None:
            groups[str(key)].append(row)
    out = {}
    for key, group in sorted(groups.items()):
        if len(group) < 10:
            continue
        out[key] = {
            "n": len(group),
            "accuracy": sum(int(r[9]) for r in group) / len(group),
            "avg_confidence": sum(float(r[10]) for r in group) / len(group),
        }
    return out


def _jst_hour(created: str) -> int:
    return (_parse(created).astimezone(timezone(timedelta(hours=9)))).hour


def build():
    init_db()
    with sqlite3.connect(DB) as con:
        con.row_factory = sqlite3.Row
        rows = con.execute(
            """SELECT * FROM predictions
               WHERE actual_direction_5m IS NOT NULL OR actual_direction_10m IS NOT NULL
               ORDER BY prediction_id"""
        ).fetchall()

        inserted = 0
        parse_errors = 0
        for row in rows:
            for horizon in ("5m", "10m"):
                try:
                    experience = _experience_from_row(row, horizon)
                    if experience is None:
                        continue
                    cur = con.execute(
                        """INSERT OR IGNORE INTO experience_ledger
                        (prediction_id,horizon,created_at_utc,target_at_utc,model_version,
                         production_mode,regime,predicted_direction,actual_direction,correct,
                         confidence,entropy,margin,warning_flags,data_quality_flags,source,
                         base_price,actual_price,probability_json,feature_hash,settled_at_utc)
                        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                        experience,
                    )
                    inserted += int(cur.rowcount > 0)
                except Exception:
                    parse_errors += 1

        con.commit()
        ledger_rows = con.execute(
            "SELECT * FROM experience_ledger ORDER BY settled_at_utc, experience_id"
        ).fetchall()

    payload = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "research_only": True,
        "production_changed": False,
        "source": "post_settlement_predictions",
        "inserted_since_last_run": int(inserted),
        "parse_errors": int(parse_errors),
        "total_experiences": len(ledger_rows),
        "horizons": {},
    }

    for horizon in ("5m", "10m"):
        hrows = [r for r in ledger_rows if r["horizon"] == horizon]
        serial = [tuple(r) for r in hrows]
        recent = {}
        for window in WINDOWS:
            subset = serial[-window:] if len(serial) > window else serial
            recent[str(window)] = _stats(subset)

        weak = []
        cases = {
            "regime": lambda r: r[6] or "UNKNOWN",
            "predicted_direction": lambda r: r[7],
            "hour_jst": lambda r: _jst_hour(r[2]),
            "confidence_bucket": lambda r: (
                "0.33-0.40" if r[10] < 0.40 else
                "0.40-0.50" if r[10] < 0.50 else
                "0.50-0.60" if r[10] < 0.60 else
                "0.60-0.70" if r[10] < 0.70 else "0.70+"
            ),
            "warning": lambda r: "NONE" if r[13] in ("[]", "null", "") else r[13],
            "model_version": lambda r: r[4],
            "production_mode": lambda r: r[5] or "UNKNOWN",
        }
        case_output = {}
        for name, fn in cases.items():
            stats = _case_stats(serial, fn)
            case_output[name] = stats
            for case, value in stats.items():
                if value["n"] >= 30 and value["accuracy"] < 0.35:
                    weak.append({
                        "dimension": name, "case": case,
                        "n": value["n"], "accuracy": value["accuracy"],
                    })
        weak.sort(key=lambda x: (x["accuracy"], -x["n"]))

        payload["horizons"][horizon] = {
            "total": _stats(serial),
            "recent": recent,
            "cases": case_output,
            "weak_cases_for_research": weak[:25],
        }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY.write_text(json.dumps(payload, indent=2, sort_keys=True) + "
", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))
    if parse_errors:
        raise SystemExit(f"experience ledger parse errors: {parse_errors}")
    return payload


if __name__ == "__main__":
    build()
