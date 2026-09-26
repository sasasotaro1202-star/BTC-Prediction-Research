from __future__ import annotations

import json
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPERIENCE = ROOT / "data" / "experience" / "experience_summary.json"
FLAT = ROOT / "data" / "historical_research" / "flat_diagnostic.json"
SNAPSHOT = ROOT / "data" / "historical_research" / "performance_snapshot.json"
CHANGE = ROOT / "data" / "historical_research" / "performance_change.json"
PREDICTIONS_DB = ROOT / "data" / "predictions.db"
HORIZONS = ("5m", "10m")
def _load(path: Path) -> dict:
    if not path.is_file():
        raise SystemExit(f"required performance artifact missing: {path}")
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise SystemExit(f"invalid performance artifact: {path}: {exc}") from exc
    if not isinstance(obj, dict):
        raise SystemExit(f"performance artifact is not an object: {path}")
    return obj


def _strict_pit_scores(horizon: str, db_path: Path = PREDICTIONS_DB) -> dict:
    """Score settled Binance-primary predictions that pass the canonical strict-PIT contract.

    Legacy/pre-contract/unknown-venue and fallback-venue rows are deliberately
    excluded so the monitoring baseline exactly matches the production Champion
    input domain used by the OOS comparison gate.
    """
    if horizon not in {"5m", "10m"}:
        raise ValueError(f"unsupported horizon: {horizon}")
    if not db_path.is_file():
        return {
            "status": "unavailable_missing_database",
            "n": 0,
            "accuracy": None,
            "logloss": None,
            "brier": None,
            "ece": None,
            "mode_counts": {},
        }

    try:
        from model_compare import prediction_precedes_target, strict_pit_provenance_reason
    except ModuleNotFoundError:
        from src.model_compare import prediction_precedes_target, strict_pit_provenance_reason
    try:
        from calibration import multiclass_metrics
    except ModuleNotFoundError:
        from src.calibration import multiclass_metrics

    actual_col = f"actual_direction_{horizon}"
    target_col = f"target_{horizon}"
    with sqlite3.connect(db_path) as con:
        rows = con.execute(
            f"""SELECT created_at_utc,{target_col},{actual_col},
                       p_up_{horizon},p_down_{horizon},p_flat_{horizon},
                       model_version,scenario_json
                FROM predictions
                WHERE {actual_col} IS NOT NULL
                ORDER BY created_at_utc"""
        ).fetchall()

    eligible = []
    mode_counts = {}
    for row in rows:
        created_at, target_at, actual, p_up, p_down, p_flat, model_version, scenario_text = row
        if str(model_version or "").startswith("DEGRADED_NO_FRESH_DATA"):
            continue
        if not prediction_precedes_target(created_at, target_at):
            continue
        try:
            scenario = json.loads(scenario_text or "{}")
        except (TypeError, ValueError, json.JSONDecodeError):
            scenario = {}
        if strict_pit_provenance_reason(scenario, created_at) is not None:
            continue
        # Performance monitoring must match the production Champion domain:
        # Binance-primary predictions only. Fallback venue rows remain useful
        # research observations but must not alter the production baseline.
        if scenario.get("production_mode") != "binance_primary":
            continue
        try:
            probs = [float(p_up), float(p_down), float(p_flat)]
        except (TypeError, ValueError):
            continue
        if not all(0.0 <= p <= 1.0 for p in probs) or not isinstance(actual, str):
            continue
        eligible.append((probs[0], probs[1], probs[2], actual))
        mode = str(scenario.get("production_mode", "unknown"))
        mode_counts[mode] = mode_counts.get(mode, 0) + 1

    if not eligible:
        return {"n": 0, "accuracy": None, "logloss": None, "brier": None, "ece": None, "mode_counts": mode_counts}

    accuracy, logloss, brier, ece = multiclass_metrics(eligible, horizon)
    return {
        "n": len(eligible),
        "accuracy": float(accuracy),
        "logloss": float(logloss),
        "brier": float(brier),
        "ece": float(ece),
        "mode_counts": mode_counts,
    }


def _current_scores(
    experience_path: Path = EXPERIENCE,
    flat_path: Path = FLAT,
    db_path: Path | None = None,
) -> dict:
    experience = _load(experience_path)
    flat = _load(flat_path)
    if experience.get("schema_version") != 1:
        raise SystemExit("experience summary schema mismatch")
    if flat.get("ok") is not True:
        raise SystemExit("flat diagnostic is not healthy enough for performance comparison")

    scores = {}
    for horizon in HORIZONS:
        exp_h = experience.get("horizons", {}).get(horizon, {})
        flat_h = flat.get("settled_metrics", {}).get(horizon, {})
        total = exp_h.get("total", {})
        recent100 = exp_h.get("recent", {}).get("100", {})
        final = flat_h.get("final", {})
        calibrated = flat_h.get("calibrated", {})

        if not isinstance(total, dict) or not isinstance(recent100, dict):
            raise SystemExit(f"missing experience metrics for {horizon}")
        if not isinstance(final, dict) or not isinstance(calibrated, dict):
            raise SystemExit(f"missing stage metrics for {horizon}")

        strict_db = db_path
        if strict_db is None:
            # Resolve a DB next to the supplied test/production data root so
            # temporary artifact tests remain self-contained.
            strict_db = Path(experience_path).parents[1] / "predictions.db"
        strict_pit = _strict_pit_scores(horizon, strict_db)
        scores[horizon] = {
            "experience_total_n": int(total.get("n", 0)),
            "experience_total_accuracy": float(total["accuracy"]),
            "experience_recent100_n": int(recent100.get("n", 0)),
            "experience_recent100_accuracy": float(recent100["accuracy"]),
            "final_n": int(final["n"]),
            "final_accuracy": float(final["accuracy"]),
            "final_logloss": float(final["logloss"]),
            "final_brier": float(final["brier"]),
            "final_ece": float(final["ece"]),
            "calibrated_n": int(calibrated["n"]),
            "calibrated_accuracy": float(calibrated["accuracy"]),
            "calibrated_logloss": float(calibrated["logloss"]),
            "calibrated_brier": float(calibrated["brier"]),
            "calibrated_ece": float(calibrated["ece"]),
            "strict_pit_status": str(strict_pit.get("status", "ok")),
            "strict_pit_n": int(strict_pit["n"]),
            "strict_pit_accuracy": strict_pit["accuracy"],
            "strict_pit_logloss": strict_pit["logloss"],
            "strict_pit_brier": strict_pit["brier"],
            "strict_pit_ece": strict_pit["ece"],
            "strict_pit_mode_counts": strict_pit["mode_counts"],
        }
    return scores


def compare_and_record(
    snapshot_path: Path = SNAPSHOT,
    change_path: Path = CHANGE,
    experience_path: Path = EXPERIENCE,
    flat_path: Path = FLAT,
    db_path: Path | None = None,
) -> dict:
    current = _current_scores(experience_path, flat_path, db_path=db_path)

    previous = {}
    if snapshot_path.is_file():
        previous = _load(snapshot_path).get("scores", {})
        if not isinstance(previous, dict):
            previous = {}

    changes = []
    changed = False
    for horizon in HORIZONS:
        old = previous.get(horizon, {})
        new = current[horizon]
        for key, new_value in new.items():
            old_value = old.get(key)
            if not isinstance(new_value, (int, float)) or not isinstance(old_value, (int, float)):
                continue
            # Sample-size movement is tracked, but it is not itself a score change.
            if key.endswith("_n"):
                continue
            delta = float(new_value) - float(old_value)
            if abs(delta) > 1e-12:
                changed = True
                changes.append({
                    "horizon": horizon,
                    "metric": key,
                    "old": float(old_value),
                    "new": float(new_value),
                    "delta": delta,
                })

    payload = {
        "schema_version": 1,
        "changed": changed,
        "comparison_available": bool(previous),
        "generated_at_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        "changes": changes,
        "scores": current,
    }
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_path.write_text(json.dumps({"schema_version": 1, "scores": current}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    change_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(
        f"BTC PERFORMANCE CHANGE: {'YES' if changed else 'NO'} "
        f"(comparison_available={bool(previous)})"
    )
    if changes:
        for item in changes:
            print(
                "  "
                f"{item['horizon']} {item['metric']}: "
                f"{item['old']:.10f} -> {item['new']:.10f} "
                f"(delta {item['delta']:+.10f})"
            )
    elif not previous:
        print("  baseline snapshot initialized; no prior score available for delta comparison.")
    else:
        print("  no tracked Accuracy/LogLoss/Brier/ECE score changes.")

    return payload


def main() -> int:
    compare_and_record()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
