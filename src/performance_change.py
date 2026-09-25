from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPERIENCE = ROOT / "data" / "experience" / "experience_summary.json"
FLAT = ROOT / "data" / "historical_research" / "flat_diagnostic.json"
SNAPSHOT = ROOT / "data" / "historical_research" / "performance_snapshot.json"
CHANGE = ROOT / "data" / "historical_research" / "performance_change.json"
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


def _current_scores(experience_path: Path = EXPERIENCE, flat_path: Path = FLAT) -> dict:
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
        }
    return scores


def compare_and_record(
    snapshot_path: Path = SNAPSHOT,
    change_path: Path = CHANGE,
    experience_path: Path = EXPERIENCE,
    flat_path: Path = FLAT,
) -> dict:
    current = _current_scores(experience_path, flat_path)

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
