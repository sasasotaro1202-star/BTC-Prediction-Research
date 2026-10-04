"""PIT-safe multi-step state-trajectory research primitives.

The trajectory model predicts future *observable model/market state buckets*
from states available strictly before the prediction point. It is research-only:
it never changes Production artifacts or registry state.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np

try:
    from src.feature_schema import FEATURES
except ModuleNotFoundError:
    from feature_schema import FEATURES

STATE_COUNT = 18
STATE_NAME = "direction_x_momentum_x_disagreement"
TRANSITION_SMOOTHING = 2.0

_DIRECTION_INDEX = {"DOWN": 0, "FLAT": 1, "UP": 2}
_MOMENTUM_FEATURE_WEIGHTS = (
    ("ret_1m", 0.30),
    ("ret_3m", 0.20),
    ("ret_5m", 0.15),
    ("ret_10m", 0.10),
    ("acceleration", 0.10),
    ("ema_gap_5m", 0.10),
    ("ema_gap_10m", 0.05),
)
_FEATURE_INDEX = {name: idx for idx, name in enumerate(FEATURES)}


def _normalize_probability(row: np.ndarray) -> np.ndarray:
    values = np.asarray(row, dtype=float)
    values = np.nan_to_num(values, nan=1.0 / 3.0, posinf=1.0, neginf=0.0)
    values = np.clip(values, 0.0, 1.0)
    total = float(values.sum())
    return values / total if total > 1e-12 else np.full(3, 1.0 / 3.0)


def _momentum_bucket(features: np.ndarray) -> int:
    score = 0.0
    weight_total = 0.0
    row = np.asarray(features, dtype=float)
    for name, weight in _MOMENTUM_FEATURE_WEIGHTS:
        idx = _FEATURE_INDEX.get(name)
        if idx is None or idx >= len(row):
            continue
        value = row[idx]
        if not np.isfinite(value):
            continue
        score += weight * float(np.sign(value))
        weight_total += weight
    if weight_total <= 0.0:
        return 1
    score /= weight_total
    if score <= -0.20:
        return 0
    if score >= 0.20:
        return 2
    return 1


def state_id(probabilities: np.ndarray, features: np.ndarray) -> int:
    """Map one prediction-time observation to a deterministic coarse state."""
    p = np.asarray(probabilities, dtype=float)
    if p.ndim == 2:
        mean_p = _normalize_probability(np.mean(p, axis=0))
        disagreement = float(np.std(p, axis=0).mean())
    else:
        mean_p = _normalize_probability(p)
        disagreement = 0.0
    direction = int(np.argmax(mean_p))
    momentum = _momentum_bucket(np.asarray(features, dtype=float))
    disagreement_bucket = 1 if disagreement >= 0.08 else 0
    return direction * 6 + momentum * 2 + disagreement_bucket


def state_components(identifier: int) -> tuple[str, str, str]:
    """Decode a state id for human-readable artifacts."""
    value = int(identifier)
    if value < 0 or value >= STATE_COUNT:
        raise ValueError(f"invalid_state_id:{identifier}")
    direction = ("DOWN", "FLAT", "UP")[value // 6]
    remainder = value % 6
    momentum = ("NEGATIVE", "NEUTRAL", "POSITIVE")[remainder // 2]
    disagreement = "HIGH" if remainder % 2 else "LOW"
    return direction, momentum, disagreement


def states_from_panel(panel: dict[str, np.ndarray], features: Iterable[np.ndarray]) -> list[int]:
    """Create prediction-time state ids without reading outcomes."""
    names = sorted(panel)
    if not names:
        raise ValueError("empty_panel")
    arrays = [np.asarray(panel[name], dtype=float) for name in names]
    n = len(arrays[0])
    if any(a.ndim != 2 or a.shape[1] != 3 or len(a) != n for a in arrays):
        raise ValueError("invalid_panel_shape")
    feature_rows = list(features)
    if len(feature_rows) != n:
        raise ValueError("feature_panel_length_mismatch")
    result = []
    for i in range(n):
        row_panel = np.stack([a[i] for a in arrays], axis=0)
        result.append(state_id(row_panel, np.asarray(feature_rows[i], dtype=float)))
    return result


def build_transition_matrix(
    state_sequences: Iterable[Iterable[int]],
    *,
    smoothing: float = TRANSITION_SMOOTHING,
) -> np.ndarray:
    """Estimate a smoothed first-order transition matrix from prior states."""
    sequences = []
    for seq in state_sequences:
        values = [int(x) for x in seq]
        if values:
            if any(x < 0 or x >= STATE_COUNT for x in values):
                raise ValueError("state_sequence_contains_invalid_id")
            sequences.append(values)

    next_counts = np.ones(STATE_COUNT, dtype=float)
    for seq in sequences:
        if len(seq) >= 2:
            for target in seq[1:]:
                next_counts[target] += 1.0
    global_prior = next_counts / float(next_counts.sum())

    counts = np.tile((float(smoothing) * global_prior)[None, :], (STATE_COUNT, 1))
    for seq in sequences:
        for source, target in zip(seq, seq[1:]):
            counts[source, target] += 1.0

    row_sums = counts.sum(axis=1, keepdims=True)
    return counts / np.maximum(row_sums, 1e-12)


def _direction_distribution(state_distribution: np.ndarray) -> np.ndarray:
    result = np.zeros(3, dtype=float)
    for identifier, probability in enumerate(state_distribution):
        direction = identifier // 6
        result[direction] += float(probability)
    return _normalize_probability(result)


def forecast_probabilities(
    current_state: int,
    transition_matrix: np.ndarray,
    max_steps: int,
) -> dict[str, list[float]]:
    """Forecast future direction probabilities from the current state."""
    matrix = np.asarray(transition_matrix, dtype=float)
    if matrix.shape != (STATE_COUNT, STATE_COUNT):
        raise ValueError("transition_matrix_shape_mismatch")
    if not (0 <= int(current_state) < STATE_COUNT):
        raise ValueError("current_state_out_of_bounds")
    steps = int(max_steps)
    if steps < 1:
        raise ValueError("max_steps_must_be_positive")

    distribution = np.zeros(STATE_COUNT, dtype=float)
    distribution[int(current_state)] = 1.0
    output: dict[str, list[float]] = {}
    for step in range(1, steps + 1):
        distribution = distribution @ matrix
        output[str(step)] = _direction_distribution(distribution).tolist()
    return output


def trajectory_forecast_matrix(
    panel: dict[str, np.ndarray],
    features: Iterable[np.ndarray],
    prior_state_sequences: Iterable[Iterable[int]],
    *,
    max_steps: int,
) -> dict[str, object]:
    """Return row-wise multi-step future-state direction probabilities."""
    state_ids = states_from_panel(panel, features)
    prior_sequences = [list(seq) for seq in prior_state_sequences]
    transition = build_transition_matrix(prior_sequences)
    rows: dict[str, list[list[float]]] = {str(step): [] for step in range(1, int(max_steps) + 1)}
    for identifier in state_ids:
        forecast = forecast_probabilities(identifier, transition, int(max_steps))
        for step, probs in forecast.items():
            rows[step].append(probs)

    transition_count = sum(max(0, len(seq) - 1) for seq in prior_sequences)
    return {
        "status": "OK" if transition_count >= 50 else "PARTIAL",
        "state_space": STATE_COUNT,
        "state_definition": STATE_NAME,
        "state_ids": state_ids,
        "direction_probabilities_by_step": rows,
        "transition_matrix": transition.tolist(),
        "transition_evidence": {
            "observed_transitions": int(transition_count),
            "minimum_for_full_status": 50,
            "prior_sequences": len(prior_sequences),
        },
    }


def blend_probabilities(
    base: np.ndarray,
    trajectory: np.ndarray,
    *,
    weight: float = 0.10,
) -> np.ndarray:
    """Blend a trajectory direction forecast into base probabilities."""
    base_arr = np.asarray(base, dtype=float)
    traj_arr = np.asarray(trajectory, dtype=float)
    if base_arr.shape != traj_arr.shape or base_arr.ndim != 2 or base_arr.shape[1] != 3:
        raise ValueError("trajectory_blend_shape_mismatch")
    if not (0.0 <= float(weight) <= 0.50):
        raise ValueError("trajectory_blend_weight_out_of_bounds")
    mixed = (1.0 - float(weight)) * base_arr + float(weight) * traj_arr
    totals = mixed.sum(axis=1, keepdims=True)
    return mixed / np.maximum(totals, 1e-12)


def _probability_state_features(probabilities, previous_probabilities):
    """Encode model-probability dynamics into the canonical momentum slots.

    This is explicitly a model-state proxy, not a claim about a latent market
    variable. It is useful when running the trajectory layer from persisted
    prediction events without requiring a second feature snapshot.
    """
    p = _normalize_probability(np.asarray(probabilities, dtype=float))
    prev = _normalize_probability(np.asarray(previous_probabilities, dtype=float)) if previous_probabilities is not None else p
    delta = p - prev
    momentum = float(delta[2] - delta[0])
    features = np.zeros(len(FEATURES), dtype=float)
    for name in ("ret_1m", "ret_3m", "ret_5m", "ret_10m"):
        features[_FEATURE_INDEX[name]] = momentum
    features[_FEATURE_INDEX["acceleration"]] = momentum
    features[_FEATURE_INDEX["ema_gap_5m"]] = float(p[2] - p[0])
    features[_FEATURE_INDEX["ema_gap_10m"]] = float(p[2] - p[0])
    return features


def run(db_path=None, output_dir=None):
    """Run the prequential trajectory research over persisted predictions.

    Outcomes are used only for scoring a prediction already produced at time T.
    Transition states and transition probabilities at T use rows strictly before
    T. The function always persists an explicit terminal status.
    """
    import hashlib
    import os
    import sqlite3
    from datetime import datetime, timezone
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    db = Path(db_path) if db_path is not None else root / "data" / "predictions.db"
    out_root = Path(output_dir) if output_dir is not None else root / "data" / "historical_research"
    out_root.mkdir(parents=True, exist_ok=True)

    try:
        from model_compare import metrics, prediction_precedes_target, strict_pit_provenance_reason
    except ModuleNotFoundError:
        from src.model_compare import metrics, prediction_precedes_target, strict_pit_provenance_reason

    results = {}
    for horizon in ("5m", "10m"):
        artifact = out_root / f"time_state_trajectory_{horizon}.json"
        base = {
            "schema_version": 1,
            "experiment": "time_state_trajectory_v1",
            "horizon": horizon,
            "research_only": True,
            "production_changed": False,
            "promotion_allowed": False,
            "analysis_git_sha": os.environ.get("GITHUB_SHA", "LOCAL_UNPINNED"),
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        try:
            if not db.is_file() or db.stat().st_size <= 0:
                raise RuntimeError("prediction_database_missing")
            actual_col = f"actual_direction_{horizon}"
            target_col = f"target_{horizon}"
            with sqlite3.connect(db) as con:
                rows = con.execute(
                    f"""SELECT created_at_utc,{target_col},{actual_col},
                               p_up_{horizon},p_down_{horizon},p_flat_{horizon},
                               scenario_json,model_version
                        FROM predictions
                        ORDER BY created_at_utc"""
                ).fetchall()

            eligible = []
            pit_excluded = 0
            invalid_probability = 0
            previous_created = None
            spacing_history = []
            state_history = []
            transition_snapshots = []
            scored = []

            for row in rows:
                created, target, actual, p_up, p_down, p_flat, scenario_text, model_version = row
                if str(model_version or "").startswith("DEGRADED_NO_FRESH_DATA"):
                    continue
                try:
                    current_dt = datetime.fromisoformat(str(created).replace("Z", "+00:00"))
                except (TypeError, ValueError):
                    pit_excluded += 1
                    continue
                if previous_created is not None:
                    delta_minutes = (current_dt - previous_created).total_seconds() / 60.0
                    if delta_minutes > 0:
                        spacing_history.append(delta_minutes)
                previous_created = current_dt

                try:
                    scenario = __import__("json").loads(scenario_text or "{}")
                except (TypeError, ValueError, __import__("json").JSONDecodeError):
                    scenario = {}

                if scenario.get("production_mode") != "binance_primary":
                    continue
                if strict_pit_provenance_reason(scenario, created) is not None:
                    pit_excluded += 1
                    continue
                try:
                    p = np.asarray([float(p_down), float(p_flat), float(p_up)], dtype=float)
                except (TypeError, ValueError):
                    invalid_probability += 1
                    continue
                if not np.all(np.isfinite(p)) or np.any(p < 0.0) or np.any(p > 1.0):
                    invalid_probability += 1
                    continue
                p = _normalize_probability(p)

                previous_p = None
                if eligible:
                    previous_p = eligible[-1]["probability"]
                features = _probability_state_features(p, previous_p)
                sid = state_id(p.reshape(1, -1), features)

                spacing = float(np.median(spacing_history[-500:])) if spacing_history else 5.0
                target_minutes = int(horizon.rstrip("m"))
                target_steps = max(1, int(round(target_minutes / max(spacing, 1e-6))))
                prior_states = list(state_history)

                if prior_states:
                    transition = build_transition_matrix([prior_states])
                    future = forecast_probabilities(sid, transition, target_steps)
                    trajectory_p = np.asarray(future[str(target_steps)], dtype=float)
                else:
                    trajectory_p = np.full(3, 1.0 / 3.0, dtype=float)

                if actual is not None and prediction_precedes_target(created, target):
                    blended = blend_probabilities(
                        p.reshape(1, -1),
                        trajectory_p.reshape(1, -1),
                        weight=0.10,
                    )[0]
                    eligible.append({
                        "created": created,
                        "probability": p,
                        "state_id": sid,
                        "actual": actual,
                    })
                    transition_snapshots.append(sid)
                    scored.append({
                        "baseline": p.tolist(),
                        "trajectory": trajectory_p.tolist(),
                        "blend": blended.tolist(),
                        "actual": actual,
                        "state_id": sid,
                        "target_steps": target_steps,
                        "spacing_minutes": spacing,
                    })
                else:
                    eligible.append({
                        "created": created,
                        "probability": p,
                        "state_id": sid,
                        "actual": None,
                    })

                state_history.append(sid)

            minimum_scored = 100
            if len(scored) < minimum_scored:
                result = {
                    **base,
                    "status": "DEFERRED",
                    "reason": "insufficient_strict_pit_scored_rows",
                    "scored_rows": len(scored),
                    "minimum_scored_rows": minimum_scored,
                    "transition_rows": max(0, len(state_history) - 1),
                }
            else:
                y = [item["actual"] for item in scored]
                baseline = np.asarray([item["baseline"] for item in scored], dtype=float)
                trajectory = np.asarray([item["trajectory"] for item in scored], dtype=float)
                blend = np.asarray([item["blend"] for item in scored], dtype=float)
                base_metrics = metrics(y, baseline)
                trajectory_metrics = metrics(y, trajectory)
                blend_metrics = metrics(y, blend)
                result = {
                    **base,
                    "status": "OK",
                    "scored_rows": len(scored),
                    "transition_rows": max(0, len(state_history) - 1),
                    "pit_excluded_rows": pit_excluded,
                    "invalid_probability_rows": invalid_probability,
                    "fixed_blend_weight": 0.10,
                    "oos_prequential": True,
                    "future_labels_used_for_transition_training": False,
                    "metrics": {
                        "baseline": base_metrics,
                        "trajectory": trajectory_metrics,
                        "blend": blend_metrics,
                        "delta_blend_vs_baseline": {
                            "accuracy": blend_metrics["accuracy"] - base_metrics["accuracy"],
                            "logloss": blend_metrics["logloss"] - base_metrics["logloss"],
                            "brier": blend_metrics["brier"] - base_metrics["brier"],
                            "ece": blend_metrics["calibration_error"] - base_metrics["calibration_error"],
                        },
                    },
                    "latest": {
                        "prediction_cutoff": scored[-1]["created"],
                        "state_id": int(scored[-1]["state_id"]),
                        "state_components": list(state_components(scored[-1]["state_id"])),
                        "target_steps": int(scored[-1]["target_steps"]),
                        "observed_spacing_minutes": float(scored[-1]["spacing_minutes"]),
                        "trajectory_probability": scored[-1]["trajectory"],
                        "blended_probability": scored[-1]["blend"],
                    },
                    "selection": {
                        "status": "FIXED_PRE_REGISTERED_CANDIDATE",
                        "holdout_used": False,
                        "weight_selection_from_scored_rows": False,
                    },
                }
            artifact.write_text(__import__("json").dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            results[horizon] = result
        except Exception as exc:
            result = {
                **base,
                "status": "FAILED",
                "reason": f"{type(exc).__name__}:{exc}",
            }
            artifact.write_text(__import__("json").dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            results[horizon] = result

    registry_path = out_root / "time_state_trajectory_registry.json"
    registry = {
        "schema_version": 1,
        "experiment": "time_state_trajectory_v1",
        "research_only": True,
        "production_changed": False,
        "promotion_allowed": False,
        "analysis_git_sha": os.environ.get("GITHUB_SHA", "LOCAL_UNPINNED"),
        "horizons": results,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    registry_path.write_text(__import__("json").dumps(registry, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    v6_registry = out_root / "maximum_future_generalization_v6_registry.json"
    if v6_registry.is_file():
        try:
            merged = __import__("json").loads(v6_registry.read_text(encoding="utf-8"))
            merged["time_state_trajectory"] = {
                "status": "RECORDED",
                "artifact": registry_path.name,
                "research_only": True,
                "production_changed": False,
                "horizons": {
                    h: {
                        "status": results[h].get("status"),
                        "scored_rows": results[h].get("scored_rows", 0),
                        "delta_blend_vs_baseline": results[h].get("metrics", {}).get("delta_blend_vs_baseline"),
                    }
                    for h in ("5m", "10m")
                },
            }
            v6_registry.write_text(__import__("json").dumps(merged, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        except (OSError, ValueError, TypeError):
            # The dedicated trajectory registry remains authoritative for this
            # research lane; a malformed parent registry must not erase it.
            pass

    print(__import__("json").dumps(registry, indent=2, sort_keys=True))
    return registry
