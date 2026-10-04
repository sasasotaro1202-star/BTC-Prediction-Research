"""PIT-safe multi-step state-trajectory research primitives.

The trajectory model predicts future *observable model/market state buckets*
from states available strictly before the prediction point. It is research-only:
it never changes Production artifacts or registry state.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np

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
    if not (1 <= int(current_state) < STATE_COUNT):
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
    transition = build_transition_matrix(prior_state_sequences)
    rows: dict[str, list[list[float]]] = {str(step): [] for step in range(1, int(max_steps) + 1)}
    for identifier in state_ids:
        forecast = forecast_probabilities(identifier, transition, int(max_steps))
        for step, probs in forecast.items():
            rows[step].append(probs)

    transition_count = sum(max(0, len(list(seq)) - 1) for seq in prior_state_sequences)
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
            "prior_sequences": sum(1 for _ in prior_state_sequences),
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
