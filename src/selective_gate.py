"""Production-safe selective prediction gate for short-horizon BTC forecasts.

The gate is deliberately conservative: it can abstain, but it never turns a
low-confidence prediction into a high-confidence one. Thresholds are treated
as frozen policy parameters and MUST be selected on training/validation data,
never on a final holdout or the live evaluation window.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional


CLASSES = ("DOWN", "FLAT", "UP")


@dataclass(frozen=True)
class SelectiveDecision:
    eligible: bool
    predicted_class: Optional[str]
    confidence: float
    margin: float
    reason: str


def decide(
    probs: Mapping[str, float],
    *,
    min_confidence: float = 0.70,
    min_margin: float = 0.15,
    structural_class: Optional[str] = None,
    secondary_class: Optional[str] = None,
    max_cross_exchange_gap: float = 0.0005,
    cross_exchange_gap: Optional[float] = None,
) -> SelectiveDecision:
    """Return an eligibility decision without mutating the probabilities.

    This is a selective-prediction gate, not a performance guarantee.  The
    caller must evaluate coverage and accuracy on a frozen future window.
    """
    if any(c not in probs for c in CLASSES):
        return SelectiveDecision(False, None, 0.0, 0.0, "missing_class_probability")

    values = [float(probs[c]) for c in CLASSES]
    if any(v < 0.0 or v != v for v in values):
        return SelectiveDecision(False, None, 0.0, 0.0, "invalid_probability")

    total = sum(values)
    if total <= 0:
        return SelectiveDecision(False, None, 0.0, 0.0, "invalid_probability_sum")

    ranked = sorted(((float(probs[c]) / total, c) for c in CLASSES), reverse=True)
    confidence, predicted = ranked[0]
    margin = confidence - ranked[1][0]

    if confidence < min_confidence:
        return SelectiveDecision(False, predicted, confidence, margin, "confidence_below_gate")
    if margin < min_margin:
        return SelectiveDecision(False, predicted, confidence, margin, "class_margin_below_gate")
    if structural_class is not None and structural_class != predicted:
        return SelectiveDecision(False, predicted, confidence, margin, "model_structure_disagreement")
    if secondary_class is not None and secondary_class != predicted:
        return SelectiveDecision(False, predicted, confidence, margin, "cross_venue_disagreement")
    if cross_exchange_gap is not None and abs(float(cross_exchange_gap)) > max_cross_exchange_gap:
        return SelectiveDecision(False, predicted, confidence, margin, "cross_exchange_divergence")

    return SelectiveDecision(True, predicted, confidence, margin, "eligible")
