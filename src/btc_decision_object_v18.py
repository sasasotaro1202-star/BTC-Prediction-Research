"""Research-only v18 Prediction Decision Object.

The object is a causal snapshot of a prediction decision at prediction time.
It does not alter the production predictor and contains no network calls.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Literal, Mapping


Action = Literal[
    "maintain",
    "revise",
    "recompute",
    "deep_recompute",
    "information_add",
    "strategy_change",
    "model_change",
    "horizon_change",
    "output_change",
    "scenario",
    "prediction_set",
    "abstain",
    "fallback",
]
PITStatus = Literal["verified", "deferred", "failed"]

_ACTIONS = frozenset((
    "maintain", "revise", "recompute", "deep_recompute", "information_add",
    "strategy_change", "model_change", "horizon_change", "output_change",
    "scenario", "prediction_set", "abstain", "fallback",
))
_PIT_STATUSES = frozenset(("verified", "deferred", "failed"))


def _finite_probability_map(values: Mapping[str, Any]) -> dict[str, float]:
    out = {}
    for key, value in values.items():
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid_probability:{key}") from exc
        if not 0.0 <= number <= 1.0:
            raise ValueError(f"probability_out_of_range:{key}")
        out[str(key)] = number
    if not out:
        raise ValueError("empty_probability_distribution")
    total = sum(out.values())
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"probability_sum_not_one:{total}")
    return out


def _canonical_hash(payload: Mapping[str, Any]) -> str:
    raw = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True)
class PredictionDecisionObject:
    prediction_time_ms: int
    target: str
    horizon: str
    granularity: str
    output_type: str
    probability: dict[str, float]
    distribution: dict[str, float]
    predictability: float | None
    confidence: float | None
    uncertainty: float | None
    current_state: str | None
    future_state: str | None
    current_regime: str | None
    future_regime: str | None
    trajectory: tuple[dict[str, Any], ...]
    scenarios: tuple[dict[str, Any], ...]
    branches: tuple[dict[str, Any], ...]
    disagreement: float | None
    error_correlation: float | None
    future_failure: float | None
    time_to_failure_ms: int | None
    ood: bool
    novelty: float | None
    tail_risk: float | None
    forecast_lifetime_ms: int | None
    freshness_ms: int | None
    revision_risk: float | None
    update_need: float | None
    next_update_time_ms: int | None
    information_value: float | None
    model: str
    strategy: str
    compute_budget: str
    update_policy: str
    action: Action
    pit_status: PITStatus
    provenance: tuple[dict[str, Any], ...]
    prediction_id: str = ""

    def __post_init__(self) -> None:
        if int(self.prediction_time_ms) <= 0:
            raise ValueError("invalid_prediction_time")
        if not self.target or not self.horizon or not self.granularity or not self.output_type:
            raise ValueError("missing_prediction_identity")
        if not self.model or not self.strategy:
            raise ValueError("missing_model_strategy")
        if self.action not in _ACTIONS:
            raise ValueError(f"invalid_action:{self.action}")
        if self.pit_status not in _PIT_STATUSES:
            raise ValueError(f"invalid_pit_status:{self.pit_status}")

        probability = _finite_probability_map(self.probability)
        distribution = _finite_probability_map(self.distribution)
        if probability != distribution:
            raise ValueError("probability_distribution_mismatch")
        object.__setattr__(self, "probability", probability)
        object.__setattr__(self, "distribution", distribution)

        for name in (
            "predictability",
            "confidence",
            "uncertainty",
            "disagreement",
            "error_correlation",
            "future_failure",
            "novelty",
            "tail_risk",
            "revision_risk",
            "update_need",
            "information_value",
        ):
            value = getattr(self, name)
            if value is not None and not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{name}_out_of_range")

        for name in ("time_to_failure_ms", "forecast_lifetime_ms", "freshness_ms"):
            value = getattr(self, name)
            if value is not None and int(value) < 0:
                raise ValueError(f"{name}_negative")

        if self.next_update_time_ms is not None:
            if int(self.next_update_time_ms) < int(self.prediction_time_ms):
                raise ValueError("next_update_before_prediction")

        verified = 0
        for record in self.provenance:
            if not isinstance(record, Mapping):
                raise ValueError("invalid_provenance_record")
            available = record.get("available_at_ms")
            if available is None:
                raise ValueError("provenance_missing_available_at")
            try:
                available = int(available)
            except (TypeError, ValueError) as exc:
                raise ValueError("invalid_provenance_available_at") from exc
            if available > int(self.prediction_time_ms):
                raise ValueError("future_information_in_provenance")
            verified += 1

        if self.pit_status == "verified" and not verified:
            raise ValueError("verified_pit_requires_provenance")

        calculated_id = self.compute_prediction_id()
        if self.prediction_id and self.prediction_id != calculated_id:
            raise ValueError("prediction_id_mismatch")
        object.__setattr__(self, "prediction_id", calculated_id)

    def payload_without_id(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("prediction_id", None)
        payload["trajectory"] = list(self.trajectory)
        payload["scenarios"] = list(self.scenarios)
        payload["branches"] = list(self.branches)
        payload["provenance"] = list(self.provenance)
        return payload

    def compute_prediction_id(self) -> str:
        return _canonical_hash(self.payload_without_id())

    def to_dict(self) -> dict[str, Any]:
        payload = self.payload_without_id()
        payload["prediction_id"] = self.prediction_id or _canonical_hash(
            self.payload_without_id()
        )
        return payload

    def require_verified_pit(self) -> "PredictionDecisionObject":
        if self.pit_status != "verified":
            raise ValueError(f"pit_not_verified:{self.pit_status}")
        return self

    @property
    def highest_probability_class(self) -> str:
        return max(self.probability, key=self.probability.get)

    @property
    def confidence_gap(self) -> float:
        ordered = sorted(self.probability.values(), reverse=True)
        return 0.0 if len(ordered) < 2 else ordered[0] - ordered[1]

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PredictionDecisionObject":
        return cls(
            prediction_time_ms=int(payload["prediction_time_ms"]),
            target=str(payload["target"]),
            horizon=str(payload["horizon"]),
            granularity=str(payload["granularity"]),
            output_type=str(payload["output_type"]),
            probability=dict(payload["probability"]),
            distribution=dict(payload["distribution"]),
            predictability=payload.get("predictability"),
            confidence=payload.get("confidence"),
            uncertainty=payload.get("uncertainty"),
            current_state=payload.get("current_state"),
            future_state=payload.get("future_state"),
            current_regime=payload.get("current_regime"),
            future_regime=payload.get("future_regime"),
            trajectory=tuple(payload.get("trajectory", ())),
            scenarios=tuple(payload.get("scenarios", ())),
            branches=tuple(payload.get("branches", ())),
            disagreement=payload.get("disagreement"),
            error_correlation=payload.get("error_correlation"),
            future_failure=payload.get("future_failure"),
            time_to_failure_ms=payload.get("time_to_failure_ms"),
            ood=bool(payload.get("ood", False)),
            novelty=payload.get("novelty"),
            tail_risk=payload.get("tail_risk"),
            forecast_lifetime_ms=payload.get("forecast_lifetime_ms"),
            freshness_ms=payload.get("freshness_ms"),
            revision_risk=payload.get("revision_risk"),
            update_need=payload.get("update_need"),
            next_update_time_ms=payload.get("next_update_time_ms"),
            information_value=payload.get("information_value"),
            model=str(payload["model"]),
            strategy=str(payload["strategy"]),
            compute_budget=str(payload["compute_budget"]),
            update_policy=str(payload["update_policy"]),
            action=payload["action"],
            pit_status=payload["pit_status"],
            provenance=tuple(payload.get("provenance", ())),
            prediction_id=str(payload.get("prediction_id", "")),
        )


def build_research_decision(
    *,
    prediction_time_ms: int,
    target: str,
    horizon: str,
    granularity: str,
    probability: Mapping[str, float],
    model: str,
    strategy: str,
    pit_status: PITStatus,
    provenance: tuple[dict[str, Any], ...],
    action: Action = "maintain",
    output_type: str = "probability",
    compute_budget: str = "standard",
    update_policy: str = "adaptive",
    **optional: Any,
) -> PredictionDecisionObject:
    """Convenience builder for research experiments only."""
    probabilities = dict(probability)
    return PredictionDecisionObject(
        prediction_time_ms=prediction_time_ms,
        target=target,
        horizon=horizon,
        granularity=granularity,
        output_type=output_type,
        probability=probabilities,
        distribution=probabilities.copy(),
        predictability=optional.get("predictability"),
        confidence=optional.get("confidence"),
        uncertainty=optional.get("uncertainty"),
        current_state=optional.get("current_state"),
        future_state=optional.get("future_state"),
        current_regime=optional.get("current_regime"),
        future_regime=optional.get("future_regime"),
        trajectory=tuple(optional.get("trajectory", ())),
        scenarios=tuple(optional.get("scenarios", ())),
        branches=tuple(optional.get("branches", ())),
        disagreement=optional.get("disagreement"),
        error_correlation=optional.get("error_correlation"),
        future_failure=optional.get("future_failure"),
        time_to_failure_ms=optional.get("time_to_failure_ms"),
        ood=bool(optional.get("ood", False)),
        novelty=optional.get("novelty"),
        tail_risk=optional.get("tail_risk"),
        forecast_lifetime_ms=optional.get("forecast_lifetime_ms"),
        freshness_ms=optional.get("freshness_ms"),
        revision_risk=optional.get("revision_risk"),
        update_need=optional.get("update_need"),
        next_update_time_ms=optional.get("next_update_time_ms"),
        information_value=optional.get("information_value"),
        model=model,
        strategy=strategy,
        compute_budget=compute_budget,
        update_policy=update_policy,
        action=action,
        pit_status=pit_status,
        provenance=provenance,
    )
