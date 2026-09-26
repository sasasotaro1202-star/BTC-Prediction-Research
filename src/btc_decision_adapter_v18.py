"""Research-only adapter from BTC public-venue Situation Cards to v18 Decision Objects.

This module never computes a prediction. It only packages an already-produced
probability distribution together with causal shadow-venue evidence. Missing
or non-causal evidence is fail-closed.
"""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from typing import Any, Mapping

from src.btc_decision_object_v18 import PredictionDecisionObject


def _canonical_event_ids(ids: list[str]) -> str:
    raw = json.dumps(sorted(ids), separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _provenance(events: list[dict[str, Any]], used_ids: list[str], cutoff_ms: int) -> tuple[dict[str, Any], ...]:
    by_group: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        event_id = str(event.get("event_id", ""))
        if event_id in used_ids:
            by_group[(str(event.get("source", "")), str(event.get("event_type", "")))].append(event)

    found_ids = [str(e.get("event_id", "")) for group in by_group.values() for e in group]
    if set(used_ids) != set(found_ids) or len(found_ids) != len(used_ids):
        raise ValueError("situation_card_missing_event_provenance")

    out: list[dict[str, Any]] = []
    for (source, event_type), group in sorted(by_group.items()):
        if not source or not event_type:
            raise ValueError("invalid_shadow_event_identity")
        available = []
        retrieved = []
        event_times = []
        payload_hashes = []
        for event in group:
            try:
                event_time = int(event["event_time_ms"])
                available_at = int(event["available_at_ms"])
                retrieved_at = int(event["retrieved_at_ms"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("invalid_shadow_event_provenance") from exc
            if event_time > available_at:
                raise ValueError("event_after_available_time")
            if available_at > cutoff_ms:
                raise ValueError("future_shadow_information")
            if retrieved_at < available_at:
                raise ValueError("retrieval_before_available_time")
            payload_hash = str(event.get("payload_sha256", ""))
            if not payload_hash:
                raise ValueError("missing_shadow_payload_hash")
            event_times.append(event_time)
            available.append(available_at)
            retrieved.append(retrieved_at)
            payload_hashes.append(payload_hash)

        out.append(
            {
                "source": source,
                "event_type": event_type,
                "available_at_ms": max(available),
                "retrieved_at_ms": max(retrieved),
                "event_time_min_ms": min(event_times),
                "event_time_max_ms": max(event_times),
                "event_count": len(group),
                "event_ids_sha256": _canonical_event_ids([str(e["event_id"]) for e in group]),
                "payload_hashes_sha256": hashlib.sha256(
                    json.dumps(sorted(payload_hashes), separators=(",", ":")).encode("utf-8")
                ).hexdigest(),
            }
        )
    return tuple(out)


def build_decision_from_situation_card(
    *,
    cutoff_ms: int,
    probability: Mapping[str, float],
    situation_card: Mapping[str, Any],
    events: list[dict[str, Any]],
    target: str = "direction",
    horizon: str = "5m",
    granularity: str = "1m",
    model: str = "precomputed_probability",
    strategy: str = "public_venue_shadow",
    compute_budget: str = "standard",
    update_policy: str = "research_only",
    current_regime: str | None = None,
    action: str = "maintain",
) -> PredictionDecisionObject:
    """Package causal Situation Card evidence without producing a prediction."""
    if int(situation_card.get("prediction_cutoff_ms", -1)) != int(cutoff_ms):
        raise ValueError("situation_card_cutoff_mismatch")

    used_ids = [str(x) for x in situation_card.get("used_event_ids", [])]
    if len(used_ids) != len(set(used_ids)):
        raise ValueError("duplicate_situation_event_ids")

    provenance = _provenance(events, used_ids, int(cutoff_ms)) if used_ids else ()
    pit_status = "verified" if provenance else "deferred"

    return PredictionDecisionObject(
        prediction_time_ms=int(cutoff_ms),
        target=target,
        horizon=horizon,
        granularity=granularity,
        output_type="probability",
        probability=dict(probability),
        distribution=dict(probability),
        predictability=None,
        confidence=None,
        uncertainty=None,
        current_state="public_venue_shadow",
        future_state=None,
        current_regime=current_regime,
        future_regime=None,
        trajectory=(),
        scenarios=(),
        branches=(),
        disagreement=None,
        error_correlation=None,
        future_failure=None,
        time_to_failure_ms=None,
        ood=False,
        novelty=None,
        tail_risk=None,
        forecast_lifetime_ms=None,
        freshness_ms=situation_card.get("freshness_ms"),
        revision_risk=None,
        update_need=None,
        next_update_time_ms=None,
        information_value=None,
        model=model,
        strategy=strategy,
        compute_budget=compute_budget,
        update_policy=update_policy,
        action=action,
        pit_status=pit_status,
        provenance=provenance,
    )


__all__ = ["build_decision_from_situation_card"]