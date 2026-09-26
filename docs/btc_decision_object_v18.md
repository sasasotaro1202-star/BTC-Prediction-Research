# BTC Prediction Decision Object v18

Research-only implementation of the attached ULTIMATE UNIFIED SOURCE v18.

The object captures the prediction-time decision state rather than only the
predicted class/probability. It includes target, horizon, granularity, output,
state/regime, predictability, uncertainty, disagreement, failure risk, OOD,
novelty, tail risk, forecast lifetime, freshness, revision/update need,
information value, model, strategy, compute/update policy, action, PIT status
and provenance.

## Safety contract

Every provenance record must contain `available_at_ms`, and it must satisfy:

`available_at_ms <= prediction_time_ms`

Missing/unknown availability is rejected. A `verified` PIT object must contain
at least one provenance record. This object does not promote or modify a model.

## Evaluation use

The research layer can use these objects to study:

- case-level routing and action selection;
- forecast lifetime / revision risk;
- predictability and forecastability ceiling;
- model disagreement and error correlation;
- OOD / novelty / tail-risk stratification;
- information-value decisions;
- chronological OOS policy evaluation.

The object ID is deterministic: it is the SHA-256 of the canonical payload
excluding the ID itself. This makes replay and provenance comparison
deterministic without mutating historical snapshots.
