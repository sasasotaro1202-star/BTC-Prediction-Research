# Production 70% Accuracy Target

## Objective

The project will target >=70% directional accuracy for predictions admitted to Production, while preserving point-in-time integrity and out-of-sample validity.

This is a target, not a claimed result. The system must never obtain the target by tuning thresholds on the final holdout, cherry-picking a short favorable period, silently excluding losses, changing labels after evaluation, mixing Binance-primary and fallback cohorts, or presenting tiny samples without coverage.

## Why selective prediction is required

Current historical research shows that universal 5m/10m direction prediction is noisy. The 45-day research artifact has full-sample accuracy around the mid-40% range, while some high-confidence subsets exceed 70% only at very low coverage. The scientifically defensible route toward a 70% target is selective prediction / abstention rather than forcing every market state into a directional class.

Production must distinguish:

1. eligible — passes the frozen production gate;
2. abstain — evidence is insufficient;
3. fallback — a non-primary source was required;
4. invalid — PIT, provenance, or data-integrity requirements fail.

## Promotion gate

A candidate selective policy may be promoted only if a frozen future/OOS evaluation demonstrates:

- accuracy >= 70%;
- a pre-declared minimum number of settled predictions;
- coverage reported alongside accuracy;
- stable performance across multiple chronological blocks;
- no material calibration regression;
- no PIT/leakage violation;
- no dependence on a single venue or short regime;
- reproducible results from the same dataset and code revision.

A single 70% result on a handful of predictions is not sufficient.

## Model-selection protocol

Thresholds, features, model type, ensemble weights, calibration and gate rules are selected using training/validation periods only. The final holdout remains untouched until the candidate is frozen.

Candidate methods should include Random Forest / ExtraTrees, gradient boosting, calibrated probabilities, microstructure features, volatility-adaptive neutral zones, regime-aware gating, cross-venue agreement, order-book imbalance, taker-flow imbalance, open-interest/funding context, and selective abstention.

The simplest model that survives the OOS gate wins; complexity is not a promotion criterion.

## Important interpretation

The 70% target is not a promise that BTC can be predicted with 70% accuracy on every 5-minute or 10-minute interval. Short-horizon prediction is noisy and high accuracy can depend heavily on filtering, label construction, or regime selection.

The production KPI is therefore accuracy plus coverage, with calibration, robustness and PIT safety as mandatory guardrails.

## Current status

The selective gate implementation in src/selective_gate.py is intentionally conservative and is not evidence of 70% accuracy. It can abstain but cannot manufacture confidence.

The gate must remain Research/Conditional until a frozen future evaluation proves the target.
