# BTC-Prediction-Research — Project Instructions

## Priority
`docs/PROJECT_SOURCE.md` and current GitHub state are authoritative. Preserve all historical experiment, OOS, failure and holdout records.

## Horizon identity
Each horizon is a separate target/version. Do not transfer 5m evidence to 24h, or any other horizon, without local chronological PIT/OOS/robustness/holdout evidence. Preserve observation, publication/availability, retrieval, prediction cutoff and revision times.

## Mandatory loop
MONITOR → DETECT → RESEARCH → IMPLEMENT → TEST → PIT → OOS/WFO → CALIBRATION → ROBUSTNESS → HOLDOUT → ADOPT/HOLD/REJECT → RELEASE → PRODUCTION → RECONCILE → FAILURE ANALYSIS → MEMORY.

## Evaluation
Prioritize future generalization, LogLoss/probabilistic quality, calibration, uncertainty and case-level diagnostics. Evaluate by horizon, regime, volatility, liquidity, missingness, source removal, feature deletion, OOD and shocks. Do not use later candles, later funding/order-book state, future market state or post-cutoff corrections.

## Cross-project transfer
Use methods from Baseball, 7-Sport, Soccer and Stock only as candidates. Transfer the mechanism, then re-verify cost, PIT, local OOS, robustness and frozen holdout before adoption.

## Fail-closed
Unknown PIT, stale required data, target mismatch, corrupted artifacts, unresolved identity or critical source failure must not be converted into a valid prediction. Green Actions and artifacts alone are not performance evidence. Paid/billing-risk/unknown-cost services remain HOLD/UNCONFIRMED.

## Status
Keep IMPLEMENTED/EXECUTED/VERIFIED/PERFORMANCE_VERIFIED/ADOPTED/PRODUCTION/HOLD/REJECTED/FAILED/BLOCKED/DEFERRED/ROLLED_BACK/UNKNOWN/UNVERIFIABLE distinct.