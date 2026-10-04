# PROJECT_INSTRUCTIONS — BTC-Prediction-Research

## Mission
Optimize future generalization of the BTC prediction system, not historical fit alone. Treat DATA → TIME/PIT → TARGET → FEATURES → MODELS → ENSEMBLE → ROUTING → CALIBRATION → UNCERTAINTY → PREDICTABILITY → INFORMATION ACQUISITION → DECISION → PRESENTATION → OUTCOME → EXPERIENCE → FAILURE ANALYSIS → RESEARCH → OOS/WFO → ROBUSTNESS → HOLDOUT → SHADOW → PROMOTION → PRODUCTION → MONITORING → RECOVERY as one evidence-driven system.

Primary target: UP / FLAT / DOWN. Primary production horizons: 5m and 10m. Any new target/horizon remains research-only until it passes local PIT, chronological OOS/WFO, calibration, robustness, frozen holdout, reproducibility and promotion gates.

## Source of truth
The canonical technical Project Source governs detailed policy. Current GitHub HEAD, code, registries, immutable research artifacts, workflow state and validated evidence take precedence over stale historical notes.

Never rewrite mature predictions, outcomes, OOS, holdout or failure evidence to make the current state look better.

## Every run
Re-check the latest GitHub HEAD/default branch, code/config, dependencies, tests, workflows, Actions, artifacts, registries, production/champion/challenger state, current research/OOS/holdout evidence, failures, coverage debt and frontier. Prefer REUSE → REPAIR → INTEGRATE → TEST → VERIFY.

## Current evidence snapshot
The current recorded strict PIT audit is PASS for the admitted primary scope: 5m strict primary settled = 402 and 10m strict primary settled = 400, each independently above the 300-row per-horizon gate; active current-scope PIT violations = 0. Legacy observations remain quarantined rather than silently counted as valid evidence (legacy_unverified = 169, legacy_violation = 41). Robustness evidence is still sample-limited: current Production model evidence is 261 rows for 5m and 404 rows for 10m versus the 1,000-row robustness minimum, so both remain research-only. The situation-metadata cohort still needs 2,602 additional qualifying 5m-cycle rows for 5m and 2,604 for 10m to reach the 3,000-row maturity target. The innovative prediction-control v2 line remains HOLD/research-only and must not affect Production until longer independent live-primary evidence is accumulated and verified.
Current production artifact bindings recorded in the repository are:
- 5m: `bootstrap.soft_ensemble.v5.4`
- 10m: `bootstrap.bootstrap_rf`

These are production artifact metadata, not automatic evidence of superiority. Runtime must verify registry ↔ metadata ↔ artifact ↔ feature schema before use.

## PIT / time
Keep event/observation/publish/available/retrieved/processing/prediction/outcome/revision times separate. Historical feature availability must be proven with `source_available_time <= prediction_cutoff`. Retrieved-at alone is not PIT evidence. Unknown or unverifiable availability is UNKNOWN/DEFERRED/REJECTED, never PASS.

For primary readiness, strict PIT evidence is gated independently per production horizon. The PIT artifact must provide `primary_horizon_gate` entries for both 5m and 10m, and each entry must report `strict_primary_settled >= min_strict_pit_rows`, `minimum >= min_strict_pit_rows`, and `ready = true`. Aggregate verified-primary counts cannot substitute for a missing or failing horizon-specific gate.

## Leakage
Audit data, feature, target, label, publication, revision, settlement, same-event, cross-fold, calibration, model-selection, hyperparameter, experiment-selection, research-priority, holdout, cross-project, knowledge-time, metadata and benchmark contamination.

## Data integrity
Missing ≠ zero. Preserve AVAILABLE / MISSING / STALE / UNKNOWN / UNVERIFIABLE / INVALID / DEGRADED states. Preserve source lineage, independence groups, snapshots, schema versions, revision behavior and canonical identity. Do not double-count mirrors, wrappers or republishers as independent evidence.

## Evaluation
Use chronological walk-forward OOS/WFO. Random temporal splits are prohibited for Production-grade evidence. Separate candidate selection from final evaluation and keep frozen holdout protected from selection, calibration, routing, feature, source and research-priority tuning. Report LogLoss, Brier, Accuracy and ECE with sample size, effective sample size, confidence intervals, variance, worst/newest fold and practical effect size where applicable.

## Models / routing
Maintain simple baselines. Challengers require demonstrated incremental value, robustness and incumbent same-observation comparison. Specialist routing must have sufficient sample/fold/class support and its own PIT/OOS/robustness/holdout evidence; otherwise route to a broader validated model or fallback.

Separate Prediction Confidence from Data, Source, PIT, Model, Regime and System Confidence. Predictability is not accuracy.

## Decision layer
Supported actions include PREDICT_NOW, ACQUIRE_MORE, WAIT, RECOMPUTE, ROUTE, FALLBACK and ABSTAIN. Abstention is a valid outcome when policy says information quality is inadequate, but the abstention policy itself must be evaluated OOS.

## Research / external methods
External Web/Search/Paper/OSS/plugin methods are discovery inputs only. Translate them into reproducible GitHub-compatible implementations and validate locally. Do not transfer performance claims from other projects or external benchmarks.

Research state must distinguish DISCOVERED → SOURCE_VERIFIED → LOCALLY_REPRODUCED → OOS_CONFIRMED → ROBUST → INDEPENDENTLY_CONFIRMED → PRODUCTION_CONFIRMED.

Negative results are retained with failure conditions and reopen triggers.

## Reliability / automation
Use bounded retry/backoff, concurrency control, watchdog/heartbeat, stale-run detection, idempotent writes, checkpoints, recovery, replay, rollback and artifact preservation. Never hide failures, force success, use fail-open shell suppression, or treat cancelled/retried workflows as successful evidence.

## Cost / security
Prefer verified free, OSS, local and cached sources. Unknown-cost or billing-risk services are not automatic dependencies. Protect secrets, pin actions where appropriate, verify artifact integrity and reject production candidates with unresolved security uncertainty.

## Production state
Production is a bundle, not a model file: model artifact, feature schema, source registry version, PIT policy, target definition, calibration, router, fallback, output schema, monitoring, rollback target and manifest must remain consistent.

Block contradictory states such as PIT FAIL + PRODUCTION ACTIVE, registry/model mismatch, artifact hash mismatch, invalid holdout + promotion, stale trusted source, or snapshot mismatch.

## Completion
Green CI, an existing model, generated artifacts, or a completed workflow do not prove performance verification or Production readiness.

Completion requires evidence covering SPEC, CODE, DATA, PIT, LEAKAGE, OOS/WFO, CALIBRATION, ROBUSTNESS, HOLDOUT, REPRODUCIBILITY, RECOVERY, MONITORING, ROLLBACK, STATE CONSISTENCY, RESULT PRESENTATION and KNOWLEDGE LINEAGE.

## Status
Use IMPLEMENTED / EXECUTED / VERIFIED / PERFORMANCE_VERIFIED / PROMOTION_CANDIDATE / ADOPTED / PRODUCTION / STABLE / HOLD / REJECTED / FAILED / BLOCKED / DEFERRED / ROLLED_BACK / UNKNOWN / UNVERIFIABLE / SUPERSEDED / RETIRED distinctly.

## Loop
MONITOR → DETECT → TRIAGE → RESEARCH → IMPLEMENT → TEST → PIT → OOS/WFO → CALIBRATION → ROBUSTNESS → HOLDOUT → SHADOW → ADOPT/HOLD/REJECT → RELEASE → PRODUCTION → RECONCILE → FAILURE ANALYSIS → MEMORY → NEXT RESEARCH.


## Binary target research snapshot
The latest fixed CI run for binary_sign_v1 completed successfully. It remains research-only and did not change Production.

- 5m chronological OOS: 18,500 evaluated rows; ExtraTrees best development candidate; OOS accuracy 0.51254, logloss 0.69323, Brier 0.25004, ECE 0.00097.
- 10m chronological OOS: 18,500 evaluated rows; ExtraTrees best development candidate; OOS accuracy 0.51557, logloss 0.69285, Brier 0.24985, ECE 0.00279.
- 5m live Binance-primary strict-PIT: n=395, accuracy 0.51392, logloss 0.69166, Brier 0.24926, ECE 0.00629.
- 10m live Binance-primary strict-PIT: n=395, accuracy 0.56962, logloss 0.68836, Brier 0.24761, ECE 0.05565.
- Frozen holdout is descriptive only and remains protected from candidate selection.
- Production replacement remains prohibited; independent longer robustness, calibration and shadow evidence are still required.

## Binary research integrity additions

For binary_sign_v1, live-primary research training data must pass a knowledge-time firewall: a label is eligible only when its target/outcome timestamp is strictly before the first live prediction cutoff. Creation time alone is insufficient.

Binary chronological WFO evidence must retain a same-block frequency baseline, fold timestamps, sample size, Accuracy 95% CI, approximate effective sample size, worst/newest fold, and non-degraded fold fractions. Missing evidence is FAIL-CLOSED. The frequency baseline cannot be selected as a trainable challenger.

These controls are research-only and must not modify Production without the full local PIT → OOS/WFO → calibration → robustness → holdout → shadow → promotion sequence.


## Binary evidence lineage

Binary research evidence must record the analysis Git SHA used to generate it. CI must fail when the recorded analysis SHA does not equal the workflow's GITHUB_SHA. The later bot evidence commit is separate and does not replace the analysis SHA.

This is an evidence lineage control only; it does not promote or activate any binary target in Production.


## Prediction-confidence reliability

The existing Experience Policy OOS module is the canonical research surface for prediction-confidence reliability. It must report confidence buckets, observed accuracy, confidence gap, Accuracy 95% CI, ECE, Brier score, and an explicit 0.70+ high-confidence bucket. It also reports calibration of the learned prediction-error probability.

These are post-outcome research diagnostics only. They must not directly change Production routing, abstention, calibration, or model artifacts. High-confidence overprediction is evidence for further research, not automatic promotion or suppression.


## Frontier historical time-bound integrity

Historical backfill must not move the durable latest event-time bound backward. State updates use MIN for earliest and MAX for latest. Reversed or malformed bounds are treated as integrity issues, not silently corrected.

When bounds are invalid, the frontier loop must prioritize historical reacquisition and may bypass normal acquisition cooldown. No frontier acquisition is production-eligible. Invalid existing state remains unverified until a new acquisition provides evidence.

## Cross-project mechanism governance — 2026-10-04

The five-repository research set is used only for mechanism discovery and engineering controls. Performance metrics, OOS results, holdouts, production states, or predictions from another repository are never transferable evidence for BTC.

Reference mechanisms incorporated into BTC governance:
- 7-Sport: fail closed on pre-event enrichment failure, success-only checkpoint reuse, single-writer semantics for critical registries, and event-cluster-aware evaluation.
- Soccer: explicit prediction-cutoff lineage, optional feature-level PIT provenance, mature-prior training for temporal meta models, and fail-closed diagnostic boundaries.
- Baseball: universal dataset/source contracts, explicit readiness states, immutable prediction/experience integrity audits, and outcome-maturity-aware conformal research.
- Stock: genuinely prequential/nested model-window-ranking selection, contiguous prior-fold evidence, dependence-aware bootstrap evidence, explicit selection statistics, and run provenance manifests.

BTC adoption rule:
DISCOVER → ABSTRACT_MECHANISM → COMPATIBILITY → LOCAL_IMPLEMENTATION → TEST → LOCAL_PIT → LOCAL_OOS/WFO → ROBUSTNESS → FROZEN_HOLDOUT → SHADOW → PROMOTION.

## Feature-level PIT firewall

When feature-level provenance is supplied, feature_pit_status must be PASS, feature_snapshot_cutoff must be valid and no later than the prediction cutoff, and feature_max_available_at must be valid and no later than the prediction cutoff. Optional per-feature provenance is checked similarly. Missing optional lineage is not upgraded to PASS by inference.

## Evidence provenance binding

Production artifact audits must bind the evaluated artifact set to GITHUB_SHA (or explicitly LOCAL_UNPINNED) and hash PROJECT_INSTRUCTIONS.md, docs/PROJECT_SOURCE.md, and requirements.txt. A policy/config provenance change is an audit change even when the model files are unchanged.

## Statistical dependence and nested selection

Evaluation units must respect dependence. Multiple snapshots from one underlying event/case are not treated as independent evidence when the evaluation layer can identify the cluster. Model/router/window/return-estimator selection must be based only on outcomes available before the scored fold; global same-OOS selections cannot leak into the outer evaluation. Where applicable, contiguous prior folds and block/HAC/cluster-aware uncertainty are preferred.

## Long-running research state

Requested/queued/pending/waiting/in_progress are active transient workflow states. Only explicit terminal conclusions are evidence. Long-running jobs use checkpoints, idempotent writes, bounded retries, stable concurrency semantics, immutable research snapshots, and single-writer handling for critical state. Main-branch changes must not silently invalidate or mutate an already-started immutable research snapshot.

## Current-state discipline

The evidence snapshot in this file is descriptive and may age immediately. On every run, re-read current GitHub HEAD, current artifacts, current Actions, current registries, and current production state before making a claim or decision. Never use this instruction file's historical numbers as a substitute for fresh evidence.


## Broad Pattern Matrix Research

When asked to try many patterns, use a dedicated research-only matrix rather than changing Production. The matrix separates target/horizon, feature family, model family/parameters, training window, calibration, ensemble, routing, uncertainty/selective actions, timing/information acquisition, and source scope.

The current matrix is 10 feature sets × 8 deterministic model variants × 3 training-window policies = 240 configurations per primary horizon (5m and 10m). It screens on chronological development WFO, retains a same-block frequency baseline, applies purge/embargo, and re-evaluates a bounded finalist set with prequential temperature calibration, incumbent same-observation comparison, dependence-aware block bootstrap, effective sample size, worst/newest block and protected frozen-holdout description.

Feature patterns: all_15, returns_momentum, volatility_regime, candle_shape, volume_flow, trend, compact_cross, mean_reversion, price_structure, flow_trend.
Model patterns: logreg_c0.03, logreg_c0.1, logreg_c1.0, logreg_c3.0, extra_trees, rf, hgb, soft_ensemble.
Window patterns: expanding, recent_1500, recent_3000.
Finalist selection is multi-objective rank-based across LogLoss, Brier, accuracy stability, improved-fold ratios and worst-case block behavior, with deterministic diversity bonuses across feature pattern, model family and training window. Frozen holdout remains excluded from selection and gate.

Exploratory winners are not Production evidence. The frozen holdout never chooses a pattern, feature, model, window, calibration, router, threshold or promotion decision.

Selective prediction and timing remain separate research axes. Valid actions include PREDICT_NOW / ACQUIRE_MORE / WAIT / RECOMPUTE / ROUTE / FALLBACK / ABSTAIN. Any learned action must obey the knowledge-time maturity firewall.

