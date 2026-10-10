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
The latest verified PIT evidence artifact was generated at 2026-10-08T05:34:26Z. Because this instruction file is mutable, the artifact values below are a descriptive snapshot only; every run must re-read the current artifact before making a decision.

The strict PIT artifact reports checked_predictions = 814, verified_predictions = 645, verified_primary_predictions = 478, verified_fallback_predictions = 167, active current-scope PIT violations = 0, and legacy_unverified_count = 169 with legacy_violation_count = 41 quarantined. The independent primary horizon gates are 5m = 475 and 10m = 476, both ready above the 300-row minimum. Situation-metadata readiness is 471 for 5m and 472 for 10m; the artifact estimates 2,529 and 2,528 additional qualifying rows respectively to reach the 3,000-row maturity target.

Robustness evidence remains insufficient for Production promotion: live-Binance-primary robustness sample size is 332 for 5m and 476 for 10m versus the 1,000-row minimum; promotion_evidence_eligible = false for both horizons.

Calibration remains unverified: the 5m calibration artifact has n_settled = 332 with missing fit/holdout LogLoss evidence; the 10m artifact has n_settled = 476 with fit_logloss = 1.0508521665801227 and holdout_logloss = 1.103117546942235. Accordingly the current promotion gate remains `HOLD` with `promotion_allowed = false`.

Current production artifact bindings remain:
- 5m: `bootstrap.soft_ensemble.v5.4`
- 10m: `bootstrap.bootstrap_rf`

These are production artifact bindings, not evidence of superiority. Runtime must continue verifying registry ↔ metadata ↔ artifact ↔ feature schema before use.

## PIT / time
Keep event/observation/publish/available/retrieved/processing/prediction/outcome/revision times separate. Historical feature availability must be proven with `source_available_time <= prediction_cutoff`. Retrieved-at alone is not PIT evidence. Unknown or unverifiable availability is UNKNOWN/DEFERRED/REJECTED, never PASS.

For primary readiness, strict PIT evidence is gated independently per production horizon. The PIT artifact must provide `primary_horizon_gate` entries for both 5m and 10m, and each entry must report `strict_primary_settled >= min_strict_pit_rows`, `minimum >= min_strict_pit_rows`, and `ready = true`. Aggregate verified-primary counts cannot substitute for a missing or failing horizon-specific gate.
Promotion safety uses the same independent horizon rule: `promotion_allowed` must remain false unless both 5m and 10m `primary_horizon_gate` entries are present, ready, and above the strict minimum. Aggregate verified-primary counts are never sufficient for Promotion.

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

Production-first backpressure: when the 5m Live Cycle is queued, pending or in progress, the Continuous Supervisor must not dispatch an additional evidence-routed research lane. The selected research candidate remains observable and is reconsidered on a later heartbeat after Production capacity clears. This is an operational queue-control rule only and never changes research priority or Production promotion policy.

Use bounded retry/backoff, concurrency control, watchdog/heartbeat, stale-run detection, idempotent writes, checkpoints, recovery, replay, rollback and artifact preservation. Never hide failures, force success, use fail-open shell suppression, or treat cancelled/retried workflows as successful evidence.

24H research lineage rule: an active 24H research snapshot remains valid across main-branch state-only drift when changed paths are limited to `data/**` and `models/*.json` / `models/*.joblib`. Any workflow, source, test, dependency, configuration, instruction, or Project Source change invalidates that immutable snapshot and must fail-closed and recover through the watchdog. The research workspace remains bound to its original SHA.

## Cost / security
Prefer verified free, OSS, local and cached sources. Unknown-cost or billing-risk services are not automatic dependencies. Protect secrets, pin actions where appropriate, verify artifact integrity and reject production candidates with unresolved security uncertainty.

## Recent-performance recovery trigger
A primary-horizon recent window with at least 100 settled cases and Accuracy <= 0.35 is a research-only degradation trigger even when the previous performance snapshot reports changed=false. A recent-window Accuracy gap of at least 5 percentage points below the same horizon's total experience Accuracy is an additional research-only trigger. These triggers route to experience-policy research and a recency-weighted challenger lane; they must never directly alter Production. The 35% floor and 5-point gap are recovery triggers, not adoption criteria.

## Production state
Production is a bundle, not a model file: model artifact, feature schema, source registry version, PIT policy, target definition, calibration, router, fallback, output schema, monitoring, rollback target and manifest must remain consistent.

Block contradictory states such as PIT FAIL + PRODUCTION ACTIVE, registry/model mismatch, artifact hash mismatch, invalid holdout + promotion, stale trusted source, or snapshot mismatch.

Bootstrap training is research-only. It may evaluate a historical seed candidate, but it must never overwrite an existing Production model or model_registry entry. Live scheduled workflows must not use model age alone as a Production refresh trigger; Production replacement requires the full local OOS/WFO → calibration → robustness → frozen holdout → shadow → promotion sequence.

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
Finalist selection is multi-objective rank-based across LogLoss, Brier, accuracy stability, improved-fold ratios and worst-case block behavior, with deterministic diversity bonuses across feature pattern, model family and training window. Frozen holdout remains excluded from selection and gate. Long-running matrix execution checkpoints each completed screen/finalist, binds resumption to the exact analysis Git SHA, data fingerprint and candidate manifest, and restores prior-run checkpoint artifacts only when those identities match. Only successfully evaluated screens/finalists are marked reusable; failed/errored candidates remain explicit failures and are not reused as successful completion on a later same-SHA resume.

Exploratory winners are not Production evidence. The frozen holdout never chooses a pattern, feature, model, window, calibration, router, threshold or promotion decision.

Selective prediction and timing remain separate research axes. Valid actions include PREDICT_NOW / ACQUIRE_MORE / WAIT / RECOMPUTE / ROUTE / FALLBACK / ABSTAIN. Any learned action must obey the knowledge-time maturity firewall.

## GitHub-side autonomous research routing

The BTC Continuous Supervisor invokes `src/autonomous_research_router.py` every 5-minute heartbeat and evaluates an ordered list of research-only candidates from an explicit allowlist. It dispatches at most one stale candidate per heartbeat. Durable PIT/health failures are hard stops; otherwise insufficient current-generation calibration evidence is prioritized, followed by robustness/holdout research, confidence-reliability research, material drift/model-disagreement uncertainty research, selective-prediction research, eligible data-frontier work, and finally bounded V13 refresh. The Supervisor must also be event-wakeable from completed Production/data/research workflows so GitHub Actions schedule delays do not become a single point of failure; duplicate wakeups are collapsed by the Supervisor concurrency guard.

A higher-priority candidate being active or non-stale must not starve lower-priority research. The Supervisor therefore falls through the ordered candidate list and selects the first dispatchable lane. Router failure remains fail-closed to the read-only readiness audit, and that fallback is represented as a valid candidate in the route artifact.

High-confidence overprediction is a research trigger only. The trigger uses the mature `0.70+` bucket with at least 50 settled cases and confidence-minus-accuracy gap at least 0.15 in either primary horizon. It cannot directly suppress, recalibrate, route, or promote Production.
Material drift is also a research trigger only. Durable research-only drift evidence at or above the configured drift/model-disagreement threshold may prioritize `btc_uncertainty_layer_oos.yml`; it cannot directly change Production.

Router output is validated for allowlist membership, exact workflow thresholds, ordered priorities, unique candidates, production_impact=false, evidence_state and signal structure. The supervisor artifact records the first candidate, selected candidate, dispatch result and route signals. The router never selects a Production model or changes the Production registry. Existing PIT, OOS/WFO, calibration, robustness, frozen-holdout, shadow, promotion, recovery and rollback gates remain mandatory. No paid API or external LLM service is required.


## GitHub-side research PR merge automation

Autonomous engineering PRs may be merged by the repository workflow only when they are explicitly marked with `<!-- btc-automerge:research-only -->`, target `main`, originate from the same repository, use an allowed engineering branch prefix, and contain no Production/data/registry/holdout state changes. The merge gate evaluates the current PR HEAD and refuses to merge while any current-head check or commit status is pending/failed. It merges with the expected HEAD SHA using squash semantics so a race cannot merge a superseded commit.

The auto-merge workflow never checks out or executes PR code, keeps Production mutation paths fail-closed, uploads an audit artifact, and treats merge rejection as HOLD rather than success. It does not bypass PIT/OOS/calibration/robustness/holdout/shadow/promotion policy. Research PRs without the explicit marker remain manual by design.


The research PR auto-merge firewall treats every `.github/workflows/*` change as sensitive and requires manual review; the auto-merge gate itself is therefore not self-modifiable by an auto-merged research PR.

## Conditional return-distribution / tail research lane

A research-only return-distribution lane is part of the autonomous research queue. It uses only matured, strict-PIT Binance-primary prediction observations from the canonical prediction ledger and estimates conditional q10/q50/q90 endpoint returns with chronological walk-forward evaluation plus purge/embargo.

The lane reports pinball loss, 80% interval coverage, tail-breach rates, interval width, median-sign accuracy, newest-block behavior and a training-window quantile baseline. It must remain research_only=true, production_changed=false, and promotion_evidence_eligible=false. It does not claim intrahorizon drawdown because the current research contract does not persist the full future path.

Evidence is bound to both analysis_git_sha and a prediction-DB snapshot hash. The Continuous Supervisor treats the lane as an allowlisted, non-production research candidate, dispatches it when its evidence is missing/stale, and wakes again after its completion. This lane must never bypass the existing PIT → OOS/WFO → calibration → robustness → frozen holdout → shadow → promotion sequence.

## Immutable research snapshot recovery

Long-running research workflows must remain valid when `main` advances while they are queued or executing. The analysis is bound to its immutable `GITHUB_SHA` and the prediction-DB snapshot used for that analysis; a concurrent main-branch state commit must not invalidate the already-started research run. Before committing research evidence, the workflow must reconcile against the current `main` and preserve the analysis SHA as provenance. A stale-head or concurrent state commit is not itself a research failure. Only actual execution, integrity, PIT, OOS, or boundary failures are evidence of failure. The dedicated 24H autonomous marathon is a stricter recovery case: its Stage 1 must start from current `main`, and an active 24H run whose `head_sha` is no longer current `main` is treated as an obsolete generation and may be cancelled by the dedicated 24H Watchdog. The Watchdog then dispatches a fresh current-`main` generation and verifies creation. This changes runner/recovery behavior only; obsolete or cancelled runs remain non-evidence and are never silently promoted.


## Automation recovery backoff
The Continuous Supervisor and Workflow Watchdog apply bounded exponential backoff to repeated non-success generations. The base cooldown is 5 minutes and is capped at 80 minutes. This prevents deterministic failures from creating an unbounded GitHub Actions dispatch storm while preserving automatic recovery. The stale-run janitor is cancellation-only for the production/research spine; recovery dispatch is centralized in the Supervisor, Watchdog and Production Sentinel.



## Workflow contract validation lane
Full unit tests run on source, test, script, and dependency changes. GitHub Actions workflow changes are validated by a lightweight dedicated contract lane, including compile checks and recovery/automation contract tests. This avoids repeatedly queueing the full suite for orchestration-only edits while preserving fail-closed validation of the automation control plane.


## Shared recovery failure-streak helper
Supervisor and Watchdog must source `scripts/ci_failure_streak.sh` for the failure-streak calculation. The helper counts only consecutive non-success terminal generations since the most recent explicit success and stops at the first neutral/unknown terminal conclusion. The helper is independently unit-tested and syntax-checked by Ops Preflight. Duplicate inline jq recovery logic is prohibited.


104. CURRENT RESEARCH-WATCHDOG LIVENESS ADDENDUM

The research control plane must not depend on a single dispatcher for degradation recovery. The independent Watchdog therefore tracks the research-only Recency Challenger and Broad Pattern Matrix lanes in its bounded recovery inventory.

Recency Challenger:
- stale threshold: 86400 seconds
- active grace: 14400 seconds, aligned with its 240-minute workflow timeout
- research_only=true
- production_changed=false
- no Promotion authority

Broad Pattern Matrix:
- stale threshold: 86400 seconds
- active grace: 7200 seconds, aligned with its 120-minute workflow timeout
- before expensive computation, the workflow compares its execution SHA with current main
- only durable state-only drift is tolerated; code/config/workflow/policy drift is FAIL-CLOSED
- same-SHA checkpoints remain reusable only under the existing checkpoint contract

These controls are liveness/reproducibility controls only. They do not change model selection, Production, or Frozen Holdout evidence.


105. PERFORMANCE-REGRESSION ROUTING PRIORITY

When durable post-outcome evidence shows material recent performance regression, the recovery lanes are intentionally prioritized below calibration-collection hard safety work but above routine robustness/holdout refresh, tail diagnostics, generic uncertainty research, and routine frontier work.

Priority ordering for this condition:
1. btc_adaptive_calibration_replay.yml = 95 when calibration evidence is incomplete
2. btc_experience_policy_oos.yml = 94 when material performance regression is detected
3. btc_recency_challenger.yml = 93 on the same trigger

The higher priority only changes research scheduling. It does not select, promote, replace, or modify a Production model, registry, calibration artifact, router, or Frozen Holdout.


106. RECENCY CHALLENGER PIT / PROMOTION BOUNDARY

The Recency Challenger is an exploratory historical model-comparison lane. Historical candle rows are event-time aligned, but source publication/availability time is not proven from the archive loader. Therefore its artifacts must declare `pit_status=UNVERIFIABLE_HISTORICAL_AVAILABILITY`, `promotion_evidence_eligible=false`, and `promotion_allowed=false`.

The workflow validates current-main lineage before expensive computation. Durable data/model state drift may be tolerated under the existing long-run policy; executable/config/workflow/policy drift is FAIL-CLOSED.

A positive holdout result from this lane is a hypothesis/research signal only. It cannot become Production evidence without an independent strict-PIT implementation on the required current-generation observations.


107. PIT AUDIT SAFETY PRIORITY UNDER PRODUCTION BACKPRESSURE

The PIT OOS Audit is a production-safety prerequisite rather than optional research. The Continuous Supervisor must recover it independently before applying Production-first suppression to discretionary research lanes.

Operational rule: `dispatch_if_stale btc_pit_oos_audit.yml 900` runs before the `production_active` backpressure branch. This keeps the PIT audit freshness gate current without dispatching model-research lanes during an active Live Cycle.

The independent Watchdog already retains a separate bounded PIT audit recovery path. This is redundancy for audit liveness, not an authorization to modify Production.


107. RECENCY RECOVERY CADENCE

When durable recent-performance regression is active, the routed Recency Challenger uses a bounded 6-hour freshness threshold instead of its routine 24-hour threshold. This reduces recovery latency while retaining the existing concurrency, current-main, Watchdog, and fail-closed controls.

The 6-hour threshold applies only to evidence-driven recovery routing. Routine scheduled Recency Challenger execution remains daily.

No Production, registry, calibration, ledger, Frozen Holdout, or promotion authority is changed.


108. RECENCY VALIDATION SELECTION ABLATION

The Recency Challenger exploratory candidate is selected by equal rank across validation Accuracy, LogLoss, and Brier on the same chronological validation slice. The prior Accuracy-first selection is retained for auditability so the selection effect can be compared directly.

Frozen Holdout remains descriptive-only and cannot tune the selection rule. The challenger also runs automatically when its research source/workflow/tests change, while ordinary data-state updates do not trigger the expensive lane.

This is an evidence-generation change only. A positive result is not Production evidence without independent strict-PIT, OOS/WFO, robustness, calibration, Frozen Holdout, and shadow validation.


109. DEFERRED PIT AUDIT REFRESH CONTRACT

A safely deferred Live Cycle does not create a new market prediction, but settlement of previously created predictions can still advance while the cycle is deferred. Therefore a deferred cycle must not permanently suppress the PIT/OOS audit.

During a deferred cycle:
- if the persisted PIT/OOS audit is fresh, the cycle may skip a redundant re-audit;
- if the audit is missing, malformed, future-dated, or older than 900 seconds, the cycle reruns the read-only PIT/OOS audit against the existing immutable prediction ledger;
- no market snapshot, prediction, or Production change is fabricated by the refresh;
- an unknown refresh-policy result is FAIL-CLOSED.

This policy exists to keep the Continuous Supervisor's PIT freshness hard-stop synchronized with the actual prediction/settlement ledger while avoiding a new PIT-state commit every five minutes when no new audit is needed.

The refresh decision is implemented by `src/pit_deferred_audit_policy.py` and covered by unit tests. The policy changes audit liveness only; it does not relax PIT validity, OOS/WFO, calibration, robustness, holdout, shadow, or promotion requirements.
