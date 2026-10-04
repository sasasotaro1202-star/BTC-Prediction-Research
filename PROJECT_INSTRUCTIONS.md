# PROJECT_INSTRUCTIONS — BTC-Prediction-Research

## 1. Mission
Optimize Future Generalization of the BTC prediction system, not historical fit alone.

Treat the system as one evidence chain:

DATA
→ TIME/PIT
→ TARGET
→ FEATURES
→ MODELS
→ ENSEMBLE
→ ROUTING
→ CALIBRATION
→ UNCERTAINTY
→ PREDICTABILITY
→ INFORMATION ACQUISITION
→ DECISION
→ PRESENTATION
→ OUTCOME
→ EXPERIENCE
→ FAILURE ANALYSIS
→ RESEARCH
→ OOS/WFO
→ ROBUSTNESS
→ FROZEN HOLDOUT
→ SHADOW
→ PROMOTION
→ PRODUCTION
→ MONITORING
→ RECOVERY
→ NEXT RESEARCH

Optimize, separately and jointly:
Future Generalization, case-level correctness, probabilistic quality, calibration, uncertainty quality, predictability awareness, OOD robustness, information value, forecast lifetime, timing, routing, fallback/abstention, compute efficiency, reproducibility and operational reliability.

Do not optimize for model complexity, source count, feature count, workflow count or green Actions by themselves.

## 2. Source of truth and every run
The canonical technical source is docs/PROJECT_SOURCE.md.

At the start of every material run:
1. Re-read PROJECT_INSTRUCTIONS.md.
2. Re-read the current canonical Project Source.
3. Inspect current GitHub default branch/HEAD, code, configuration, dependencies and workflow definitions.
4. Inspect current GitHub Actions states, jobs, artifacts and failures/cancellations/skips.
5. Inspect Production, Champion, Challenger, registry and artifact bindings.
6. Inspect data/source/PIT/OOS/calibration/robustness/holdout evidence, coverage debt and research frontier.
7. Decide the highest expected-value next action and execute it autonomously when safe.

Current GitHub state is authoritative over stale chat results and embedded historical notes.

Never rewrite mature predictions, outcomes, OOS, holdouts or failure records merely to make the current state look consistent.

Prefer:
REUSE → REPAIR → INTEGRATE → TEST → VERIFY
over creating duplicate mechanisms.

## 3. Canonical target and horizon
Primary classification:
- UP
- FLAT
- DOWN

Primary Production horizons:
- 5m
- 10m

Every target/horizon must preserve:
target_definition_version
observation_time
prediction_cutoff
horizon
label_rule
settlement_rule
revision_rule
missing_outcome_rule

New targets/horizons such as return, volatility, tail event, regime transition, predictability, failure probability or forecast lifetime are research-only until they independently pass the complete local gates.

Binary UP/DOWN (binary_sign_v1) is research-only. It must never silently replace the three-class Production target.

## 4. Time / PIT contract
Keep these times separate where relevant:
event_time
observation_time
source_publish_time
source_available_time
retrieved_time
processing_time
prediction_time
outcome_time
revision_time

Canonical timezone is UTC.

The causal PIT rule is:
source_available_time <= prediction_cutoff

retrieved_time alone is never proof of historical availability.

Unknown, missing, contradictory, malformed or unverifiable availability is:
UNKNOWN / DEFERRED / REJECTED
and never PASS by inference.

Primary readiness is horizon-specific. 5m cannot borrow PIT evidence from 10m and vice versa. The PIT artifact must contain independent primary_horizon_gate evidence for both 5m and 10m.

When feature-level provenance is supplied, also require:
feature_pit_status = PASS
feature_snapshot_cutoff <= prediction_cutoff
feature_max_available_at <= prediction_cutoff

Optional per-feature available_at records must obey the same cutoff rule. Missing optional lineage is not upgraded to PASS.

## 5. Leakage / knowledge-time firewall
Audit at minimum:
data leakage, feature leakage, target/label leakage, publication leakage, revision leakage, settlement leakage, same-event leakage, cross-fold leakage, calibration leakage, model-selection leakage, hyperparameter leakage, experiment-selection leakage, research-priority leakage, holdout leakage, cross-project leakage, knowledge-time leakage, metadata leakage and benchmark contamination.

A row's creation time is not proof that its outcome was known.

For experience-derived training, meta-models, error/failure models, predictability models, routing or calibration:
- the target/outcome must be mature before the training prediction cutoff;
- invalid/missing maturity timestamps are UNKNOWN/DEFERRED;
- outcomes settling at the same scored boundary may not train that boundary;
- same-event revisions are not independent samples.

## 6. Data / source integrity
Missing ≠ zero.

Preserve explicit states:
AVAILABLE / MISSING / STALE / UNKNOWN / UNVERIFIABLE / INVALID / DEGRADED / SOURCE_FAILED / NOT_APPLICABLE

Preserve:
source lineage, upstream owner, endpoint, source type, independence group, license, cost, coverage, freshness, availability method, publication behavior, revision behavior, schema version, parser version, latency, incremental information value and production status.

A mirror/wrapper/republisher of the same upstream is not an independent source.

Required-source partial failure must remain visible and must propagate to downstream status. A successful subset does not justify an overall success state.

Recommended source/scope readiness state:
REGISTERED
→ ADAPTER
→ PIT
→ OOS
→ ROBUST
→ FROZEN_HOLDOUT
→ SHADOW
→ PRODUCTION

Failure states remain explicit:
HOLD / REJECTED / BLOCKED / DEFERRED / FAILED / UNKNOWN

Current/future live data may be usable for current prediction without proving historical PIT. Historical OOS requires separate historical availability evidence.

## 7. Feature contract
Feature presence in source code is not proof of Production use.

The exact Production feature bundle must be reproducible by:
- ordered feature list;
- feature-set identity;
- feature manifest/policy version;
- source-set fingerprint;
- schema hash;
- target/horizon;
- model version;
- calibration/router/fallback configuration.

Use feature states:
ACTIVE
CONDITIONAL
OBSERVATION_ONLY
RESEARCH_CANDIDATE

Feature selection is a research axis. Evaluate family, subset, window, representation, aggregation, interaction, source/channel, model, training window, calibration and routing as separate or jointly prequential choices.

Feature selection is target/horizon-specific.

High-dimensional feature accumulation is not a success criterion. Prefer incremental information and source/group ablations. Retire redundant, unstable, leakage-prone, high-maintenance and low-value features.

## 8. Model ecology / routing / decision
Maintain simple baselines.

Challengers must show incremental value against the incumbent on the same chronological observations and then pass robustness and protected holdout evidence.

Specialist routing requires adequate sample/fold/class support, PIT-safe context, calibration and recent stability. Otherwise:
ROUTE → broader validated model → FALLBACK → ABSTAIN/DEFER

Allowed decisions:
PREDICT_NOW
ACQUIRE_MORE
WAIT
RECOMPUTE
ROUTE
FALLBACK
ABSTAIN

Abstention is a valid decision, but the abstention policy must itself be evaluated OOS.

Separate:
Prediction Confidence
Data Confidence
Source Confidence
PIT Confidence
Model Confidence
Regime Confidence
System Confidence

Predictability is a separate concept from probability confidence.

Track, where computable:
model disagreement, OOD, regime stability, recent error, calibration stability, data quality, source reliability and forecast lifetime.

Do not fabricate Predictability/OOD/Failure Risk/Delta when runtime does not compute them. Use explicit NOT_COMPUTED/NOT_AVAILABLE states.

## 9. Calibration
Calibration is chronological.

Never tune calibration on the same outer OOS block being scored.

Calibration artifacts must bind to the exact model artifact/generation by model version and model hash.

Calibration failure is a safety failure, not a cosmetic metric issue.

## 10. OOS / WFO / statistical integrity
Production-grade evaluation is:
Chronological
→ Walk-forward
→ PIT-safe
→ Candidate-selection separated
→ Frozen-holdout separated

Random temporal splits are not Production-grade evidence.

Nested/prequential selection rule for every outer fold:
1. use strictly earlier outcomes to select model/window/features/router/calibration/weighting;
2. freeze the selected configuration;
3. score the untouched current fold;
4. only then expose that fold's outcome to later selection.

Where candidate windows are compared, prefer contiguous prior-fold support.

Retain for each material candidate:
train_end, test_start, test_end, test_n, baseline, candidate metrics, effect size, confidence interval, effective sample size and selection statistics.

Respect dependence. Multiple snapshots from one underlying market event are not automatically independent evidence. Use event/case cluster identifiers where available and use HAC, moving-block bootstrap or cluster-aware methods where appropriate.

Maintain multiple-comparison awareness and record degrees of freedom:
candidate count, feature trials, source trials, hyperparameter trials, calibration trials, router/timing trials and selection iterations.

## 11. Adoption gates
Reference gates are evidence thresholds, not automatic promotion rules.

Typical reference:
- primary OOS relative improvement ≥3%;
- auxiliary improvement ≥1%;
- ≥70% non-degraded evaluation periods;
- newest protected evaluation non-degraded;
- no material calibration degradation;
- zero active PIT violations.

Also evaluate:
sample size, dependence, ESS, CI, effect size, regime concentration, robustness, implementation risk, complexity and operational failure surface.

Production requires:
PIT PASS
Leakage PASS
Reproducibility
Robustness
Incumbent same-observation comparison
Calibration safety
Newest/frozen holdout safety
Independent confirmation
Explicit promotion

No single fold, benchmark, live streak or external claim is sufficient.

## 12. Holdout firewall
Frozen holdout may not be used for:
model selection
feature selection
hyperparameter tuning
source selection
router tuning
calibration tuning
timing optimization
research prioritization
adoption winner selection

Holdout access must be auditable.

A descriptive holdout score does not authorize promotion.

## 13. Research handoff / state machine
Expensive OOS may start only with explicit:
TESTS_PASSED = true
AUDIT_PASSED = true

Missing/invalid handoff is BLOCKED.

Research status:
DISCOVERED
→ SOURCE_VERIFIED
→ LOCALLY_REPRODUCED
→ OOS_CONFIRMED
→ ROBUST
→ INDEPENDENTLY_CONFIRMED
→ PRODUCTION_CONFIRMED

Code-exists ≠ implemented
Implemented ≠ executed
Executed ≠ verified
Verified ≠ performance_verified
Performance_verified ≠ promotion
Promotion ≠ production
Green CI ≠ research success

## 14. Evidence lineage / freshness
Every material result should retain:
Git SHA
ref
environment/dependencies
configuration
seed
data snapshot/hash
feature schema/hash
source registry/version
target version
model artifact/hash
calibration artifact/hash
router/fallback version
holdout policy
experiment fingerprint

Production Artifact Audit must bind its evidence to:
GITHUB_SHA (or LOCAL_UNPINNED)
GITHUB_REF_NAME
PROJECT_INSTRUCTIONS.md SHA256
docs/PROJECT_SOURCE.md SHA256
requirements.txt SHA256
model and metadata SHA256
runtime reload result
feature schema

Any change capable of altering PIT semantics, target/labeling, feature schema, scoring, model selection, calibration, routing, adoption, production artifact identity or policy provenance is evidence-affecting and must invalidate stale evidence as appropriate.

Later evidence-publication commits do not replace the analysis SHA of the code that actually ran the experiment.

## 15. Workflow / automation reliability
Distinguish:
REQUESTED
QUEUED
PENDING
WAITING
IN_PROGRESS
SUCCESS
FAILURE
CANCELLED
SKIPPED

Only explicit terminal success is execution evidence.

Use:
checkpoint
resume
idempotency
bounded retry/backoff
stable concurrency
single-writer critical state
deterministic writes
watchdog
heartbeat
stale-run detection
immutable research snapshots
replay
recovery
rollback

Success-only checkpoint reuse is required. Failed/partial checkpoints are not treated as completed evidence.

Long-running capture/research must not be cancelled by ordinary main commits when its immutable snapshot remains valid. Avoid duplicate long-running collectors; serialize or otherwise coordinate durable writers.

Stale remote state must never be overwritten by an older local snapshot. Compare local and remote generation/end-time/version before publish and re-check after conflicts.

## 16. Safe degradation / recovery
When a critical dependency is unavailable:
FULL
→ REDUCED
→ FALLBACK
→ SELECTIVE
→ ABSTAIN
→ RECOVERY

PIT failure is not the same as ordinary data availability degradation.

Required recovery evidence:
checkpoint_id
stage
scope
input snapshot
completed outputs
pending work
expected next state
artifact hashes
recovery safety

Recovery must be bounded and observable.

## 17. Production bundle / state consistency
Production is a bundle, not a model file.

The bundle must reconcile:
model
feature schema
source registry
PIT policy
target definition
calibration
router
fallback
output schema
monitoring
rollback target
manifest

Block contradictions such as:
PIT FAIL + PRODUCTION ACTIVE
registry/model mismatch
model hash mismatch
invalid calibration binding
invalid holdout + promotion
stale trusted source
snapshot mismatch
candidate artifact exposed as Production

## 18. Experience / result presentation
Prediction revisions are append-only.

Store:
prediction_before
prediction_after
delta
reason
changed sources/features/model/regime/calibration

Outcome maturity must be verified before Experience scoring.

Historical/research and current Production views must remain separate.

Current result presentation should expose, when available:
Target
Horizon
UP/FLAT/DOWN probabilities
Top prediction
Prediction State
Decision
Reliability
Prediction Cutoff
Data As-of
Data Age
Model
Ensemble
Model Agreement
Predictability
OOD
Failure Risk
Data Reliability
Source Health
PIT
Forecast Lifetime
Previous → Current Delta
Change Drivers
Prediction ID
Experiment ID
Model Version
Data Snapshot
Feature Schema
Git SHA
PIT result
Leakage result
Reproducibility

Do not infer missing fields.

## 19. Research prioritization / failure frontier
Choose the next research task from:
expected OOS gain
information gain
failure reduction
coverage debt
urgency
novelty
transferability
cost
runtime
reproducibility
complexity
failure surface
risk

Frontiers:
DATA
SOURCE
FEATURE
MODEL
ROUTING
CALIBRATION
TIMING
TARGET
REGIME
UNCERTAINTY
FAILURE
INFORMATION ACQUISITION
UNKNOWN-UNKNOWN

Prefer a small number of high-information experiments over large uncontrolled search.

Retain negative knowledge:
hypothesis
scope
data
failure
failed conditions
reason
confidence
reopen trigger

Stop/defer duplicated or low-yield research.

## 20. Cost / security
Priority:
Verified Free
→ Free Quota
→ OSS/local
→ Cached
→ Lightweight compute

Paid-only, billing-risk, unknown-cost, auto-renew trial or quota-overage dependencies are not automatically adopted.

Do not expose secrets in code, logs, artifacts, reports, commits or test fixtures.

Review:
workflow permissions
action pinning
dependency/supply-chain risk
artifact tampering
license/redistribution restrictions
rate limits
source retention

## 21. Cross-project mechanism transfer
Reference repositories:
- Baseball-Prediction-System
- BTC-Prediction-Research
- 7-Sport-Prediction-Research
- Soccer-Prediction-Research
- Stock-Daily-Prediction-3000

Only mechanisms transfer, never performance evidence, holdouts, production states, predictions or datasets.

Mechanisms incorporated as references:
7-Sport:
- explicit fail-closed enrichment semantics
- success-only checkpoint discipline
- critical-state single-writer semantics
- event-aware evaluation

Soccer:
- explicit prediction-cutoff lineage
- optional feature-level PIT provenance
- mature-prior temporal meta-learning
- fail-closed source/diagnostic states
- nested feature-selection contract

Baseball:
- universal data/source contract
- explicit readiness states
- immutable prediction/experience audit
- source quality separated from identity coverage
- outcome-maturity-aware conformal/uncertainty research

Stock:
- nested/prequential model-window-ranking selection
- contiguous prior-fold evidence
- dependence-aware moving-block bootstrap
- multiple-comparison-aware selection statistics
- heartbeat/transient-state handling
- run provenance manifest

BTC adoption path:
DISCOVER
→ ABSTRACT_MECHANISM
→ COMPATIBILITY
→ LOCAL_IMPLEMENTATION
→ TEST
→ LOCAL_PIT
→ LOCAL_OOS/WFO
→ ROBUSTNESS
→ LOCAL_FROZEN_HOLDOUT
→ SHADOW
→ PROMOTION

Cross-project success never bypasses a BTC-local gate.

## 22. Current-state discipline
Volatile metrics and Action statuses are not authoritative because this file is static.

On every run, read the live artifacts instead:
- data/live_cycle_status.json
- data/historical_research/production_integrity.json
- data/historical_research/pit_oos_audit.json
- data/historical_research/research_input_audit.json
- data/historical_research/research_readiness.json
- data/historical_research/performance_snapshot.json
- data/historical_research/performance_change.json
- data/historical_research/robustness_oos_report.json
- data/historical_research/promotion_gate.json
- relevant model/registry/source manifests
- current GitHub Actions state

Never use historical numbers embedded in instructions as a substitute for fresh state.

## 23. Completion definition
Completion is not:
- green Action
- code existence
- generated artifact
- model-file existence
- one good live streak
- one good fold
- external benchmark superiority

Completion requires evidence appropriate to the change across:
SPEC
CODE
DATA
PIT
LEAKAGE
OOS/WFO
CALIBRATION
ROBUSTNESS
FROZEN_HOLDOUT
REPRODUCIBILITY
RECOVERY
MONITORING
ROLLBACK
STATE_CONSISTENCY
RESULT_PRESENTATION
KNOWLEDGE_LINEAGE

Unverified requirements remain UNKNOWN/UNVERIFIABLE/BLOCKED/HOLD.

## 24. Status taxonomy
Use these distinctly:
IMPLEMENTED
EXECUTED
VERIFIED
PERFORMANCE_VERIFIED
PROMOTION_CANDIDATE
ADOPTED
PRODUCTION
STABLE
HOLD
REJECTED
FAILED
BLOCKED
DEFERRED
ROLLED_BACK
UNKNOWN
UNVERIFIABLE
SUPERSEDED
RETIRED

## 25. Permanent operating loop
MONITOR
→ DETECT
→ TRIAGE
→ RESEARCH
→ IMPLEMENT
→ TEST
→ PIT
→ OOS/WFO
→ CALIBRATION
→ ROBUSTNESS
→ FROZEN_HOLDOUT
→ SHADOW
→ ADOPT/HOLD/REJECT
→ RELEASE
→ PRODUCTION
→ RECONCILE
→ FAILURE ANALYSIS
→ MEMORY
→ NEXT RESEARCH

The top-level objective remains:
PIT integrity > apparent backtest gain
Future generalization > historical fit
Case-level correctness > aggregate-only optimization
Calibration > raw confidence
Independent evidence > source count
Robustness > single-period improvement
Failure learning > repeated failure
Safe degradation > forced prediction
Reproducibility/state consistency > convenient output
Information efficiency > complexity
