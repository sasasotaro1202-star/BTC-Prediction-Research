BTC-Prediction-Research

PROJECT SOURCE

ULTIMATE FINAL / CANONICAL TECHNICAL SOURCE

TARGET REPOSITORY:
https://github.com/sasasotaro1202-star/BTC-Prediction-Research

⸻

0. SOURCE ROLE

本SourceはBTC Projectの詳細・永続仕様である。

現在のGitHub実装、registry、artifact、workflow、validated evidenceを一次情報として扱い、古い情報を盲目的に維持しない。

仕様変更は、
EVIDENCE
→ IMPACT CHECK
→ IMPLEMENTATION CHECK
→ VALIDATION
→ VERSION UPDATE
の順で行う。

過去のprediction・OOS・holdout・outcomeを書き換える目的でSourceを変更してはならない。

⸻

1. SYSTEM MISSION

最終システムは単なるBTC price predictorではない。

以下を一体として扱う。

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
→ RESEARCH PRIORITIZATION
→ OOS
→ ROBUSTNESS
→ PROMOTION
→ PRODUCTION
→ MONITORING
→ RECOVERY
→ REDISCOVERY

目的は将来未知状態へのgeneralizationを最大化し、不要な予測・誤った確信・リーケージ・無駄な計算を最小化すること。

⸻

2. CURRENT REPOSITORY INTEGRATION

既存Repositoryには多数のResearch / Watchdog / PIT / OOS / Calibration / Selective / Uncertainty / Regime / Microstructure / Recency / Production / Recovery機構が存在する。

代表的な既存機構:

* btc_24h_autonomous_research
* btc_24h_watchdog
* btc_9h_research_marathon
* btc_adaptive_calibration_replay
* btc_autonomous_data_frontier
* btc_binance_flow_research
* btc_binance_ws_collector
* btc_calibration_frozen_replay
* btc_canonical_state_compaction
* btc_class_prior_recalibration_oos
* btc_continuous_supervisor
* btc_current_production_prediction
* btc_dual_memory_oos
* btc_exogenous_research
* btc_experience_policy_oos
* btc_frontier_branch_validation
* btc_frozen_archive_replay
* btc_historical_adapter_validation
* btc_innovative_prediction_control_v2
* btc_integrity
* btc_interaction_research
* btc_live_cycle
* btc_live_heartbeat_guard
* btc_maximum_future_generalization_v6
* btc_microstructure_research
* btc_multiscale_frozen_replay
* btc_ops_preflight
* btc_pit_oos_audit
* btc_production_sentinel
* btc_public_venue_shadow
* btc_recency_challenger
* btc_recency_research
* btc_rich_production_challenger
* btc_rolling_challenger
* btc_selective_prediction_oos
* btc_selective_research
* btc_stale_run_janitor
* btc_time_regime_research
* btc_ultimate_final_v13_e2e
* btc_uncertainty_layer_oos
* btc_unit_tests
* btc_watchdog
* btc_x_research_audit

同等機能を新規作成する前に、
既存機能の再利用・統合・修正・廃止可能性を検討する。

⸻

3. CANONICAL TARGET

Primary classification:

UP
FLAT
DOWN

Primary horizons:

5m
10m

Target semantics:

target_definition_version
observation_time
prediction_cutoff
horizon
label_rule
settlement_rule
revision_rule
missing_outcome_rule

を必ずversion管理する。

将来candidateとして、

RETURN
VOLATILITY
TAIL EVENT
REGIME TRANSITION
PREDICTABILITY
FAILURE PROBABILITY
FORECAST LIFETIME

等を研究してよいが、既存targetと混同しない。

⸻

4. TIME MODEL

時刻は単一fieldにまとめない。

最低限:

event_time
observation_time
source_publish_time
source_available_time
retrieved_time
processing_time
prediction_time
outcome_time
revision_time

を保存する。

UTCをcanonicalとする。

Exchange/local timezoneは必要に応じ補助情報として保存。

⸻

5. PIT CONTRACT

Historical featureがpredictionに利用可能だったことを証明できる必要がある。

原則:

source_available_time <= prediction_cutoff

だけをPIT availabilityの基準とする。

retrieved_time <= cutoffだけではPIT PASSにしない。

publication bufferは補助的安全策であり、
実際のavailability証明と同一視しない。

PIT unknown:
→ UNKNOWN / DEFERRED / REJECTED

⸻

6. LEAKAGE TAXONOMY

監査対象:

DATA LEAKAGE
FEATURE LEAKAGE
TARGET LEAKAGE
LABEL LEAKAGE
PUBLICATION LEAKAGE
REVISION LEAKAGE
SETTLEMENT LEAKAGE
SAME-EVENT LEAKAGE
CROSS-FOLD LEAKAGE
CALIBRATION LEAKAGE
MODEL-SELECTION LEAKAGE
HYPERPARAMETER LEAKAGE
EXPERIMENT-SELECTION LEAKAGE
RESEARCH-PRIORITY LEAKAGE
HOLDOUT LEAKAGE
CROSS-PROJECT LEAKAGE
KNOWLEDGE-TIME LEAKAGE
METADATA LEAKAGE
BENCHMARK CONTAMINATION

⸻

7. KNOWN PIT RISK REGISTER

既知のリスク:

1. historical dataにfeature-level available_at/publication/retrieval/revision provenanceが不足する可能性
2. prediction-level timestampだけの監査では全inputのPITを証明できない
3. fetch failureを0に置換すると欠損を情報として誤認する可能性
4. settlement venue fallbackでlabel意味が変わる可能性
5. publication bufferは真のhistorical availabilityと同一ではない
6. Binance-derived series間に独立性がない可能性
7. production artifactのfull reloadとmetadata exact-matchが不十分な可能性

これらは解消が証明されるまでOPEN/DEFERRED等で管理する。

⸻

8. DATA QUALITY CONTRACT

Data completenessをrow countで定義しない。

評価軸:

ENTITY COVERAGE
EVENT COVERAGE
OUTCOME COVERAGE
FEATURE COVERAGE
TIMESTAMP COVERAGE
AVAILABLE-AT COVERAGE
PIT COVERAGE
IDENTITY COVERAGE
SOURCE COVERAGE
SOURCE INDEPENDENCE
FRESHNESS
REVISION AWARENESS
SCHEMA INTEGRITY
DUPLICATE INTEGRITY
MISSINGNESS
RECONCILIATION
REPRODUCIBILITY

critical production inputでは、

UNKNOWN AVAILABILITY
INVALID TIMESTAMP
UNRESOLVED IDENTITY
DUPLICATE
IMMATURE OUTCOME
PIT FAILURE

を許容しない。

⸻

9. MISSING DATA POLICY

Missingは0ではない。

状態:

AVAILABLE
MISSING
STALE
UNKNOWN
UNVERIFIABLE
INVALID
DEGRADED

を分ける。

Modelがmissing valueを扱う場合でも、
「欠損だった」という事実が保持される。

critical information missing時は、

RECOMPUTE
FALLBACK
ABSTAIN
BLOCK

のいずれかを選択する。

⸻

10. SOURCE REGISTRY

各sourceに:

source_id
upstream_owner
endpoint
source_type
independence_group
license
cost_status
coverage
freshness
availability_method
publication_method
revision_behavior
schema_version
parser_version
last_success
last_failure
failure_reason
PIT_strength
latency
incremental_information_value
production_status

を持たせる。

mirror / wrapper / republisher / archiveはupstream lineageを追跡し、独立source数を水増ししない。

⸻

11. SOURCE GRAPH

SourceはGraphとして管理する。

UPSTREAM
→ DATASET
→ MIRROR
→ WRAPPER
→ FEATURE
→ MODEL

同じroot upstreamに依存するデータを独立証拠として二重計上しない。

Source disagreementは、
failureだけでなくinformation conflictとして記録する。

⸻

12. FEATURE LINEAGE

Feature-level provenanceを可能な限り保持する。

feature
source_id
source_record_id
available_at
retrieved_at
transform_id
transform_version
aggregation_window
normalization
revision_state

を追跡する。

⸻

13. SNAPSHOT CONTRACT

Research snapshotはimmutable。

minimum:

snapshot_id
creation_time
data_range
source_versions
schema_hash
content_hash
manifest_hash
row_count
coverage_summary

Correctionは新Snapshotとして保存する。

旧Snapshotを上書きしない。

⸻

14. ENTITY / VENUE INTEGRITY

asset
venue
instrument
feed
source-specific identifier

を分離する。

canonical identity mappingを保存し、
ambiguous mappingはsilent mergeしない。

⸻

15. TARGET / LABEL IMMUTABILITY

一度成熟したhistorical labelを後知恵で置き換えない。

訂正が必要なら、

original outcome
revision
final outcome

をversion chainとして残す。

⸻

16. RESEARCH EXPERIMENT SCHEMA

Experimentごとに:

experiment_id
hypothesis
research_question
scope
target_definition
horizon
data_snapshot
source_set
feature_set
model
hyperparameters
router
calibration
selection_rule
oos_definition
folds
seed
environment
artifact_hash
created_at
status
result
decision
failure_reason

を保存する。

⸻

17. EXPERIMENT FINGERPRINT

重複Researchを防ぐため、

data fingerprint
feature fingerprint
model fingerprint
target fingerprint
OOS fingerprint
configuration fingerprint

を作り、

NOVEL
RELATED
DUPLICATE
SUPERSEDED

を判定する。

⸻

18. RESEARCH SEARCH ROUTER

検索目的を区別:

DIRECT SEARCH
METHOD SEARCH
FAILURE SEARCH
COUNTEREXAMPLE SEARCH
IMPLEMENTATION SEARCH
BENCHMARK SEARCH
NEGATIVE-EVIDENCE SEARCH
FRONTIER SEARCH
CROSS-DOMAIN SEARCH
UNKNOWN-UNKNOWN SEARCH

肯定証拠だけでなく反証・失敗例も探索する。

⸻

19. RESEARCH INGESTION

External research:

DISCOVERED
→ SOURCE_VERIFIED
→ METHOD_ABSTRACTED
→ RELEVANCE_CHECKED
→ COST_CHECKED
→ PIT_CHECKED
→ LOCAL_IMPLEMENTATION
→ LOCAL_REPRODUCTION
→ OOS
→ ROBUSTNESS
→ HOLDOUT
→ ACCEPT / HOLD / REJECT

⸻

20. KNOWLEDGE EVIDENCE LEVEL

知識を以下に分類:

OBSERVED
SOURCE_VERIFIED
LOCALLY_REPRODUCED
OOS_CONFIRMED
ROBUST
INDEPENDENTLY_CONFIRMED
PRODUCTION_CONFIRMED
CONTRADICTED
DEPRECATED

読んだだけのmethodをproduction evidenceとみなさない。

⸻

21. NEGATIVE KNOWLEDGE

失敗したresearchを削除しない。

保存:

hypothesis
scope
data
failure
failed_conditions
reason
confidence
reopen_trigger

⸻

22. OOS STANDARD

Production-grade evaluation:

Chronological
Walk-forward
PIT-safe
Candidate selection separate
Final evaluation separate
Frozen holdout separate

必要に応じ:

purge
embargo
nested chronological evaluation
event/block bootstrap
dependence-aware statistics

⸻

23. CORE METRICS

Primary:

LogLoss
Brier
Accuracy
ECE

Additional:

Calibration Slope
Calibration Intercept
Sharpness
Resolution

Distribution:

Mean
Median
Std
Min
Max
Worst Fold
Newest Fold
Recent Window

Segments:

Volatility
Regime
Confidence
Time of Day
Market Condition
OOD
Prediction Age
Source State

必ずsample sizeとeffective sample sizeを併記する。

⸻

24. STATISTICAL INTEGRITY

candidate comparisonでは、

effect size
absolute delta
relative delta
confidence interval
variance
sample size
effective sample size
multiple testing
dependence
power
practical significance

を評価。

p-value単独で採用しない。

⸻

25. RESEARCH DEGREES OF FREEDOM

保存:

candidate count
feature trials
source trials
hyperparameter trials
calibration trials
router trials
timing trials
target trials
selection iterations

探索量を無視してwinnerを評価しない。

⸻

26. ADOPTION GATES

既存RESEARCH_STANDARDをcanonical policyとする。

参考sample thresholds:

~2k OOS preliminary
~5k robustness
~10k final adoption

ただしsample size aloneでは採用しない。

最低条件:

PIT PASS
Leakage PASS
Reproducibility
Robustness
Incumbent same-observation comparison
Calibration safety
Newest holdout safety
Independent confirmation

⸻

27. MODEL ECOLOGY

Model roles:

GENERALIST
RECENCY EXPERT
VOLATILITY EXPERT
MICROSTRUCTURE EXPERT
REGIME EXPERT
LONG-MEMORY EXPERT
CALIBRATION EXPERT
RISK EXPERT
SELECTIVE EXPERT
FALLBACK EXPERT

各Model:

skill
diversity
failure_profile
calibration
data_dependency
compute
latency
stability
age
lifetime

⸻

28. ROUTING

Routerは、

market regime
volatility
liquidity
microstructure
OOD
data quality
source reliability
model disagreement
recent performance
forecast lifetime

等からmodel selectionを行う候補。

Router自身にもPIT/OOS/robustness/holdoutを適用する。

⸻

29. UNCERTAINTY

以下を分離:

Prediction Confidence
Data Confidence
Source Confidence
PIT Confidence
Model Confidence
Regime Confidence
System Confidence

総合判断で相互混同しない。

⸻

30. PREDICTABILITY

Predictabilityはaccuracyではない。

inputs:

model disagreement
OOD
regime stability
recent error
calibration stability
data quality
source reliability
forecast lifetime

を利用する。

⸻

31. FAILURE RISK

予測そのものとは別に、

「このpredictionが壊れる確率」

を研究可能にする。

Failure predictorも通常modelと同じPIT/OOS規則を受ける。

⸻

32. INFORMATION ACQUISITION

可能なaction:

PREDICT_NOW
ACQUIRE_MORE
WAIT
RECOMPUTE
ROUTE
FALLBACK
ABSTAIN

選択時:

VOI
information quality
PIT
latency
compute
availability
risk

を評価する。

⸻

33. FORECAST LIFETIME

Predictionに:

valid_from
valid_until
prediction_age
invalidation_reason

を持たせる。

states:

FRESH
AGING
STALE
INVALIDATED

新情報・regime shock・critical source degradationでinvalidate可能にする。

⸻

34. PREDICTION EVENT LOG

Prediction更新を上書きしない。

各revision:

prediction_before
prediction_after
delta
reason
changed_sources
changed_features
changed_model
changed_regime
changed_calibration

を保存する。

⸻

35. RESULT PRESENTATION CONTRACT

CURRENT

Target
Horizon
UP
FLAT
DOWN
Top prediction
Prediction State
Decision Reliability
Prediction Cutoff
Data As-of
Data Age

DETAIL

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

CHANGE

Previous
Current
Absolute Delta
Percentage/Probability-point Delta
Change Drivers

AUDIT

Prediction ID
Experiment ID
Model Version
Data Snapshot
Feature Schema
Git SHA
PIT Result
Leakage Result
Reproducibility

Historical / Research candidate / Current productionを別表示する。

⸻

36. CHANGE EXPLANATION

予測変化について、

new information
source update
feature change
model change
calibration change
regime change
routing change

を可能な限り原因別に記録する。

⸻

37. OUTCOME / RECONCILIATION

PredictionとOutcomeを分離。

Outcomeについて:

outcome_id
event/reference
outcome
source
published_at
available_at
retrieved_at
maturity
revision
source_agreement

を保持。

Outcome maturity前にExperienceへ登録しない。

⸻

38. EXPERIENCE MEMORY

canonical mature caseのみExperienceへ登録。

同一eventのprediction revisionsでExperience件数を水増ししない。

Experienceから歴史predictionを改変しない。

⸻

39. FAILURE ANALYSIS

Failure categories:

MODEL
DATA
SOURCE
TIME
TARGET
ROUTING
CALIBRATION
UNCERTAINTY
DECISION
AUTOMATION
OUTCOME
RESEARCH
PRIORITY

さらに:

PRIMARY
SECONDARY
CONTRIBUTING
REDUCIBLE
INFORMATION_LIMITED
UNAVOIDABLE

を保存。

⸻

40. COUNTERFACTUAL FAILURE ANALYSIS

可能な場合:

different model
different feature set
additional source
different timing
different target
different routing

をsimulate/replayし、
failure sourceを切り分ける。

⸻

41. DRIFT

Separate:

Covariate Shift
Label Shift
Prior Shift
Concept Drift
Source Drift
Schema Drift
Entity Drift
Coverage Drift
Timing Drift
Regime Transition

Drift detector自身のfalse positive / false negativeも評価する。

⸻

42. REGIME

regimeを単一labelに限定しない。

候補:

volatility
trend
liquidity
correlation
microstructure
calendar
information
macro
market structure

Transition state:
STABLE_A
TRANSITION
STABLE_B
UNKNOWN

⸻

43. SCOPE FRONTIER

BTC research scope:

L0:
primary market/data

L1:
independent spot/perpetual venues

L2:
derivatives/options

L3:
on-chain

L4:
macro / central-bank / rates

L5:
public news/social/event information

L6:
novel / unknown frontier

各scopeは別々にPIT/OOS/robustness検証。

⸻

44. SOURCE VALUE

Source追加の価値を、

coverage improvement
incremental information
OOS improvement
calibration improvement
failure reduction
latency
cost
maintenance

で評価する。

「source数増加」を成功としない。

⸻

45. FEATURE RETIREMENT

Featureを追加するだけでなく、

redundant
unstable
drifting
leakage-prone
high-maintenance
low-value

featureを削除候補にする。

⸻

46. COMPLEXITY BUDGET

監視:

feature count
source count
model count
router complexity
calibration layers
workflow count
dependencies
latency
maintenance burden
failure surface

performance gainに対するcomplexity増加を評価する。

⸻

47. FAILURE SURFACE

新機能追加時に、

new dependencies
new failure modes
new data assumptions
new fallback paths
new state transitions

を評価。

小さなgainのための過剰なfailure surface増加を警戒する。

⸻

48. ONLINE / OFFLINE PARITY

同一input snapshotについて、

Historical pipeline
Production-like pipeline

を比較。

チェック:

features
missingness
timestamps
model inputs
probabilities
calibration
routing
state

⸻

49. DETERMINISTIC REPLAY

可能な限り:

Git SHA
Data Snapshot
Feature Schema
Config
Environment
Seed

を固定し、
同じ入力から同等結果が得られることを確認する。

⸻

50. CHAOS TEST

定期的に想定:

source timeout
API outage
schema change
duplicate
corrupt timestamp
stale data
bad artifact
cancelled workflow
dependency failure
resource exhaustion

Safe DegradationとRecoveryを検証。

⸻

51. SAFE DEGRADATION

Full
→ Reduced
→ Fallback
→ Selective
→ Abstain
→ Recovery

Critical integrity violationはBLOCK。

単なるdata availability低下とPIT violationを同一扱いしない。

⸻

52. RECOVERY

Checkpoint:

checkpoint_id
stage
scope
input_snapshot
completed_outputs
pending_work
expected_next_state
artifact_hashes
recovery_safety

を保存。

必要:
retry
backoff
resume
idempotency
rollback
replay
watchdog

⸻

53. ARTIFACT INTEGRITY

重要artifact:

model
dataset
calibration
registry
report
checkpoint

にhash/manifest/versionを付与。

⸻

54. STATE MACHINE

許可状態遷移を明示する。

例:

RESEARCH
→ CANDIDATE
→ OOS
→ ROBUST
→ HOLDOUT
→ SHADOW
→ PROMOTION
→ PRODUCTION

異常:

PRODUCTION
→ DEGRADING
→ INVESTIGATING
→ RECALIBRATION / RETRAIN
→ SHADOW
→ REPLACE / ROLLBACK

⸻

55. STATE CONSISTENCY

以下の矛盾をBLOCK:

PIT FAIL + PRODUCTION ACTIVE

REGISTRY PRODUCTION + MODEL MISSING

MODEL HASH MISMATCH

HOLDOUT INVALID + PROMOTION VALID

STALE SOURCE + TRUSTED PRODUCTION

Snapshot mismatch

⸻

56. HOLDOUT FIREWALL

Frozen Holdoutを、

model selection
feature selection
hyperparameter tuning
router tuning
calibration tuning
source selection
research prioritization

に使用禁止。

Holdout accessそのものをaudit eventにする。

⸻

57. RESEARCH PRIORITIZER

次の研究を、

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
risk

からprioritizeする。

⸻

58. RESEARCH STOPPING

以下で停止・延期:

no progress
duplicate
low information gain
poor reproducibility
high complexity
high failure surface
PIT uncertainty

状態:
STOP
DEFER
REDIRECT
SUPERSEDE

⸻

59. RESEARCH SATURATION

Research Yieldを監視。

new information rate
OOS improvement rate
failure reduction
coverage improvement

が長期停滞した場合、
adjacent/frontier/new-data/new-targetへ探索を移す。

⸻

60. UNKNOWN FRONTIER

探索対象:

unknown source
unknown feature
unknown regime
unknown target
unknown timing
unknown failure
unknown interaction
unknown market structure
unknown model family

未知を「存在しない」と仮定しない。

⸻

61. CROSS-PROJECT TRANSFER

他4 Projectのmechanismを利用する場合:

DISCOVER
→ ABSTRACT MECHANISM
→ COMPATIBILITY
→ LOCAL IMPLEMENTATION
→ LOCAL PIT
→ LOCAL OOS
→ LOCAL ROBUSTNESS
→ LOCAL HOLDOUT
→ SHADOW
→ PROMOTION

他Projectのperformanceをそのままtransfer evidenceにしない。

⸻

62. EXTERNAL INTELLIGENCE TRANSLATION

Web / Search / Paper / OSS / Plugin / external forecast等の情報は、

Discovery
→ Source verification
→ Reproducibility check
→ GitHub-compatible implementation
→ Local evaluation

へ変換する。

ChatGPT UI専用PluginをGitHub Actionsから直接使用できると仮定しない。

必要ならAPI/CLI/OSS/public endpoint/local implementation/cacheへ変換する。

⸻

63. COST FIREWALL

優先:

Verified Free
→ Free Quota
→ OSS/local
→ Cached
→ Lightweight compute

Paid-only
Billing-risk
Unknown-cost
Trial with billing risk

は自動導入しない。

UNKNOWN COST = HOLD / UNCONFIRMED

⸻

64. SECURITY

監視:

secret exposure
dependency vulnerability
workflow permission
action pinning
artifact tampering
untrusted code
supply-chain risk

Security uncertaintyがあるCandidateはproductionに入れない。

⸻

65. AUTOMATION HEALTH

Automation metrics:

false success
false recovery
retry rate
duplicate execution
checkpoint recovery
mean recovery time
stale artifacts
resource waste
workflow failure frequency

を保存する。

⸻

66. REPORT INTEGRITY

Raw result
→ Stored result
→ Computed metric
→ Report
→ Dashboard

をcross-checkする。

Report formattingによる数値誤りもFailureとして扱う。

⸻

67. PERFORMANCE CHANGE REPORT

性能変更時は必ず:

metric
previous
current
absolute delta
relative delta
sample size
effective sample size
confidence interval
worst fold
newest fold
calibration change

を記録。

ユーザーへの作業報告でもPerformance Deltaを明示する。

⸻

68. ABNORMAL IMPROVEMENT

急激な性能改善は自動的にaudit triggerとする。

疑う対象:

leakage
duplicate
label contamination
selection bias
data revision
target change
future information
benchmark contamination
evaluation bug

⸻

69. REPRODUCIBILITY

最低限保持:

Git SHA
environment
dependencies
config
seed
data snapshot
feature schema
model artifact
calibration
router
target definition

⸻

70. PRODUCTION BUNDLE

Production stateはmodel fileだけでは成立しない。

Bundle:

model
feature schema
source registry version
PIT policy
target definition
calibration
router
fallback
output schema
monitoring
rollback target
manifest

を一体としてversion管理する。

⸻

71. PRODUCTION SENTINEL

Production sentinelは、

Data Health
PIT Health
Model Health
Calibration
Drift
OOD
Failure Risk
Latency
Source Health
Artifact Integrity
State Consistency

を監視する。

⸻

72. ABSTENTION

Abstentionはfailureではない。

適切な条件下で、

ABSTAIN

を成功状態として記録できるようにする。

ただしabstention policy自体をOOSで評価する。

⸻

73. FORECAST QUALITY ≠ SYSTEM QUALITY

分離指標:

Forecast Quality
Data Quality
Decision Quality
Automation Quality
Research Quality
Recovery Quality
Security Quality

総合状態を一つの平均値だけで表現しない。

⸻

74. SELF-EVOLUTION

Source/Code/Workflow/Research Policyの改善は、

Gap
→ Hypothesis
→ Proposed Change
→ Impact Analysis
→ Test
→ PIT/OOS if relevant
→ Independent Validation
→ Promotion
→ Version Update

で行う。

自己生成変更が自己承認のみでProductionへ入ることを禁止する。

⸻

75. FINAL COMPLETION EVIDENCE

「Action Green」
「コードが存在」
「モデルが動く」
だけでは完成ではない。

Evidence Bundleとして、

SPEC
CODE
DATA
PIT
LEAKAGE
OOS
CALIBRATION
ROBUSTNESS
HOLDOUT
REPRODUCIBILITY
RECOVERY
MONITORING
ROLLBACK
STATE CONSISTENCY
RESULT PRESENTATION
KNOWLEDGE LINEAGE

を検証する。

⸻

76. ULTIMATE PRINCIPLE

このProjectは「最も複雑なBTC予測システム」を作ることを目的としない。

最終的に残すべきものは、

* 将来一般化に寄与する
* PITが証明できる
* 再現できる
* robustである
* maintenance可能
* failureを検出できる
* 必要な場合に予測を拒否できる
* 現在の状態を正しく説明できる
* 次の改善につながる

というEvidenceを持つ機構だけである。

追加より統合。
複雑化より情報効率。
平均性能より将来一般化。
予測数より正しい予測。
自信よりcalibration。
成功数よりfailure理解。
自動化率より安全な自律性。

常に、

MONITOR
→ DETECT
→ RESEARCH
→ IMPLEMENT
→ VERIFY
→ ADOPT / HOLD / REJECT
→ MONITOR

を繰り返す。

77. BINARY TARGET CANDIDATE — UP / DOWN ONLY

Binary target experiment version: binary_sign_v1.

Classes:
DOWN
UP

future_return > 0
→ UP

future_return <= 0
→ DOWN

FLATはbinary targetでは存在しない。

ただし既存Productionの3-class artifactは自動で置換しない。
Binary targetは、

LOCAL PIT
→ CHRONOLOGICAL OOS
→ ROBUSTNESS
→ CALIBRATION
→ FROZEN HOLDOUT
→ SHADOW
→ PROMOTION

を独立に通過するまでRESEARCH ONLYとする。

3-class Production evidenceをbinary targetのevidenceとして直接transferしない。

=== COPY END ===

⸻

78. CURRENT EVIDENCE SNAPSHOT

2026-10-04 current repository evidence:

strict PIT admitted primary scope:
5m strict primary settled = 397
10m strict primary settled = 396
per-horizon minimum = 300
active PIT violations = 0

Legacy evidence remains quarantined:
legacy_unverified = 169
legacy_violation = 41

Current Production models:
5m = bootstrap.soft_ensemble.v5.4
10m = bootstrap.bootstrap_rf

Current live Binance-primary robustness evidence is not yet promotion-ready:
5m current-production-model cohort = 254 / 1000 minimum
10m current-production-model cohort = 396 / 1000 minimum

Therefore Robustness remains RESEARCH_ONLY / insufficient_data.

Situation-metadata maturity still requires 2,607 additional qualifying 5m-cycle rows for the 5m primary cohort and 2,608 for the 10m primary cohort to reach the 3,000-row maturity target.

⸻

79. CURRENT PRODUCTION EVIDENCE ENVELOPE

The current-production prediction workflow must emit an auditable evidence envelope containing:

prediction probabilities
target timestamps
forecast lifetime
model versions
Git SHA
calibration state
situation state
data quality
source provenance
PIT temporal checks
reliability fields
research-only boundaries

Predictability, OOD, Failure Risk, Delta and Change Drivers must be explicitly marked
NOT_COMPUTED_AT_PRODUCTION_RUNTIME
or
NOT_AVAILABLE_IN_SINGLE_CURRENT_PREDICTION_RUN
when those quantities are not produced by the runtime.

Do not infer or fabricate these values.

For every source with usable status, PIT ordering must be auditable as:

available_at
→ retrieved_at
→ prediction_cutoff
→ decision time

and any contradictory valid-source ordering is a failure.

5m and 10m primary target_at timestamps must be recorded independently.
Extended horizons remain RESEARCH_ONLY.

⸻

80. BINARY TARGET CI INTEGRITY

Binary target experiment:

binary_sign_v1
classes = DOWN / UP
Production replacement = prohibited

The Binary Target Research workflow must restore the repository prediction database from
data/predictions.db.gz
before reading live-primary prediction evidence.

A missing or uninitialized predictions table is a CI infrastructure failure, not evidence
that live-primary binary OOS is empty.

The binary workflow must keep:

LOCAL PIT
→ CHRONOLOGICAL OOS
→ ROBUSTNESS
→ CALIBRATION
→ FROZEN HOLDOUT
→ SHADOW
→ PROMOTION

independent and research-only.

The current fixed workflow includes explicit repository-state restoration and a dedicated
contract test.

⸻

81. ROBUSTNESS EVIDENCE ELIGIBILITY

Robustness reports must fail closed on immature live evidence.

promotion_evidence_eligible = true
is allowed only when:

data_source = live_binance_primary
AND
status = ok
AND
n >= 1000
AND
final_holdout_protected = true

Live cohorts below the minimum remain insufficient_data and cannot be labeled
promotion evidence eligible.

Archive diagnostics are never promotion evidence and never change Production.

=== COPY END ===


⸻

82. BINARY TARGET RESEARCH SNAPSHOT

The latest fixed CI run for binary_sign_v1 completed successfully. It remains RESEARCH_ONLY and did not change Production.

* 5m chronological OOS: 18,500 evaluated rows; ExtraTrees best development candidate; OOS accuracy 0.51254, logloss 0.69323, Brier 0.25004, ECE 0.00097.
* 10m chronological OOS: 18,500 evaluated rows; ExtraTrees best development candidate; OOS accuracy 0.51557, logloss 0.69285, Brier 0.24985, ECE 0.00279.
* 5m live Binance-primary strict-PIT: n=395, accuracy 0.51392, logloss 0.69166, Brier 0.24926, ECE 0.00629.
* 10m live Binance-primary strict-PIT: n=395, accuracy 0.56962, logloss 0.68836, Brier 0.24761, ECE 0.05565.
* Frozen holdout is descriptive only and remains protected from candidate selection.
* Production replacement remains prohibited; independent longer robustness, calibration and shadow evidence are still required.

=== COPY END ===

⸻

83. BINARY LIVE KNOWLEDGE-TIME FIREWALL

For the binary_sign_v1 live-primary evaluation, archive training labels must satisfy:

label maturity / target timestamp
<
first live prediction cutoff

Filtering by prediction-row creation time alone is insufficient because a historical prediction can exist before the live window while its future outcome becomes known after the live window starts.

Rows with invalid or missing target timestamps are excluded from the training set.

This firewall is research-only and has no Production activation effect.

⸻

84. BINARY OOS STATISTICAL EVIDENCE CONTRACT

Binary Target Research must preserve both model and baseline evidence.

For each chronological WFO block and candidate:

train_end
test_start
test_end
test_n
same-block frequency baseline
candidate LogLoss/Brier/Accuracy

must be retained.

The aggregate evidence must additionally expose:

Accuracy 95% CI
approximate effective sample size
non-degraded fold fraction
worst LogLoss fold
newest fold
mean relative LogLoss improvement versus the same-block baseline

A baseline cannot be selected as a trainable challenger.

Missing diagnostic evidence is a verification failure, not a zero or PASS.



⸻

85. BINARY EVIDENCE CODE-LINEAGE BINDING

Binary Target Research evidence must record the exact analysis Git SHA used to generate the evidence.

The workflow must require:

analysis_git_sha = workflow GITHUB_SHA

A research-evidence commit created after the analysis is intentionally distinct from the analysis commit. The recorded analysis SHA identifies the code/data logic under evaluation and prevents stale or regenerated evidence from being mistaken for evidence produced by the current implementation.

LOCAL_UNPINNED runs are not Production evidence.

