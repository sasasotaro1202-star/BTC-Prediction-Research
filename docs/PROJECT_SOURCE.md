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
schedule recovery
workflow_run event wakeup
concurrency collapse

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

Bootstrap trainingはresearch-onlyとする。bootstrapで開発gateを通過しても、既存Productionのmodel artifactまたはmodel_registryを直接更新してはならない。Production replacementはLOCAL PIT → LOCAL OOS/WFO → CALIBRATION → ROBUSTNESS → FROZEN HOLDOUT → SHADOW → PROMOTIONの独立証拠を要求する。scheduled live workflowはmodel ageだけを理由にProduction generationをrefreshしてはならず、既存generationを安定保持する。
Promotion Gateはaggregate strict-PIT件数だけではPASSしてはならない。5mと10mそれぞれの`primary_horizon_gate`が存在し、`ready=true`かつstrict-primary件数が共通minimum以上であることを独立に検証する。片方でも欠落・不足・FAILならpromotion_allowed=falseとする。

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

GitHub-side deterministic routing may use durable drift and model-disagreement evidence as an uncertainty-research trigger. This is research prioritization only; it cannot alter Production and must remain behind PIT/health hard gates.

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

Evidence snapshot checked on 2026-10-06:

evidence_basis_head:
9f64763c0487af247945f03345b275873aa456e9

Subsequent main commits that only synchronize or repair documentation do not alter the evidence basis above. Runtime checks must always resolve the actual current main HEAD before execution.

Strict PIT admitted primary scope:
5m strict primary settled = 453
10m strict primary settled = 452
per-horizon minimum = 300
active current-scope PIT violations = 0

Checked prediction rows:
checked_predictions = 774
verified_primary_predictions = 453
verified_fallback_predictions = 152

Legacy evidence remains quarantined:
legacy_unverified = 169
legacy_violation = 41

Current Production models:
5m = bootstrap.soft_ensemble.v5.4
10m = bootstrap.bootstrap_rf

Current live Binance-primary robustness evidence is not yet promotion-ready:
5m current-production-model cohort = 310 / 1000 minimum
10m current-production-model cohort = 452 / 1000 minimum
promotion_evidence_eligible = false for both horizons

Situation-metadata maturity:
5m situation_meta_ready = 449; additional qualifying rows needed = 2,551
10m situation_meta_ready = 448; additional qualifying rows needed = 2,552
3,000-row maturity target remains unmet.

Online-expert readiness:
5m = 449
10m = 448
No additional online-expert row debt is reported under the current artifact contract.

Calibration evidence remains unverified:
5m n_settled = 310; fit_logloss = missing; holdout_logloss = missing
10m n_settled = 452; fit_logloss = 1.0505990758858512; holdout_logloss = 1.1082636171116929

Current promotion gate:
production_safety_gate = HOLD
promotion_allowed = false
promotion_status = HOLD

Reason:
robustness_evidence_invalid_or_incomplete;
candidate_or_frozen_holdout_non_regression_not_verified;
calibration_evidence_invalid_or_missing

These values are a synchronized evidence snapshot. The documentation commit itself does not change Production state or research evidence. Later runs must re-acquire current evidence rather than treating this snapshot as permanently current.

⸻

79. CURRENT PRODUCTION EVIDENCE ENVELOPE



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
* 5m live Binance-primary strict-PIT: n=397, accuracy 0.52141, logloss 0.69166, Brier 0.24926, ECE 0.01380.
* 10m live Binance-primary strict-PIT: n=396, accuracy 0.56818, logloss 0.68858, Brier 0.24772, ECE 0.05426.
* Validated CI run: #15; analysis Git SHA: 1724b871d100942619666cba117c4b33dfbcae35.
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



⸻

86. PREDICTION-CONFIDENCE RELIABILITY DIAGNOSTIC

The existing Experience Policy OOS surface now reports post-outcome prediction-confidence reliability without using realized outcomes as model inputs.

For each primary horizon it records:

confidence bucket sample count
average prediction confidence
observed accuracy
confidence gap
Accuracy 95% CI
confidence ECE
confidence Brier score
explicit 0.70+ high-confidence bucket status

This diagnostic is descriptive/research-only. It does not alter production confidence, routing, abstention, calibration, or model artifacts.

A material negative confidence gap in the 0.70+ bucket is an overconfidence signal requiring further calibration/selective-prediction research rather than automatic production action.

The same surface also reports reliability of the learned prediction-error probability.



87. FRONTIER HISTORICAL TIME-BOUND INTEGRITY

Autonomous historical backfill moves from newer to older event-time windows. Durable frontier state must therefore extend:
- earliest_event_time with MIN(existing, batch_first)
- latest_event_time with MAX(existing, batch_last)

Overwriting latest_event_time with the most recent backfill batch is invalid and can produce first_event_time > last_event_time.

The frontier selector now:
- detects reversed/malformed durable historical time bounds
- records an explicit integrity issue instead of silently normalizing it
- forces historical archive reacquisition when the issue exists, bypassing source cooldown
- preserves production_eligible=false for all frontier acquisitions
- validates that every acquisition source exposes a time_bounds_status

Repair is evidence-generating and research-only. Existing invalid state is not manually rewritten without source-derived evidence.

⸻

88. PER-HORIZON STRICT PIT READINESS GATE

Primary readiness must consume the PIT artifact's `primary_horizon_gate` independently for 5m and 10m.

For each primary horizon:

strict_primary_settled >= min_strict_pit_rows
minimum >= min_strict_pit_rows
ready = true

are required before Readiness can advance beyond PIT_COLLECTION.

A combined `verified_primary_predictions` total is diagnostic only and cannot satisfy a missing or failed horizon-specific gate. Missing, malformed, or incomplete `primary_horizon_gate` evidence fails closed.

This guard prevents one horizon from borrowing evidence from another horizon and preserves the target/horizon-specific PIT contract. It is a readiness/read-only research control and does not activate or alter Production.
⸻

89. 24H MARATHON DEPENDENCY INTEGRITY

各24H stageが参照する `needs.<job>.result` は、必ずそのjobを同stageの直接dependencyとして `needs` に列挙する。

GitHub Actionsの `needs` contextは直接依存jobだけを保証するため、上流stageを間接依存のまま参照して空値を成功判定へ混入させてはならない。

24H Stage 4はStage 1 / 2 / 3を直接依存として保持し、terminal guardで全upstream resultを確認する。これは研究成果ではなくWorkflow Integrityの制御であり、修正時もProduction stateへ影響させない。


⸻

90. RESUMABLE FLOW COLLECTOR CONCURRENCY INTEGRITY

Binance Flow Researchのrolling collectorはcheckpointを専用branchへ保存するため、後続main commitで実行中captureをキャンセルしてはならない。

Pull Requestのvalidationはsuperseded runをcancelしてよいが、schedule / workflow_dispatchの長時間collectorはstableなcollector concurrency groupでserializeし、後続runをqueueする。main commitによるactive captureのcancelは行わない。

並行collector間のcheckpoint競合では、push retryごと、および最終publish時にlocal `end_time_ms` とremote `end_time_ms` を再比較する。localがremote以下なら古いsnapshotの上書きを行わず終了する。

この制御はData/PIT evidenceの欠損・巻き戻しを防ぐための運用整合性機構であり、Production model/stateを変更しない。


⸻

91. RESUMABLE FLOW COLLECTOR SERIALIZATION

Binance Flow Researchの長時間collectorは、schedule / workflow_dispatchだけで起動する。

長時間capture同士はstableなcollector concurrency groupでserializeし、後続runをqueueする。mainへの新しいcommitでactive captureをcancelしない。

Pull Requestのvalidationだけはhead branch単位でsuperseded runをcancelしてよい。

source / test変更はUnit TestsとPull Request validationで検証し、mainへのcommitごとに25分captureを重複起動しない。

これにより、checkpointの再開性を維持しながら、同一時間帯のcollector重複・Actions資源浪費・不要なcache競合を抑える。


⸻

92. CROSS-PROJECT MECHANISM GOVERNANCE — 2026-10-04

The five-project research set is a mechanism reference layer only:

Baseball-Prediction-System
BTC-Prediction-Research
7-Sport-Prediction-Research
Soccer-Prediction-Research
Stock-Daily-Prediction-3000

External project performance, OOS, holdout, production, prediction or data are never transferable evidence for BTC.

Reference mechanisms currently identified:

7-Sport:
* fail closed on selected-event enrichment errors
* success-only checkpoint reuse
* single-writer semantics for critical state
* event-cluster-aware evaluation

Soccer:
* explicit prediction-cutoff lineage
* optional feature-level PIT provenance
* prior-mature outcome filtering for temporal meta-learning
* fail-closed diagnostic state

Baseball:
* universal dataset/source contract
* explicit source/scope readiness state machine
* immutable production prediction / experience integrity audit
* outcome-maturity-aware temporal conformal research

Stock:
* genuinely prequential nested model/window/ranking selection
* contiguous prior-fold evidence
* dependence-aware moving-block bootstrap
* multiple-comparison-aware selection evidence
* run provenance manifest binding code/config/holdout policy

BTC transfer path:

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

⸻

93. FEATURE-LEVEL PIT FIREWALL

Prediction-level provenance is not sufficient when a feature snapshot can carry its own timing state.

When feature provenance fields are supplied, BTC strict PIT must additionally enforce:

feature_pit_status = PASS
feature_snapshot_cutoff <= prediction_cutoff
feature_max_available_at <= prediction_cutoff

Optional per-feature records are also checked for PIT status and available_at.

Missing optional feature lineage is not converted to PASS by inference. Invalid, unknown, contradictory, or post-cutoff feature lineage is fail-closed.

This control is backward-compatible with historical rows that predate feature-level provenance and does not rewrite those rows.

⸻

94. KNOWLEDGE-TIME / MATURE-OUTCOME FIREWALL

A row's creation time does not prove that its outcome was known.

For experience-derived or failure-risk research, labels used to train a prediction point must be mature before that point's prediction cutoff. Target/outcome timestamps and settlement/maturity state are preferred causal boundaries.

The current experience learner already separates prior settled batches; future hardening must preserve same-boundary batching and must not allow an outcome settled at the current test boundary to train that same boundary.

Rows with invalid maturity/target timestamps are UNKNOWN/DEFERRED, never silently usable.

⸻

95. DEPENDENCE-AWARE EVALUATION

When multiple prediction snapshots belong to the same underlying event/case, the statistical unit is the case/event cluster rather than the raw snapshot count wherever cluster identity is available.

Required safeguards include:

* event/case cluster identifiers where available
* no duplicate-cluster inflation of uncertainty estimates
* chronological block separation
* HAC, moving-block bootstrap, or cluster bootstrap where appropriate
* effective sample size reporting

The existing experience OOS batches same-settlement timestamps together. This is a partial safeguard; event-level clustering remains a frontier when snapshot revisions can share one market event without identical settlement times.

⸻

96. NESTED / PREQUENTIAL SELECTION FIREWALL

Model, feature, training-window, router, calibration, return-estimator and weighting selection must never use the outcomes of the fold being scored.

For each outer fold:

1. select from strictly earlier folds;
2. freeze the selected configuration;
3. score the untouched current fold;
4. only after scoring, expose its outcomes to later folds.

Where candidate windows exist, selection should require contiguous prior-fold support. Global same-OOS selections are not valid substitutes for nested evidence.

BTC already uses nested chronological calibration in the production challenger path; new routing/window research must follow the same contract.

⸻

97. PRODUCTION EVIDENCE PROVENANCE MANIFEST

The Production Artifact Audit must bind the audited artifact set to:

GITHUB_SHA
GITHUB_REF_NAME
PROJECT_INSTRUCTIONS.md SHA256
docs/PROJECT_SOURCE.md SHA256
requirements.txt SHA256
model artifact SHA256
metadata SHA256
runtime reload result
feature schema

Local runs without a pinned repository SHA are labeled LOCAL_UNPINNED and are not Promotion evidence.

A policy/config-only change is still an auditable state change even when the model artifacts are unchanged.

⸻

98. READINESS / SOURCE / SCOPE STATE MACHINE

Registration is not readiness.

For research sources and scopes, use explicit states such as:

REGISTERED
→ ADAPTER
→ PIT
→ OOS
→ ROBUST
→ FROZEN_HOLDOUT
→ SHADOW
→ PRODUCTION

Failure states remain visible:

HOLD
REJECTED
BLOCKED
DEFERRED
FAILED
UNKNOWN

A missing gate is not zero and never implies PASS.

⸻

99. AUTOMATION / WORKFLOW STATE INTEGRITY

The workflow state machine distinguishes:

REQUESTED
QUEUED
PENDING
WAITING
IN_PROGRESS
SUCCESS
FAILURE
CANCELLED
SKIPPED

Only explicit terminal success can produce execution evidence. A cancelled run, retry, partial stage, missing job, or stale lookup is not silently normalized to success.

Long-running research must use:

checkpoint
resume
idempotency
bounded retry
stable concurrency
immutable snapshot
single-writer critical-state handling
watchdog
heartbeat
stale-run detection

Main-branch drift must not silently change the code/data snapshot being evaluated by later immutable stages.

⸻

100. EVIDENCE COMPLETENESS / KNOWLEDGE LINEAGE

The durable evidence chain is:

RAW
→ SNAPSHOT
→ PIT
→ EXPERIMENT
→ OOS/WFO
→ CALIBRATION
→ ROBUSTNESS
→ FROZEN_HOLDOUT
→ SHADOW
→ PROMOTION
→ PRODUCTION
→ OUTCOME
→ EXPERIENCE
→ FAILURE
→ NEXT_RESEARCH

Each material result must retain enough lineage to answer:

what code ran
what data ran
what policy ran
what source ran
what target/horizon ran
what selection occurred
what holdout boundary was protected
what artifact was produced
what decision was made
why it was made

Knowledge copied from another project must be labeled mechanism-derived, not locally validated performance evidence.


⸻

101. BROAD PATTERN MATRIX RESEARCH

Broad experimentation is a first-class research capability, but complexity is budgeted.

Axes:
TARGET/HORIZON
FEATURE SET
MODEL/PARAMETERS
TRAINING WINDOW
CALIBRATION
ENSEMBLE
ROUTING
UNCERTAINTY
SELECTIVE ACTION
TIMING
INFORMATION ACQUISITION
SOURCE SCOPE

Canonical matrix:
10 feature sets × 8 deterministic model variants × 3 training-window policies = 240 configurations per horizon.

Feature sets:
all_15
returns_momentum
volatility_regime
candle_shape
volume_flow
trend
compact_cross
mean_reversion
price_structure
flow_trend

Models:
logreg_c0.03
logreg_c0.1
logreg_c1.0
logreg_c3.0
extra_trees
rf
hgb
soft_ensemble

Windows:
expanding
recent_1500
recent_3000

Finalist selection uses a deterministic multi-objective rank ensemble over development relative LogLoss/Brier improvement, accuracy stability, improved-fold ratios and worst-case block behavior, with diversity bonuses across feature pattern, model family and training window. Frozen Holdout remains excluded from selection and gate.

The matrix is research-only. It does not mutate Production, model registry, live prediction state or frozen holdout. Candidate execution errors and insufficient-fold candidates are retained as explicit research failures rather than silently discarded. Long-running execution also writes atomic per-horizon checkpoints and may restore a prior-run checkpoint only when analysis SHA, data fingerprint and candidate manifest all match; mismatched checkpoint state is ignored rather than reused. Checkpoint reuse is success-only: a failed screen/finalist is never marked complete for resume purposes, while its failure evidence is retained.

Evidence-affecting pushes may supersede older in-flight matrix runs. Scheduled/manual matrix runs are not cancelled merely because another scheduled/manual run exists; the final persistence step still rejects stale main lineage.

⸻

102. TWO-STAGE PREQUENTIAL SEARCH

Stage A:
Development chronological WFO
→ purge
→ embargo
→ train-window policy
→ fit candidate
→ score untouched block
→ same-block frequency baseline
→ block stability

Stage B:
Finalists only
→ chronological temperature calibration from earlier training rows
→ untouched later block
→ incumbent same-observation comparison
→ moving-block bootstrap
→ effective sample size
→ newest/worst block inspection

Frozen Holdout:
descriptive only, one final score surface, never used for selection or gate.

No candidate is promotion-ready from this matrix alone.

⸻

103. FEATURE ECOLOGY

研究はfeature countではなくincremental informationを目的とする。

Group ablation:
returns/momentum
volatility/regime
candle shape
volume/flow
trend
compact cross-family

Future candidates may add interactions, transformations, conditional features and redundancy deletion. Each feature set is an ordered identity linked to the canonical feature schema. Same-upstream features do not increase source independence.

Feature-level PIT provenance, when present, requires:
feature_pit_status = PASS
feature_snapshot_cutoff <= prediction_cutoff
feature_max_available_at <= prediction_cutoff

Missing optional lineage is not inferred as PASS.

⸻

104. MODEL / HYPERPARAMETER ECOLOGY

Model family and parameter regime are separate experiment axes.

Candidate dimensions include:
regularization
tree depth
leaf size
number of estimators
learning rate
class weighting
feature subsampling

Equivalent candidates must be fingerprinted and deduplicated. High-complexity candidates require incremental OOS and robustness value before adoption.

⸻

105. WINDOW / RECENCY ECOLOGY

Compare:
expanding
recent fixed windows
future rolling windows
regime-conditioned windows

Window selection must be prequential. The scored outer block cannot influence the selected window. A recent-window gain is not sufficient when newest block, worst block, calibration, robustness or failure concentration degrades.
For the research-only recency challenger, any class-frequency baseline used in eligibility must be estimated strictly from labels available before the frozen holdout. Frozen-holdout class counts are descriptive evidence only and must not define the comparator or eligibility threshold.

⸻

106. CALIBRATION / PROBABILITY ECOLOGY

Separate raw prediction skill from probability reliability.

Research:
raw
temperature scaling
future validated vector/scalar calibration
probability averaging
confidence shrinkage
selective confidence threshold

Calibration parameters are chosen from data available before the scored block. Accuracy-only gains are insufficient for adoption.

⸻

107. ROUTING / UNCERTAINTY / SELECTIVE ECOLOGY

Routing candidates:
regime
volatility
liquidity
model disagreement
OOD
data quality
source reliability
recent error
forecast age

Decision candidates:
PREDICT_NOW
ACQUIRE_MORE
WAIT
RECOMPUTE
ROUTE
FALLBACK
ABSTAIN

Separate:
Prediction Confidence
Data Confidence
Source Confidence
PIT Confidence
Model Confidence
Regime Confidence
System Confidence
Predictability

Selective policies must report coverage, retained accuracy, LogLoss, Brier, ECE and failure concentration.

⸻

108. TIMING / INFORMATION-VALUE ECOLOGY

Test the information acquisition policy itself:

immediate
5m
10m
15m
30m
late refresh
wait-for-confirmation

and:

acquire_more_source
recompute_features
wait_for_new_candle
wait_for_liquidity_confirmation
fallback_to_independent_source
abstain

Compare PIT, latency, coverage, incremental information value, compute, failure risk and forecast lifetime.

⸻

109. TARGET / HORIZON FIREWALL

Primary target remains:
UP / FLAT / DOWN
5m / 10m

Adjacent research targets include:
binary direction
return
volatility
tail event
regime transition
predictability
failure probability
forecast lifetime

Every adjacent target has an independent target_definition_version and evidence chain. No 3-class Production evidence transfers automatically.

⸻

110. MULTIPLE TESTING / DEPENDENCE

Every broad experiment records:
candidate_count
screened_count
finalist_count
experiment_fingerprint
analysis_git_sha
source/data snapshot
feature set
model
window
calibration
folds
n
effective sample size
worst block
newest block

Candidate-vs-baseline and candidate-vs-incumbent comparisons use identical observations.

Prefer HAC / moving-block bootstrap / cluster bootstrap to IID inference for time-dependent evidence. Exploratory winner ≠ performance verification ≠ promotion approval.

⸻

111. EVIDENCE AND MAIN-LINEAGE FIREWALL

Pattern output must always declare:
research_only=true
production_changed=false
promotion_allowed=false

Analysis Git SHA is the SHA that actually executed the experiment. A later evidence-persistence commit never replaces that analysis identity.

If main advances while a long experiment is running, the stale run may preserve an artifact but must refuse to push its evidence into main. This prevents stale code/data snapshots from becoming current evidence.

⸻

112. CROSS-PROJECT MECHANISM TRANSFER — 2026-10-04

Reference repositories:
Baseball-Prediction-System
BTC-Prediction-Research
7-Sport-Prediction-Research
Soccer-Prediction-Research
Stock-Daily-Prediction-3000

Transferred mechanisms only:

7-Sport:
event/cluster-aware evaluation
enrichment failure fail-closed
success-only checkpoint
critical-state single writer

Soccer:
prediction-cutoff lineage
feature-level PIT provenance
mature-prior temporal learning
fail-closed diagnostic states

Baseball:
dataset/source readiness
immutable prediction/experience audit
outcome maturity firewall
temporal conformal research

Stock:
nested/prequential selection
contiguous prior-fold evidence
dependence-aware block bootstrap
explicit TESTS_PASSED / AUDIT_PASSED handoff
evidence freshness

Transfer path:
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

Cross-project performance, OOS, holdout, Production, prediction and data snapshots never become BTC evidence.

⸻

113. PATTERN MATRIX DEFINITION OF DONE

The broad search is considered execution-complete only when:
SPEC
CODE
DATA
PIT
LEAKAGE
OOS/WFO
CALIBRATION
INCUMBENT COMPARISON
DEPENDENCE-AWARE STATISTICS
FROZEN HOLDOUT FIREWALL
REPRODUCIBILITY
RECOVERY
EVIDENCE PERSISTENCE
STATE CONSISTENCY

are represented in the artifact or explicitly marked unavailable/deferred.


⸻

114. GITHUB-SIDE RESEARCH PR MERGE FIREWALL

Repository-side engineering automation may merge a Pull Request without enabling GitHub's repository Auto-Merge feature, but only through an explicit fail-closed research gate.

Eligibility requires all of the following:

* PR targets main
* PR head repository is the same repository
* PR is not draft
* branch uses an allowed engineering prefix
* PR body contains the exact marker `<!-- btc-automerge:research-only -->`
* changed files contain no data / model / registry / holdout / Production-state path
* every current-head check run is terminal and successful/neutral/skipped
* every current commit status is success
* the merge request is pinned to the exact observed HEAD SHA

The automation must:

* use pull_request_target or another trusted workflow definition without checking out untrusted PR code
* keep only the minimum write permissions required for the merge operation
* use squash merge with the expected HEAD SHA
* record an auditable Step Summary and workflow artifact
* treat merge rejection, stale HEAD, pending checks, and safety-gate failure as HOLD/NO_ACTION
* never relax PIT, leakage, OOS/WFO, calibration, robustness, frozen-holdout, shadow, promotion, rollback, or Production policies

Sensitive-path blocking is fail-closed. A research PR that changes Production artifacts, registries, live prediction state, protected holdout state, or other protected data cannot be auto-merged.

Manual GitHub repository Auto-Merge configuration is therefore not a prerequisite for safe research-PR automation. This automation is an engineering convenience only and is never evidence of research success or Production readiness.



⸻

115. AUTO-MERGE GATE SELF-PROTECTION

The research PR auto-merge gate is protected from self-modification.

Any Pull Request changing `.github/workflows/*` is treated as sensitive and cannot be auto-merged. Changes to the auto-merge workflow, Ops Preflight, Watchdog, Supervisor, or any other GitHub Action therefore require an independently reviewed/manual merge path.


78. CONDITIONAL RETURN DISTRIBUTION / TAIL RESEARCH

A research-only lane may estimate conditional endpoint-return quantiles q10/q50/q90 for the canonical 5m and 10m horizons using only matured strict-PIT Binance-primary prediction observations from the canonical prediction ledger.

Evaluation:
- chronological walk-forward
- purge + embargo
- training-window unconditional quantile baseline
- pinball loss
- 80% interval coverage
- lower/upper tail-breach rate
- newest-block performance
- interval width

The lane is explicitly research-only. It cannot become promotion evidence and cannot change Production artifacts or model_registry. It must not claim intrahorizon maximum drawdown unless path-level future observations are added under a separately validated target/data contract.

Evidence must retain analysis Git SHA and the prediction-database snapshot hash. The autonomous supervisor may dispatch the lane when evidence is missing or stale and may be woken by its completed workflow. Any eventual promotion remains subject to the independent PIT → OOS/WFO → calibration → robustness → frozen holdout → shadow → promotion gates.

79. AUTONOMOUS RESEARCH CONTINUITY EXTENSION

The Continuous Supervisor allowlist includes the return-distribution/tail lane. Its six-hour research freshness threshold is independent of the daily and multi-hour research schedules, and completed return-tail runs are included in the Supervisor workflow_run wakeup set. This is a research continuity mechanism only; failure or staleness never authorizes Production mutation.

80. IMMUTABLE RESEARCH SNAPSHOT RECOVERY

Long-running research lanes are immutable-analysis jobs, not moving-main jobs. A queued or running job may legitimately start from a prior main commit while Live/settlement state continues advancing main. The job must preserve its `GITHUB_SHA` as the analysis provenance and its exact input/data snapshot; concurrent main movement must not force a false failure. Before publishing evidence, the workflow reconciles with the latest main and commits only the research artifact. This preserves reproducibility without allowing stale code or stale evidence to masquerade as current Production state. A failed analysis is determined by the actual research/validation result, not merely by main advancing.


81. AUTONOMOUS RECOVERY BACKOFF

Repeated failure/cancellation/timeout must not cause an unbounded 5-minute dispatch loop. The Continuous Supervisor and Workflow Watchdog maintain a failure streak and use bounded exponential recovery cooldown, starting at 5 minutes and capped at 80 minutes. The stale-run janitor remains cancellation-only so recovery dispatch has a single coordinated policy across the Supervisor, Watchdog, and Production Sentinel. This changes recovery timing only; it never converts failure into success and never changes Production safety gates.



82. WORKFLOW CONTRACT VALIDATION LANE

The automation control plane uses a lightweight dedicated workflow-contract lane for `.github/workflows/*` changes. The full unit suite remains focused on source/tests/scripts/dependency changes, while workflow edits receive compile and recovery-contract validation. This reduces avoidable queue churn without weakening validation of Supervisor, Watchdog, immutable research recovery, or stale-run controls.


83. SHARED RECOVERY FAILURE STREAK

The Supervisor and Watchdog use the shared `scripts/ci_failure_streak.sh` helper for consecutive failure/cancellation counting. This removes duplicated inline recovery logic and makes the recovery state calculation directly testable. The helper counts only the terminal non-success streak since the latest explicit success; neutral/unknown terminal conclusions stop the streak. Ops Preflight syntax-checks the helper.
