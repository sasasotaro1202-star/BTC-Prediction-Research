
# BTC ULTIMATE PREDICTION SYSTEM — MASTER DESIGN REFERENCE

Status: RESEARCH / DESIGN REFERENCE

This document consolidates the project's canonical policy, current repository architecture, and the complete probabilistic BTC prediction design. It is intentionally broader than the currently Production-active implementation.

Operational authority remains PROJECT_INSTRUCTIONS.md + current GitHub HEAD + validated immutable evidence. This document does not itself promote a research candidate or authorize a Production change.

---

# 1. MISSION

Goal: maximize future generalization to unknown future BTC market states.

The system optimizes:

- predictive quality;
- calibration;
- PIT integrity;
- leakage resistance;
- robustness;
- uncertainty quality;
- predictability estimation;
- model disagreement;
- failure-risk estimation;
- forecast lifetime;
- information acquisition value;
- timing;
- routing;
- fallback;
- abstention;
- compute efficiency;
- operational reliability;
- reproducibility;
- safe autonomous research.

System loop:

DATA
→ TIME/PIT
→ TARGET
→ FEATURES
→ STATE
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
→ HOLDOUT
→ SHADOW
→ PROMOTION
→ PRODUCTION
→ MONITORING
→ RECOVERY
→ REDISCOVERY

---

# 2. CENTRAL IDEA

The wrong abstraction:

features → predicted price

The preferred abstraction:

observations
→ latent market state
→ conditional future distribution
→ uncertainty / failure risk
→ decision under cost and risk

Define:

X_t = admissible information available by cutoff t
S_t = latent market state
Y_(t+h) = future classification target
R_(t+h) = future return
V_(t+h) = future volatility
D_(t:t+h) = path drawdown risk
F_(t+h) = prediction failure event

Core objects:

P(S_t | X_≤t)

P(Y_(t+h) | S_t, X_≤t)

P(R, V, D, F | S_t, X_≤t)

Final action:

A_t = f(forecast, uncertainty, cost, liquidity, portfolio, constraints)

---

# 3. DISTRIBUTIONAL FORECASTING

A point estimate is incomplete.

Classification:

P(UP) + P(FLAT) + P(DOWN) = 1

Return forecasting should estimate a distribution F(r), not only a mean.

Useful quantiles:

q05
q10
q50
q90
q95

Distribution quality must be evaluated separately from direction accuracy.

A model can get the class right while having badly calibrated probabilities or unrealistically narrow intervals.

---

# 4. PRIMARY TARGET CONTRACT

Primary target classes:

UP
FLAT
DOWN

Primary production horizons:

5m
10m

Mandatory target metadata:

target_definition_version
observation_time
prediction_cutoff
horizon
label_rule
settlement_rule
revision_rule
missing_outcome_rule

Candidate research targets:

RETURN
VOLATILITY
TAIL EVENT
REGIME TRANSITION
PREDICTABILITY
FAILURE PROBABILITY
FORECAST LIFETIME

Candidate targets remain research-only until independently validated.

---

# 5. TIME MODEL

Time fields are distinct:

event_time
observation_time
source_publish_time
source_available_time
retrieved_time
processing_time
prediction_time
outcome_time
revision_time

Canonical timezone: UTC.

Exchange/local time may be stored as secondary metadata.

---

# 6. PIT CONTRACT

Primary historical rule:

source_available_time <= prediction_cutoff

retrieved_time <= prediction_cutoff is not sufficient.

Retrieved time only proves when the system fetched the data. It does not prove when the information became knowable.

Unknown or unverifiable availability becomes:

UNKNOWN
DEFERRED
REJECTED

Never infer PASS from missing provenance.

---

# 7. LEAKAGE TAXONOMY

Audit:

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

Zero means the scope was actually audited.

---

# 8. DATA QUALITY CONTRACT

Completeness is not row count.

Assess:

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

Critical blockers:

UNKNOWN AVAILABILITY
INVALID TIMESTAMP
UNRESOLVED IDENTITY
DUPLICATE
IMMATURE OUTCOME
PIT FAILURE

---

# 9. MISSING DATA

Missing != zero.

States:

AVAILABLE
MISSING
STALE
UNKNOWN
UNVERIFIABLE
INVALID
DEGRADED

A missing observation may be handled by a model, but missingness itself must remain represented.

Critical missing information may cause:

RECOMPUTE
FALLBACK
ABSTAIN
BLOCK

---

# 10. SOURCE REGISTRY

Each source should carry:

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

Mirrors/wrappers/republishers do not automatically count as independent evidence.

Source graph:

UPSTREAM
→ DATASET
→ MIRROR
→ WRAPPER
→ FEATURE
→ MODEL

---

# 11. FEATURE LINEAGE

Retain where possible:

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

Feature lineage must be sufficient to replay the feature construction.

---

# 12. IMMUTABLE SNAPSHOTS

Research snapshots are immutable.

Minimum:

snapshot_id
creation_time
data_range
source_versions
schema_hash
content_hash
manifest_hash
row_count
coverage_summary

Corrections create new snapshots.

Never overwrite historical evidence to make the current result look better.

---

# 13. MARKET AS A LATENT STATE

Observed price is not the market itself.

Represent latent state:

S_t = [
Trend,
Liquidity,
Leverage,
Flow,
Volatility,
Macro,
Valuation,
Sentiment,
Breadth,
Microstructure
]

Each component may be continuous, categorical, or probabilistic.

Do not force one regime label when uncertainty is meaningful.

Example:

P(Bull)=0.62
P(Range)=0.25
P(Bear)=0.13

---

# 14. STATE DIMENSIONS

Trend:

strong up / moderate up / range / moderate down / strong down

Liquidity:

abundant / normal / tight / stressed

Leverage:

low / normal / crowded long / crowded short

Flow:

accumulation / neutral / distribution

Volatility:

compressed / normal / high / extreme

Macro:

risk-on / neutral / risk-off

Regime transitions:

STABLE_A
TRANSITION
STABLE_B
UNKNOWN

---

# 15. STATE DURATION

Duration matters.

D_t = time spent in the current regime

A newly entered bull regime is not equivalent to a long-running bull regime.

Candidate tools:

HMM
Markov switching
Hidden Semi-Markov
state-space models
Kalman filtering
particle filtering

These remain candidate methods until locally evaluated.

---

# 16. PRICE FEATURES

Core candidates:

OHLC
returns
log returns
range
ATR
realized volatility
VWAP
VWAP distance
EMA gaps
high/low structure
support/resistance
breakout distance
range position
candle body
upper wick
lower wick
acceleration

Use multiple windows.

Examples:

r_1m
r_3m
r_5m
r_10m
r_30m
r_1h
r_4h

---

# 17. VOLUME

Candidates:

spot volume
futures volume
volume imbalance
CVD
volume profile
volume trend
VWAP interaction

Normalize using rolling statistics or percentile ranks where appropriate.

Example:

VolumeZ = (Volume_t - rolling_mean) / rolling_std

---

# 18. DERIVATIVES

Candidates:

open interest
funding
basis
futures premium
liquidations
implied volatility
options skew
put/call measures
term structure
dealer/gamma proxies where valid

Interpret signals conditionally:

P(Y | X, S)

not only:

P(Y | X)

---

# 19. PRICE × OI

Four basic patterns:

Price up + OI up:
new positioning candidate

Price up + OI down:
short closing / squeeze candidate

Price down + OI up:
new short / trapped long candidate

Price down + OI down:
deleveraging candidate

This matrix is descriptive, not a complete forecast.

---

# 20. FUNDING

Funding is a positioning/crowding variable.

Funding high does not imply price must fall.

It may indicate:

- crowding;
- strong directional trend;
- asymmetric liquidation risk.

Use funding mainly as a conditional variable.

---

# 21. OPEN INTEREST

Use:

OI level
OI percentile
ΔOI
ΔOI relative to price move
OI × Funding

The percentile is often more robust than an absolute level under non-stationarity.

---

# 22. SPOT VS LEVERAGE

Healthy candidate:

Price up
Spot demand up
Volume up
OI normal
Funding normal

Fragile candidate:

Price up
Spot demand flat
OI sharply up
Funding sharply up

The first suggests broader demand.

The second suggests increased leverage dependency.

---

# 23. FRAGILITY

Conceptual fragility:

Fragility =
w1 FundingExtreme
+ w2 OIExtreme
+ w3 LiquidationDensity
+ w4 LiquidityThinness
+ w5 PositioningSkew

Direction and fragility are separate.

Bullish does not necessarily mean safe.

---

# 24. DIRECTION × FRAGILITY

Low-fragility bull:
healthy trend candidate

High-fragility bull:
crowded/unstable trend candidate

Low-fragility bear:
stable downward regime

High-fragility bear:
crash/cascade candidate

---

# 25. ORDER FLOW

Candidates:

aggressive buyer volume
aggressive seller volume
bid/ask imbalance
CVD
order-book depth
liquidity gaps
slippage
large-order interaction

Executed behavior is stronger evidence than displayed orders.

---

# 26. SPOOFING-AWARE DESIGN

Displayed liquidity can disappear.

Therefore:

displayed liquidity != executed liquidity

Order-book signals should be validated against subsequent executions and persistence where feasible.

---

# 27. LIQUIDITY MAP

Construct Liquidity(P) by price.

Overlay:

liquidation clusters
stop clusters
volume profile
large resting liquidity
thin zones

Use for:

breakout probability
sweep probability
tail risk
execution cost

---

# 28. BREAKOUT ENGINE

Candidate confirmation:

close beyond level
volume confirmation
spot confirmation
CVD confirmation
healthy OI behavior
successful retest

Conceptual score:

B =
w1 Close
+ w2 Volume
+ w3 SpotFlow
+ w4 CVD
+ w5 Retest
- w6 Crowding

Weights are hypotheses until OOS-validated.

---

# 29. FAILED BREAKOUT

A failed breakout is a distinct event.

Price > resistance
then
close < resistance

Supporting warnings:

weak spot participation
negative CVD
high leverage
rejection wick

Interpretation:

P(trend continuation) decreases

not:

P(DOWN)=1

---

# 30. DIVERGENCE ENGINE

Candidates:

price vs momentum
price vs CVD
price vs flow
price vs breadth
price vs OI

Divergence should modify continuation probability rather than become an automatic opposite-direction trade.

---

# 31. MARKET BREADTH

Compare:

BTC
ETH
SOL
crypto index
alt breadth
stablecoin behavior
BTC dominance

Relative strength:

BTCAlpha = BTC return - beta × CryptoIndex return

This helps distinguish BTC-specific strength from broad crypto beta.

---

# 32. LEADERSHIP

Track possible leadership:

BTC leadership
ETH leadership
ALT speculative leadership
broad participation
defensive BTC dominance

Leadership is a regime/cycle input, not a direct timing rule.

---

# 33. MACRO

Candidates:

DXY
US yields
real yields
central-bank expectations
liquidity
VIX
Nasdaq
S&P 500
credit conditions

Horizon matters.

Slow macro features should not automatically dominate 5m/10m models.

---

# 34. CROSS-ASSET DECOMPOSITION

A possible factor model:

BTC return =
beta1 × Nasdaq return
+ beta2 × DXY change
+ beta3 × rate change
+ residual

Residual approximates BTC-specific movement after chosen factors.

It is not proof of causality.

---

# 35. ETF / INSTITUTIONAL FLOW

Candidates:

net flow
inflow
outflow
flow acceleration
volume
premium/discount

Do not assume flow alone predicts immediate price.

---

# 36. ON-CHAIN

Candidates:

exchange inflows/outflows
active addresses
realized cap
MVRV
SOPR
NUPL
LTH supply
STH supply
dormancy
miner behavior

Many on-chain variables are slow.

Do not force them into 5m/10m unless local evidence supports incremental value.

---

# 37. VALUATION

Valuation may inform:

medium-horizon prior
cycle state
tail asymmetry
long-run expected return

Overvaluation does not imply immediate downside.

---

# 38. EVENT / NEWS INTELLIGENCE

Event effect depends on:

surprise
novelty
expectedness
positioning
regime
liquidity
sensitivity

Generic surprise:

Surprise = Actual - Expected

Standardized surprise:

Z = (Actual - Expected) / historical_std

---

# 39. REACTION FUNCTION

Important principle:

headline content != complete information

More useful:

price reaction
volume reaction
flow reaction
positioning reaction

Conceptual model:

ΔP = f(Surprise, Positioning, Liquidity, Regime)

Good news + no rally can imply prior pricing, absorption or crowding.

Bad news + no decline can imply resilience or absorption.

---

# 40. EVENT STUDY

Set event time t=0.

Candidate windows:

-24h
-12h
-6h
0
+1h
+4h
+24h

Abnormal return:

AR = ActualReturn - ExpectedReturn

Control for broad market factors where appropriate.

---

# 41. COUNTERFACTUAL

Conceptual treatment effect:

Y(1) = outcome with event
Y(0) = outcome without event

TreatmentEffect = Y(1) - Y(0)

True counterfactual is not directly observed.

Use event-study/factor-control approximations with explicit uncertainty.

---

# 42. MODEL ECOLOGY

Roles:

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

Track each model's:

skill
diversity
failure profile
calibration
data dependency
compute
latency
stability
age
lifetime

---

# 43. SIMPLE BASELINES

Always retain simple baselines:

majority/frequency
logistic regression
compact tree ensemble

Complex models must show incremental value against incumbent and baseline.

---

# 44. TREE MODELS

Candidates:

Random Forest
ExtraTrees
Gradient Boosting
LightGBM where justified

Strengths:

nonlinear interaction
threshold effects
tabular robustness

Risks:

overfitting
probability miscalibration
feature-distribution dependence

---

# 45. LOGISTIC REGRESSION

Useful as:

stability anchor
interpretability reference
calibration reference
low-complexity baseline

A simpler model can be preferable when complexity adds no future-generalizing value.

---

# 46. NEURAL MODELS

Candidate families:

temporal CNN
RNN/LSTM/GRU
Transformer
temporal attention

Use only when local OOS evidence justifies complexity.

---

# 47. STATE-SPACE MODELS

General form:

x_t = A x_(t-1) + w_t
y_t = C x_t + v_t

x_t is latent state.
y_t is observed market data.

Kalman filtering gives online state estimation.

Particle filtering can represent multiple state hypotheses.

---

# 48. REGIME-CONDITIONAL MODELING

The same feature can have different meaning by regime.

Example:

P(Return | RSI, Bull)

can differ materially from:

P(Return | RSI, Range)

Therefore use conditional models or interaction features when validated.

---

# 49. INTERACTIONS

Important candidate interactions:

OI × Funding
Momentum × Funding
Return × ΔOI
Volume × Return
ETFflow × Momentum
Trend × Regime
Liquidity × Leverage

Avoid assuming additive effects only.

---

# 50. ENSEMBLE

Basic ensemble:

P = sum_k w_k P_k

Weights may depend on:

regime
volatility
data health
recent performance
model disagreement
forecast age

Dynamic weights must be independently validated.

---

# 51. MODEL DISAGREEMENT

If two models strongly disagree, disagreement is itself information.

Track:

pairwise distance
class disagreement
rank disagreement
probability range
majority margin
entropy
rolling disagreement

Use disagreement for:

confidence reduction
routing
selective prediction
information acquisition
failure prediction

Do not automatically convert disagreement into a direction.

---

# 52. ENTROPY

For class probabilities:

H(p) = -Σ p_i log(p_i)

High entropy = diffuse prediction.

Low entropy = concentrated prediction.

Low entropy is not automatically good calibration.

---

# 53. UNCERTAINTY DECOMPOSITION

Separate:

Prediction Confidence
Data Confidence
Source Confidence
PIT Confidence
Model Confidence
Regime Confidence
System Confidence

Never collapse all uncertainty sources into a single unexplained number.

---

# 54. PREDICTABILITY

Predictability is not accuracy.

Candidate inputs:

model disagreement
OOD
regime stability
recent error
calibration stability
data quality
source reliability
forecast lifetime

Interpretation:

“How predictable is the current environment for this system?”

---

# 55. OOD

Detect:

feature novelty
distribution shift
regime novelty
source novelty
model-disagreement spike

OOD should reduce trust or trigger additional information, not automatically decide direction.

---

# 56. FUTURE FAILURE PREDICTOR

Separate forecast failure from forecast direction.

P(Failure | Z_t)

Possible inputs:

disagreement
drift
confidence
source health
regime transition
prediction age
recent model error

Failure predictor must obey PIT/OOS rules.

---

# 57. INFORMATION ACQUISITION

Valid actions:

PREDICT_NOW
ACQUIRE_MORE
WAIT
RECOMPUTE
ROUTE
FALLBACK
ABSTAIN

Decision depends on:

VOI
information quality
PIT
latency
compute
availability
risk

---

# 58. VALUE OF INFORMATION

Concept:

VOI(source) =
expected decision utility with source
- expected decision utility without source
- source cost

A source with low marginal effect on decisions has low incremental value even if it looks impressive.

---

# 59. TIMING

Separate:

direction
entry timing
confirmation timing
exit timing
forecast expiration

A system may be:

long-term bullish
short-term uncertain
waiting for confirmation

That is internally consistent.

---

# 60. FORECAST LIFETIME

Each prediction carries:

valid_from
valid_until
prediction_age
invalidation_reason

States:

FRESH
AGING
STALE
INVALIDATED

Invalidate when:

major new information
regime shock
critical source degradation
thesis invalidation
model assumption break

Candidate decay:

Weight = exp(-lambda × age)

Must be OOS-tested before adoption.

---

# 61. CALIBRATION

Calibration means predicted probability agrees with long-run frequency.

If P(UP)=0.80 for a comparable bucket, realized UP should be near 0.80.

Evaluate separately from accuracy.

---

# 62. CORE METRICS

Primary:

LogLoss
Brier
Accuracy
ECE

Additional:

calibration slope
calibration intercept
sharpness
resolution
worst fold
newest fold
recent window

Segments:

volatility
regime
confidence
time of day
market condition
OOD
prediction age
source state

Always report sample size and effective sample size.

---

# 63. LOG LOSS

Per sample:

LL_i = -log(p_i,true_class)

Aggregate:

LogLoss = mean(LL_i)

This heavily penalizes unjustified certainty.

---

# 64. BRIER

Multiclass:

Brier = mean over samples of sum_c (p_ic - y_ic)^2

Useful for probability quality.

---

# 65. ECE

ECE =
Σ_k (n_k/N) × |accuracy_k - confidence_k|

Use with other diagnostics.

---

# 66. CALIBRATION METHODS

Candidate methods:

temperature scaling
Platt-style calibration
isotonic regression
prequential calibration
bounded adaptive calibration

Calibration tuning must respect OOS and holdout boundaries.

---

# 67. DECISION ≠ FORECAST

Forecast:

What is likely?

Decision:

What should we do given cost, risk and current position?

The same forecast can produce different actions under different risk constraints.

---

# 68. SCENARIO TREE

Example structure:

Current
├─ Bull continuation
│  ├─ breakout
│  └─ pullback then continuation
├─ Range
│  ├─ upper range
│  └─ lower range
└─ Bear transition
   ├─ breakdown
   └─ false breakdown

Each branch has:

probability
trigger
expected return
tail risk
invalidation

---

# 69. EXPECTED VALUE

For scenarios j:

EV = Σ P_j × R_j

Net:

EV_net = gross EV - fees - slippage - funding - market impact

---

# 70. RISK-ADJUSTED VALUE

Conceptually:

RAVEV = EV_net - lambda × TailRisk

Do not use a risk-adjusted score without defining the risk penalty.

---

# 71. TAIL RISK

Track:

lower quantiles
upper quantiles
interval breach rate
CVaR / expected shortfall where useful
jump probability
liquidation cascade risk

Heavy tails make a normal model insufficient by itself.

Candidate families:

Student-t
skew-t
mixture distributions
EVT
empirical distribution

---

# 72. JUMP MODEL

Stylized:

dP = mu dt + sigma dW + J dN

where:

mu = directional component
sigma = continuous volatility
J = jump size
N = jump occurrence process

This separates ordinary movement from shock movement.

---

# 73. LIQUIDATION CASCADE

Stylized feedback:

liquidation
→ price move
→ additional liquidation
→ larger price move

A Hawkes-style process can represent event clustering:

lambda_t =
mu + Σ alpha exp(-beta × (t - t_i))

Research-only until verified.

---

# 74. POSITION SIZING

Conceptually:

Size =
f(EV, Volatility, TailRisk, Confidence, Liquidity, Disagreement)

A strong forecast with poor certainty should be smaller than an equally strong forecast with robust calibration and agreement.

---

# 75. KELLY

Classical:

f* = (b p - q) / b

But model uncertainty is substantial.

Full Kelly can be unstable when probabilities are wrong.

Fractional Kelly is a possible risk-control approach.

---

# 76. NO-TRADE ZONE

If:

|EV_net| < threshold

possible action:

WAIT
NO TRADE
ABSTAIN

The system is not required to trade continuously.

---

# 77. HYPOTHESIS COMPETITION

Maintain multiple hypotheses simultaneously:

H1 Bull continuation
H2 Range
H3 Distribution → Bear

Each contains:

probability
evidence for
evidence against
trigger
invalidation
expected return
tail risk

---

# 78. BAYESIAN UPDATING

P(H|E) =
P(E|H)P(H) / P(E)

New information updates beliefs.

Do not anchor on yesterday's state.

---

# 79. EVIDENCE LEDGER

Support:

trend
spot demand
flow
macro
structure

Oppose:

crowding
divergence
liquidity stress
failed breakout
regime instability

Correlated evidence must not be counted multiple times.

---

# 80. FACTOR COMPRESSION

A factor representation can be:

X = B F + epsilon

Factors may include:

Trend
Flow
Liquidity
Leverage
Macro

Purpose:

- reduce duplicate evidence;
- improve stability;
- improve interpretability.

---

# 81. MULTI-HORIZON MODELING

5m:
microstructure + immediate flow + short momentum + liquidation

10m:
same inputs plus somewhat broader context

Longer research:
15m
30m
1h
3h
6h
12h
24h

Do not merge longer-horizon evidence into production until separately validated.

---

# 82. CROSS-HORIZON ALIGNMENT

Example:

P5m(UP)=0.53
P10m(UP)=0.64

Interpretation:

near-term uncertainty
with stronger short-term upside bias

Do not hide the disagreement in one scalar.

---

# 83. ROUTING

Router candidate inputs:

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

Outputs may be:

GENERALIST
RECENCY
VOLATILITY
MICROSTRUCTURE
REGIME
SCENARIO
DEEP_COMPUTE
FALLBACK
ABSTAIN

Router itself requires PIT/OOS/robustness/holdout.

---

# 84. SELECTIVE PREDICTION

Abstention is legitimate when:

PIT cannot be proven
critical data missing
OOD extreme
model disagreement extreme
calibration unstable
failure risk high

Evaluate:

coverage
selective accuracy
selective LogLoss
selective Brier
selective calibration
worst-case behavior

---

# 85. OOS / WFO STANDARD

Production-grade evaluation is:

CHRONOLOGICAL
WALK-FORWARD
PIT-SAFE
SELECTION-SEPARATED
FINAL-EVALUATION-SEPARATED
FROZEN-HOLDOUT-PROTECTED

Candidate tools:

purge
embargo
nested chronological evaluation
block bootstrap
cluster bootstrap
HAC-aware uncertainty

---

# 86. EFFECT SIZE

Candidate vs incumbent:

Delta = M_candidate - M_incumbent

Report:

absolute delta
relative delta
effect size
confidence interval
variance
sample size
effective sample size
worst fold
newest fold
multiple testing
dependence
practical significance

p-value alone is insufficient.

---

# 87. RESEARCH DEGREES OF FREEDOM

Track:

candidate count
feature trials
source trials
hyperparameter trials
calibration trials
router trials
timing trials
target trials
selection iterations

A winner from 5000 trials should not be treated like a pre-registered single hypothesis.

---

# 88. ROBUSTNESS

Test across:

time
regime
volatility
liquidity
source mix
confidence
OOD
recent windows
newest block
feature subsets

Inspect worst-case behavior.

---

# 89. ABNORMAL IMPROVEMENT

Sudden huge gains trigger an audit.

Check:

leakage
duplicate data
label contamination
selection bias
data revision
target change
future information
benchmark contamination
evaluation bug

An abnormal improvement is a debugging trigger before it is a success claim.

---

# 90. HOLDOUT FIREWALL

Frozen holdout cannot be used for:

model selection
feature selection
hyperparameter tuning
calibration tuning
router tuning
source selection
research prioritization
promotion tuning

Holdout access should be an auditable event.

---

# 91. DRIFT

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

The drift detector also has false positives/false negatives and must be evaluated.

---

# 92. ONLINE/OFFLINE PARITY

For identical snapshots compare:

features
missingness
timestamps
model inputs
probabilities
calibration
routing
state

Mismatch means historical evidence may not reproduce Production behavior.

---

# 93. PRODUCTION BUNDLE

Production is a bundle:

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

All components must be consistent.

---

# 94. RUNTIME SAFETY

Runtime should verify:

metadata horizon
artifact identity
candidate flag
classes
feature schema
registry version
artifact SHA
probability shape
finite probabilities
probability sum

Research candidates must never be selected at runtime.

---

# 95. CHAMPION / CHALLENGER

Champion:
current Production-approved bundle

Challenger:
research candidate tested on the same observable information

A challenger is not Production simply because:

- it exists;
- it runs;
- CI is green;
- one metric is better.

---

# 96. PROMOTION STATE MACHINE

RESEARCH
→ CANDIDATE
→ OOS
→ ROBUST
→ HOLDOUT
→ SHADOW
→ PROMOTION
→ PRODUCTION

Anomaly:

PRODUCTION
→ DEGRADING
→ INVESTIGATING
→ RECALIBRATION / RETRAIN
→ SHADOW
→ REPLACE / ROLLBACK

---

# 97. OUTCOME / EXPERIENCE

Prediction and outcome remain separate.

Outcome stores:

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

Only mature validated outcomes enter Experience.

Do not count repeated prediction revisions as new independent cases.

---

# 98. FAILURE ANALYSIS

Categories:

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

Classify:

PRIMARY
SECONDARY
CONTRIBUTING
REDUCIBLE
INFORMATION_LIMITED
UNAVOIDABLE

---

# 99. COUNTERFACTUAL FAILURE ANALYSIS

Replay, when feasible, under:

different model
different feature set
additional source
different timing
different target
different router

Question:

Could the failure realistically have been avoided?

---

# 100. NEGATIVE KNOWLEDGE

Retain failed experiments.

Store:

hypothesis
scope
data
failure
failed conditions
reason
confidence
reopen trigger

A rejected method may become useful after a regime or data change.

---

# 101. CHAOS TESTS

Inject:

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

Verify safe degradation and recovery.

---

# 102. SAFE DEGRADATION

FULL
→ REDUCED
→ FALLBACK
→ SELECTIVE
→ ABSTAIN
→ RECOVERY

PIT violations are stronger blockers than ordinary noncritical availability degradation.

---

# 103. CHECKPOINTS

Store:

checkpoint_id
stage
scope
input_snapshot
completed_outputs
pending_work
expected_next_state
artifact_hashes
recovery_safety

Need:

retry
backoff
resume
idempotency
rollback
replay
watchdog

---

# 104. ARTIFACT INTEGRITY

Important artifact families:

model
dataset
calibration
registry
report
checkpoint

Use:

hash
manifest
version

---

# 105. REPRODUCIBILITY

Freeze:

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

Same input should reproduce equivalent output within defined tolerance.

---

# 106. COMPLEXITY BUDGET

Track:

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

Optimize:

FutureGeneralizationGain / AddedComplexity

not raw complexity.

---

# 107. FEATURE RETIREMENT

Remove candidates that are:

redundant
unstable
drifting
leakage-prone
high-maintenance
low-value

A strong research system removes complexity as actively as it adds it.

---

# 108. RESEARCH ROUTING

Search modes:

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

Always search for disconfirming evidence.

---

# 109. EXTERNAL RESEARCH INGESTION

DISCOVERED
→ SOURCE VERIFIED
→ METHOD ABSTRACTED
→ RELEVANCE CHECKED
→ COST CHECKED
→ PIT CHECKED
→ LOCAL IMPLEMENTATION
→ LOCAL REPRODUCTION
→ OOS
→ ROBUSTNESS
→ HOLDOUT
→ ACCEPT / HOLD / REJECT

External performance claims do not transfer into BTC Production evidence.

---

# 110. KNOWLEDGE EVIDENCE LEVEL

OBSERVED
SOURCE_VERIFIED
LOCALLY_REPRODUCED
OOS_CONFIRMED
ROBUST
INDEPENDENTLY_CONFIRMED
PRODUCTION_CONFIRMED
CONTRADICTED
DEPRECATED

Read-only knowledge is not Production evidence.

---

# 111. CROSS-PROJECT TRANSFER

Transfer mechanisms and engineering controls.

Do not transfer:

performance
OOS
holdout
Production status
predictions

Required local validation:

DISCOVER
→ ABSTRACT
→ COMPATIBILITY
→ LOCAL IMPLEMENTATION
→ TEST
→ LOCAL PIT
→ LOCAL OOS
→ ROBUSTNESS
→ HOLDOUT
→ SHADOW
→ PROMOTION

---

# 112. COST FIREWALL

Preferred:

Verified Free
→ Free Quota
→ OSS / local
→ Cached
→ Lightweight compute

Do not automatically adopt:

Paid-only
Billing-risk
Unknown-cost
Trial with billing risk

Unknown cost = HOLD / UNCONFIRMED

---

# 113. SECURITY

Monitor:

secret exposure
dependency vulnerabilities
workflow permissions
action pinning
artifact tampering
untrusted code
supply-chain risk

Security uncertainty blocks Production.

---

# 114. AUTOMATION HEALTH

Monitor:

false success
false recovery
retry rate
duplicate execution
checkpoint recovery
mean recovery time
stale artifacts
resource waste
workflow failure frequency

---

# 115. RESULT INTEGRITY

Raw result
→ Stored result
→ Metric
→ Report
→ Dashboard

Cross-check all stages.

Formatting errors in reported numbers are also failures.

---

# 116. PERFORMANCE CHANGE REPORT

For each material change:

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

---

# 117. SELF-EVOLUTION

Improvement process:

Gap
→ Hypothesis
→ Proposed Change
→ Impact Analysis
→ Test
→ PIT/OOS when relevant
→ Independent Validation
→ Promotion
→ Version Update

A self-generated change cannot self-authorize Production solely from its own evaluation.

---

# 118. RESEARCH PRIORITIZATION

Candidate priority should consider:

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

---

# 119. RESEARCH STOPPING

Stop/defer:

no progress
duplicate
low information gain
poor reproducibility
high complexity
high failure surface
PIT uncertainty

Possible states:

STOP
DEFER
REDIRECT
SUPERSEDE

---

# 120. RESEARCH SATURATION

Track:

new information rate
OOS improvement rate
failure reduction
coverage improvement

When yield saturates, move to:

adjacent data
new targets
new timing
new model families
unknown frontier

---

# 121. UNKNOWN FRONTIER

Explicitly search:

unknown source
unknown feature
unknown regime
unknown target
unknown timing
unknown failure
unknown interaction
unknown market structure
unknown model family

Do not assume the known feature set is complete.

---

# 122. IDEAL CURRENT PREDICTION OBJECT

Prediction should expose:

Target
Horizon
Prediction cutoff
Data as-of
UP probability
FLAT probability
DOWN probability
Top class
Regime
Model agreement
Predictability
Prediction confidence
Data confidence
Source confidence
PIT confidence
OOD
Failure risk
Fragility
Forecast lifetime
Main supporting evidence
Main opposing evidence
Bull trigger
Bear trigger
Invalidation
Decision
Prediction ID
Model version
Data snapshot
Feature schema
Git SHA
PIT result
Leakage result
Reproducibility

---

# 123. FINAL MATHEMATICAL OBJECT

Conceptually:

Forecast_t(h) =
P(
Y_(t+h),
R_(t+h),
V_(t+h),
D_(t:t+h),
F_(t+h)
|
X_≤t,
S_t,
C_t
)

where:

X = admissible information
S = market state
C = confidence/system condition

---

# 124. FINAL DECISION FUNCTION

A_t =
argmax_a
E[Utility(a, Forecast_t)]
- Cost(a)
- RiskPenalty(a)

subject to:

PIT_PASS
DataIntegrity_PASS
StateConsistency_PASS
policy constraints

---

# 125. FINAL RESEARCH OBJECTIVE

Maximize:

FutureGeneralization
+ Calibration
+ Robustness
+ FailureReduction
+ InformationValue

while minimizing:

Complexity
OperationalRisk
FalseConfidence

subject to:

PIT violations = 0
Leakage = 0
Holdout integrity preserved
Production safety preserved

---

# 126. CURRENT REPOSITORY MAPPING

The current repository already contains dedicated mechanisms for:

autonomous research
watchdogs
PIT/OOS audit
calibration
selective prediction
uncertainty
regime/time research
microstructure
recency challengers
rolling challengers
production prediction
production sentinel
workflow contract tests
recovery
archive/replay
data frontier
research state

Representative workflows:

btc_24h_autonomous_research
btc_24h_watchdog
btc_adaptive_calibration_replay
btc_autonomous_data_frontier
btc_binance_flow_research
btc_binance_ws_collector
btc_calibration_frozen_replay
btc_continuous_supervisor
btc_current_production_prediction
btc_experience_policy_oos
btc_frontier_branch_validation
btc_historical_research
btc_innovative_prediction_control_v2
btc_microstructure_research
btc_pit_oos_audit
btc_production_sentinel
btc_recency_challenger
btc_recency_research
btc_rolling_challenger
btc_selective_prediction_oos
btc_selective_research
btc_time_regime_research
btc_ultimate_final_v13_e2e
btc_uncertainty_layer_oos
btc_unit_tests
btc_watchdog
btc_workflow_contract_tests

The current tree contains 58 workflow files.

---

# 127. CURRENT PRODUCTION MODEL MAPPING

5m:

model_version = bootstrap.soft_ensemble.v5.4
artifact = 5m.joblib
candidate = false
classes = DOWN / FLAT / UP

10m:

model_version = bootstrap.bootstrap_rf
artifact = 10m.joblib
candidate = false
classes = DOWN / FLAT / UP

The runtime performs registry/metadata/artifact/feature checks.

---

# 128. CURRENT PRODUCTION FEATURE MAPPING

The current production metadata exposes 15 principal features:

ret_1m
ret_3m
ret_5m
ret_10m
acceleration
volatility_5m
volatility_10m
range_position_10m
body_1m
upper_wick_1m
lower_wick_1m
volume_ratio
volume_trend
ema_gap_5m
ema_gap_10m

Therefore this master design must not be mistaken for the current Production feature inventory.

---

# 129. CURRENT REPOSITORY EVIDENCE STATE

Latest inspected HEAD:

main
aff1d3dfa437a3b4bd5d31d88366038f4dde580a

Production gate:

HOLD

promotion_allowed:

false

The current promotion artifact records incomplete/invalid robustness/calibration evidence and insufficient required safety/non-regression evidence.

---

# 130. CURRENT PIT EVIDENCE

Current PIT artifact reports:

status = PASS
violations = 0
min_strict_pit_rows = 300

5m strict primary settled = 450
10m strict primary settled = 449

The primary horizon-specific strict gate is therefore populated above its minimum.

Legacy observations remain quarantined rather than silently counted.

---

# 131. CURRENT STRICT-PIT PERFORMANCE SNAPSHOT

5m:

n = 450
accuracy ≈ 0.49556
logloss ≈ 1.05328
brier ≈ 0.63396
ECE ≈ 0.08933

10m:

n = 449
accuracy ≈ 0.39644
logloss ≈ 1.06626
brier ≈ 0.64869
ECE ≈ 0.02641

These are repository evidence snapshots, not guarantees of future performance.

---

# 132. CURRENT LIVE OPERATIONAL SNAPSHOT

Latest inspected live-cycle state:

latest prediction id = 97161
latest prediction time ≈ 2026-10-05 17:31 UTC
latest completed live cycle = 1646
mode = deferred

Recent Actions include queued/pending work as well as cancellations and successes.

Therefore:

operational activity != proven continuous healthy completion

---

# 133. CURRENT RESEARCH LINES

Innovative control v2 includes research surfaces for:

adaptive ensemble
model disagreement
drift-aware routing
future failure predictor
predictability
selective prediction
regime state

It remains:

HOLD / RESEARCH ONLY

Tail-distribution research evaluates:

quantile performance
interval coverage
interval width
tail breaches
newest block behavior

A narrower interval is not automatically better if coverage collapses.

---

# 134. BINARY TARGET RESEARCH

Binary experiments are separate from canonical UP/FLAT/DOWN.

They require:

knowledge-time firewall
chronological WFO
same-block baseline
fold timestamps
accuracy confidence interval
effective sample size
worst/newest fold
non-degraded fold fraction
analysis Git SHA lineage

Binary research does not modify Production without the full local gate chain.

---

# 135. CURRENT STATUS DISCIPLINE

Use distinct states:

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

Do not convert workflow success into Production approval.

---

# 136. FINAL SYSTEM LAWS

1. Future generalization beats historical fit.
2. Current GitHub state beats stale notes.
3. Availability time is not retrieval time.
4. Unknown PIT is not PASS.
5. Missing is not zero.
6. Target semantics are versioned.
7. Mature labels are immutable.
8. Random temporal splits are not Production-grade.
9. Selection and evaluation are separated.
10. Frozen holdout is protected.
11. Probability is not confidence.
12. Predictability is not accuracy.
13. Bullish direction is not equivalent to buy-now.
14. Disagreement is information.
15. Abstention is valid.
16. Failure evidence is retained.
17. Production is a bundle.
18. Automation success is not research success.
19. Complexity requires incremental evidence.
20. The system continuously searches for reasons it is wrong.

---

# 137. ULTIMATE SYSTEM DEFINITION

A continuously self-auditing, PIT-safe, chronologically evaluated, calibrated, uncertainty-aware, regime-aware, multi-model BTC decision system that predicts distributions rather than points, can acquire information or abstain when appropriate, preserves immutable evidence, and improves only through locally validated future-generalization gains.

---

# 138. COMPLETION STANDARD

A task is not complete because:

code exists
CI is green
workflow finished
artifact exists
model runs
prediction exists

Completion requires evidence for:

SPEC
CODE
DATA
PIT
LEAKAGE
OOS/WFO
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

Permanent loop:

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
→ HOLDOUT
→ SHADOW
→ ADOPT / HOLD / REJECT
→ RELEASE
→ PRODUCTION
→ RECONCILE
→ FAILURE ANALYSIS
→ MEMORY
→ NEXT RESEARCH

Final rule:

NO EVIDENCE, NO CLAIM.
NO PIT PROOF, NO HISTORICAL TRUST.
NO ROBUSTNESS, NO PROMOTION.
NO CALIBRATION, NO CONFIDENCE.
NO REPRODUCIBILITY, NO DURABLE KNOWLEDGE.
NO SAFE FALLBACK, NO AUTONOMOUS OPERATION.
