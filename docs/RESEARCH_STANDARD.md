# BTC Research Standard

This project does not promote a model to production from a single backtest or a high raw accuracy number.

## Required validation

1. Strict chronological walk-forward OOS evaluation.
2. Purge/embargo around test windows to prevent horizon overlap leakage.
3. Separate 5-minute and 10-minute models/evaluations.
4. Multiclass UP / FLAT / DOWN probability forecasts.
5. Evaluate Accuracy, LogLoss, Brier score, and ECE/calibration.
6. Compare against simple baselines, including persistence/random-walk and class-frequency baselines.
7. Evaluate by volatility/regime, confidence bucket, and calendar period.
8. Run ablations so feature improvements are attributable rather than accidental.
9. Correct for multiple candidate-model comparisons and use a forecast-loss statistical test appropriate to the nested benchmark where applicable.
10. Include transaction costs/slippage when judging economic usefulness; predictive accuracy alone is insufficient.

## Adoption gates

Historical research is accumulated through three checkpoints:

- 2,000 OOS forecasts: preliminary gate.
- 5,000 OOS forecasts: robustness gate.
- 10,000 OOS forecasts: final adoption gate.

A candidate must improve the incumbent on the same OOS observations, with improvements in probabilistic metrics and reproducibility across checkpoints. A candidate that wins only on Accuracy while worsening LogLoss/Brier/calibration is rejected. A candidate that improves only one regime or one short window is not sufficient.

No production adoption is allowed before the 10,000-OOS gate and statistical/reproducibility confirmation.

## Live system

Live predictions continue to accumulate separately from historical research. Historical backtests are used to accelerate research, but they do not replace the live OOS record.

## Anti-overfitting rules

- Never use future observations to construct a feature at prediction time.
- Fit scalers, imputers, calibrators, and model parameters inside each training fold.
- Do not tune hyperparameters on the final OOS block.
- Preserve the incumbent model until a candidate passes every gate.
- Keep rejected candidates and their metrics for auditability.
- Prefer simpler models when performance is statistically indistinguishable.

## Interpretation

Short-horizon BTC direction can be close to random after costs. The research objective is therefore to falsify weak edges, not to manufacture a high accuracy number. Published recent work similarly emphasizes genuine walk-forward validation, leakage control, and the danger that raw predictive metrics can disappear after realistic costs. 
