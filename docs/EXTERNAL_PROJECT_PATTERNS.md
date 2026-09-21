# External project patterns adopted conservatively

This document records reusable ideas inspected from public projects without treating their reported performance as evidence for this repository.

## Purged / embargoed validation

The public `eslazarev/purged-cross-validation` project emphasizes row-level purging, embargo, expanding/rolling walk-forward validation, CPCV paths, and explicit tests of fold boundaries. Its central lesson is methodological: honest validation should prevent overlapping label windows and residual temporal dependence from contaminating evaluation.

Repository: https://github.com/eslazarev/purged-cross-validation

Adopted here:
- chronological walk-forward rather than shuffled CV;
- explicit purge/embargo tests;
- fail-closed boundary checks;
- no promotion from a single apparent winning backtest.

Not adopted blindly:
- CPCV or additional model-selection machinery without evidence that it improves the deployment design;
- any published performance number from an external repository.

## Production integrity pattern

The repository's own audit showed that operational resilience can become dangerous when fallbacks silently change the semantics of the prediction. Therefore fallback availability is now treated as a state transition, not as an invisible numeric fill.

Adopted here:
- missing critical inputs stop directional prediction;
- zero/default substitution is prohibited for critical live inputs;
- production settlement is fixed to Binance BTCUSDT USD-M futures;
- serialized production models are reloaded and probed before acceptance.

## Research-source independence

Multiple APIs, mirrors, wrappers, archives, and aggregators are not counted as independent evidence when they reproduce the same underlying dataset. Independence is assessed at the underlying information source / venue / measurement process level.

## Promotion rule

External ideas are research hypotheses only. A candidate must survive:
1. strict PIT provenance;
2. chronological walk-forward OOS;
3. independent frozen holdout;
4. multiple market regimes;
5. feature-group ablation;
6. economic-cost sensitivity where relevant;
7. reproducible artifact and source provenance checks.

No feature or data source is promoted merely because it increases in-sample accuracy, headline accuracy, correlation, or backtest profit.
