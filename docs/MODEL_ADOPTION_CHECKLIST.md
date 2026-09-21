# Model Adoption Checklist

Before changing production for either 5m or 10m:

- [ ] Candidate evaluated only with chronological OOS predictions.
- [ ] Purge/embargo applied for the forecast horizon.
- [ ] Same OOS observations used for incumbent and candidate.
- [ ] Accuracy checked.
- [ ] LogLoss improved.
- [ ] Brier improved.
- [ ] ECE/calibration not materially worse.
- [ ] Performance checked by volatility/regime.
- [ ] Performance checked by time block.
- [ ] Baseline/random-walk comparison completed.
- [ ] Candidate-search multiplicity accounted for.
- [ ] Appropriate statistical loss-difference test completed.
- [ ] 2,000-OOS checkpoint passed.
- [ ] 5,000-OOS checkpoint passed.
- [ ] 10,000-OOS checkpoint passed.
- [ ] Improvement reproduced across checkpoints.
- [ ] Economic test includes fees/slippage where applicable.
- [ ] No feature/model leakage found.
- [ ] Candidate artifact retained for audit.

Only after all applicable boxes are satisfied may the candidate replace production.
