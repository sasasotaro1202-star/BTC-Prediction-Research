# BTC Forecast Horizon Policy

## Primary Production horizons

5m and 10m remain the Production Champion horizons.

## Extended Research horizons

The live system now generates and settles these additional horizons on the same prediction event:

- 15m
- 30m
- 1h
- 3h
- 6h
- 12h
- 24h

They use the explicit research-only baseline method `research_structural_time_scaled_v1`.

Each extended forecast records its target timestamp, target-definition version, probabilities for DOWN/FLAT/UP, research-only state, and unvalidated/uncalibrated state.

The target is the next UTC grid boundary for the requested horizon.

## Promotion boundary

Extended horizons do not enter the 5m/10m Production bundle or automatic promotion path. A future learned model must pass local PIT, chronological OOS, robustness, protected holdout, shadow, and explicit promotion checks before becoming Production-grade.

Historical prediction records are never rewritten when the horizon policy evolves.
