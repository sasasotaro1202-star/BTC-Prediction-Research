# BTC-Prediction-Research — Project Source

## Verified state 2026-10-03

Current main HEAD at this refresh is `410109fa178fcb7d18ae006823de41db85d67555`. Recent implementation work includes:
- repaired GitHub Pages workflow YAML quoting;
- mobile dashboard idempotent refresh, 7-day range, stale-data indication and gap-safe plotting;
- explicit extended-horizon pending state;
- vector/mapping probability compatibility for extended research baselines;
- legacy-schema compatibility for the experience ledger and trajectory exporter;
- dashboard contract and restored-DB migration tests.

Published production state at the latest completed Live Cycle checkpoint remains:
- 5m: `bootstrap.soft_ensemble.v5.4`
- 10m: `bootstrap.bootstrap_rf`
- latest published prediction: `96970`
- latest completed successful prediction cycle: `#1345` (`37095010446`), completed 2026-10-03T04:01:12Z
- promotion state remains HOLD.

A later manual Live Cycle `#1353` reached the production-critical guard suite successfully, then failed at trajectory export because the restored database lacked extended settlement columns. That failure is the reason for the current explicit trajectory-side additive migration fix. No historical rows were rewritten.

Unit Test evidence:
- `BTC Unit Tests #2041`: 697 passed, 1 failed; the sole failure was a brittle dashboard contract assertion.
- `BTC Ops Preflight #403`: SUCCESS after the workflow YAML repair.
- `BTC Unit Tests #2044`: queued at the time of this source refresh after the restored-DB trajectory fix.
- Do not treat queued/in-progress runs as verified.

Strict PIT checkpoint remains HOLD because Binance-primary verified rows are `273/300`. The recent 20-prediction window is strict-PIT clean for both 5m and 10m, while legacy pre-contract rows remain quarantined.

Dashboard horizon coverage is `5m, 10m, 15m, 30m, 1h, 3h, 6h, 12h, 24h`. 5m and 10m remain Primary; 15m–24h remain research-only. Dashboard code does not alter target, model, calibration, PIT or promotion state.

GitHub Pages deployment is currently BLOCKED at the platform/integration permission layer: `configure-pages@v5` with `enablement: true` reached the site-creation attempt but GitHub returned `Resource not accessible by integration`. This is not treated as a successful deployment.

## Canonical rules
- BTC is a continuous market; do not apply weekday/market-close assumptions from equities.
- Horizon is part of target identity. Evidence for 5m cannot be silently reused to promote 24h.
- Distinguish observation time, source availability/publication, retrieval time, prediction cutoff and later revisions.
- Later candles, future funding/order-book state, later corrections or post-cutoff dashboard state must never leak backward.
- Preserve probability, expected-return/range outputs, uncertainty, state and outcome reconciliation per instrument-horizon-cutoff.
- Evaluate by horizon, recent regime, volatility, liquidity, missingness, source removal, feature deletion, OOD and shock periods.
- Source mirrors/wrappers of one upstream are not independent evidence.
- Production status is derived from release evidence, not dashboard/artifact existence.

## Cross-project governance alignment — 2026-10-03

The five-repository research set is:
- Baseball-Prediction-System
- BTC-Prediction-Research
- 7-Sport-Prediction-Research
- Soccer-Prediction-Research
- Stock-Daily-Prediction-3000

Cross-project transfer is mechanism-level only: DISCOVER → ABSTRACT_MECHANISM → COMPATIBILITY → ADAPT → LOCAL_PIT → LOCAL_OOS/WFO → ROBUSTNESS → LOCAL_FROZEN_HOLDOUT → SHADOW → PROMOTE.


A green workflow, artifact existence, model-file existence or external performance claim is not performance verification. Failures/cancellations/skips remain failures/cancellations/skips unless independently rerun and verified. Historical results and holdouts are not rewritten. Cost-unknown, billing-risk or paid-only sources remain HOLD/UNCONFIRMED.
