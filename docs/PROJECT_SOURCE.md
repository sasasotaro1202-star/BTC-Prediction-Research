# BTC-Prediction-Research — Project Source

## Verified state 2026-10-03

This section is a dated audit snapshot, not a live HEAD pin. Current GitHub main is re-checked every run and remains authoritative over this historical snapshot. The latest code-bearing fix recorded here is `f6c5b6319d2ff3913e4fbaeb0af5e115aa0096b6`; subsequent source-state documentation commits are intentionally treated as docs-only state updates.

Recent implementation work includes:
- repaired GitHub Pages workflow YAML quoting;
- mobile dashboard idempotent refresh, 7-day range, stale-data indication and gap-safe plotting;
- explicit extended-horizon pending state;
- vector/mapping probability compatibility for extended research baselines;
- legacy-schema compatibility for the experience ledger and trajectory exporter;
- restored-DB trajectory regression coverage;
- fixed the trajectory exporter migration path so every requested horizon now additively creates its canonical `settled_<horizon>_at_utc` column before the SELECT.

Published production state at the latest completed successful Live Cycle checkpoint remains:
- 5m: `bootstrap.soft_ensemble.v5.4`
- 10m: `bootstrap.bootstrap_rf`
- latest published prediction: `96970`
- latest completed successful prediction cycle: `#1345` (`37095010446`), completed 2026-10-03T04:01:12Z
- promotion state remains HOLD.

Manual Live Cycles `#1353`, `#1354`, `#1355` and `#1356` failed on pre-fix revisions at trajectory export. The production-critical guard suite reached the trajectory step, where restored legacy databases lacked the extended settlement columns. The fix is now present in `f6c5b6319d2ff3913e4fbaeb0af5e115aa0096b6`; no historical prediction rows were rewritten.

Verification evidence after the fix:
- `BTC Unit Tests #2049` on `f6c5b6319d2ff3913e4fbaeb0af5e115aa0096b6`: queued at this snapshot and therefore NOT YET VERIFIED.
- `BTC Ops Preflight #405` on the same SHA: queued at this snapshot and therefore NOT YET VERIFIED.
- The pre-fix Live Cycle failures remain recorded as failures and are not relabeled as success.
- A fresh successful Live Cycle on the fixed SHA has not yet been observed in this snapshot.

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
