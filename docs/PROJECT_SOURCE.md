# BTC-Prediction-Research — Project Source

## Verified state 2026-10-03

This section is a dated audit snapshot, not a live HEAD pin. Current GitHub main is re-checked every run and remains authoritative over this historical snapshot.

Recent verified implementation sequence:
- repaired restored/pre-extension trajectory database compatibility and canonical `settled_<horizon>_at_utc` migration;
- fixed `src/settle.py` runtime dependency omission (`math`);
- added extended settlement regression coverage and activated previously hidden fallback tests;
- separated Dashboard packaging from GitHub Pages deployment so packaging remains independently verifiable when Pages configuration is blocked;
- verified the repaired end-to-end Live Cycle path through `BTC Live Cycle #1357` and then `#1360`;
- #1360 also verified the post-prediction trajectory refresh path, producing 9-horizon chart data without fabricating missing outcomes.

Latest research-state commit is `fd84c10d1e5599ad6db8d43965793a266b68adda` (Merge BTC research state safely). The latest code-bearing settlement fix is `4ac119a3044b06328b929f2228a15d464d282771`.

Verification evidence:
- `BTC Unit Tests #2063` on `b7d0c4a787e271cdbf8b4c2c2c6e254f5dcaa809`: SUCCESS, 708 passed.
- `BTC Ops Preflight #408` on `4ac119a3044b06328b929f2228a15d464d282771`: SUCCESS.
- `BTC Live Cycle #1357` on `9c66a8236271326e5db1f11fb1bcf92ef7b58108`: SUCCESS, prediction `96971`.
- `BTC Live Cycle #1359` on `e10eca658920b5c31fc19c9bb1bcfef5092d556f`: FAILED during settlement because `math` was not imported; this remains an immutable failure record.
- `BTC Live Cycle #1360` on `a6de398420dd8f523e7b3a63a1210374ce7fc2df`: SUCCESS; prediction `96972`; settlement reported 5 fields resolved; 9-horizon trajectory export succeeded; state was safely merged back to main as `fd84c10d...`.
- `BTC Unit Tests #2062`: FAILED on a test fixture that omitted required 5m/10m settlement columns; this was corrected and the subsequent #2063 run passed 708 tests.
- `BTC Prediction Dashboard #29`: dashboard package job SUCCESS and `github-pages` artifact generated; deployment configuration still FAILED due the existing GitHub integration Pages permission block.
- `BTC Production Sentinel #868`: SUCCESS.
- `BTC 24H Autonomous Research #32`: queued at this snapshot; it is research-only and must not be treated as completed evidence.

Published production state remains:
- 5m: `bootstrap.soft_ensemble.v5.4`
- 10m: `bootstrap.bootstrap_rf`
- latest published prediction: `96972`
- production model/artifact bindings unchanged;
- promotion state: HOLD.

Latest strict PIT/OOS checkpoint:
- status `PASS_WITH_LEGACY_UNVERIFIED`;
- Binance-primary verified rows: `275/300`;
- strict primary settled: 5m=`274`, 10m=`274`;
- violation count: `0`;
- recent 20-prediction windows remain strict-PIT clean;
- 169 legacy observations remain quarantined;
- PIT is therefore not yet fully verified for promotion.

Latest performance snapshot after prediction `96971` settlement update:
- 5m final: accuracy `0.3844828`, LogLoss `1.1547774`, Brier `0.7005482`, ECE `0.0861810`, n=`580`;
- 5m strict-PIT: accuracy `0.4562044`, LogLoss `1.0796748`, Brier `0.6510085`, ECE `0.0556437`, n=`274`;
- 10m final: accuracy `0.4206897`, LogLoss `1.1481358`, Brier `0.6909758`, ECE `0.0834541`, n=`580`;
- 10m strict-PIT: accuracy `0.4270073`, LogLoss `1.0400122`, Brier `0.6329819`, ECE `0.0170895`, n=`274`.

Research diagnostics from the current settled ledger:
- 5m recent-100 accuracy `0.48`; the weakest tracked 5m slice is confidence `0.50-0.60` with n=`80` and accuracy `0.275`;
- 10m recent-100 accuracy `0.43`; recent weak slices include predicted FLAT (n=`31`, accuracy `0.3226`) and hour JST `9` (n=`32`, accuracy `0.34375`);
- current uncertainty/disagreement research artifacts remain research-only and have not shown sufficient evidence for production replacement.

Extended horizons:
- trajectory forecast points now exist for all `15m, 30m, 1h, 3h, 6h, 12h, 24h` horizons;
- settled outcome counts are currently 15m=`1`, 30m=`1`, 1h=`1`, 3h=`0`, 6h=`0`, 12h=`0`, 24h=`0`;
- all extended horizons remain research-only and unvalidated for promotion.

GitHub Pages deployment remains BLOCKED at the platform/integration permission layer. Dashboard packaging is independently verified, but Pages publication is not treated as successful.

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
