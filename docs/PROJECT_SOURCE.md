# BTC-Prediction-Research — Project Source

## Verified state 2026-10-03

This section is a dated audit snapshot, not a live HEAD pin. Current GitHub main is re-checked every run and remains authoritative over this historical snapshot.

Recent verified implementation sequence:
- fixed restored/pre-extension trajectory database compatibility and canonical `settled_<horizon>_at_utc` migration;
- verified the repaired trajectory path in `BTC Live Cycle #1357` (`37097981489`), which completed successfully and generated prediction `96971`;
- added all-horizon trajectory coverage and aligned extended settlement schema names;
- fixed `src/settle.py` missing `math` import discovered by `BTC Live Cycle #1359` (`37098473703`);
- added regression coverage for extended settlement finite-probability validation and activated two previously hidden settlement fallback tests;
- separated dashboard packaging from GitHub Pages enablement so dashboard site packaging can succeed even when Pages deployment is blocked.

Current main at this snapshot is `3b4daa94e8a147e65f6a6ab6bc2d88d311832199`. Latest code-bearing settlement fix is `4ac119a3044b06328b929f2228a15d464d282771`; the current HEAD also contains test-only activation/coverage commit `3b4daa94e8a147e65f6a6ab6bc2d88d311832199`.

Verification evidence:
- `BTC Unit Tests #2055` on `e10eca658920b5c31fc19c9bb1bcfef5092d556f`: SUCCESS.
- `BTC Ops Preflight #408` on `4ac119a3044b06328b929f2228a15d464d282771`: SUCCESS.
- `BTC Unit Tests #2062` on current HEAD `3b4daa94e8a147e65f6a6ab6bc2d88d311832199`: IN PROGRESS at this snapshot; do not treat as verified until completion.
- `BTC Live Cycle #1357` on `9c66a8236271326e5db1f11fb1bcf92ef7b58108`: SUCCESS, latest published prediction `96971`.
- `BTC Live Cycle #1359` on `e10eca658920b5c31fc19c9bb1bcfef5092d556f`: FAILED during settlement because `math` was referenced without import. This remains recorded as a failure; the import fix is now in `4ac119a3044b06328b929f2228a15d464d282771`.
- `BTC Prediction Dashboard #29`: packaging job SUCCESS and `github-pages` artifact created (non-expired); the Pages deployment leg remains subject to the existing GitHub integration permission block.

Published production state remains:
- 5m: `bootstrap.soft_ensemble.v5.4`
- 10m: `bootstrap.bootstrap_rf`
- latest published prediction: `96971`
- promotion state: HOLD.

Latest strict PIT/OOS checkpoint remains:
- status `PASS_WITH_LEGACY_UNVERIFIED`;
- Binance-primary verified rows `274/300`;
- violation count `0`;
- recent 20-prediction windows remain strict-PIT clean for 5m and 10m;
- legacy unverified observations remain quarantined.

Latest performance snapshot after prediction `96971`:
- 5m final: accuracy `0.3851468`, LogLoss `1.1548209`, Brier `0.7005714`, ECE `0.0857422`, n=`579`;
- 5m strict-PIT: accuracy `0.4578755`, LogLoss `1.0794920`, Brier `0.6508760`, ECE `0.0570938`, n=`273`;
- 10m final: accuracy `0.4214162`, LogLoss `1.1476375`, Brier `0.6906639`, ECE `0.0842570`, n=`579`;
- 10m strict-PIT: accuracy `0.4285714`, LogLoss `1.0385593`, Brier `0.6321078`, ECE `0.0157548`, n=`273`.

Extended horizons `15m, 30m, 1h, 3h, 6h, 12h, 24h` remain research-only with no settled observations yet. No promotion effect is derived from their dashboard presence.

GitHub Pages deployment remains BLOCKED at the platform/integration permission layer. The new dashboard workflow now isolates site packaging from that deployment step, preserving a verifiable `github-pages` artifact without relabeling the Pages deployment as successful.

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
