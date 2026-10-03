# BTC-Prediction-Research — Project Source

## Verified state 2026-10-03
main latest observed HEAD after re-check: ff251b15fdb0300f1ff70918e472d3f51c772301. Latest observed repository work also includes a fix repairing dashboard workflow YAML quoting.
README documents 5m as Production Champion; 10m/15m/30m/1h/3h/6h/12h/24h are available, while 15m–24h remain research-only until independent PIT/OOS/robustness/holdout evidence supports promotion.

## Canonical rules
- BTC is a continuous market; do not apply weekday/market-close assumptions from equities.
- Horizon is part of target identity. Evidence for 5m cannot be silently reused to promote 24h.
- Distinguish observation time, source availability/publication, retrieval time, prediction cutoff and later revisions.
- Later candles, future funding/order-book state, later corrections or post-cutoff dashboard state must never leak backward.
- Preserve probability, expected-return/range outputs, uncertainty, state and outcome reconciliation per instrument-horizon-cutoff.
- Evaluate by horizon, recent regime, volatility, liquidity, missingness, source removal, feature deletion, OOD and shock periods.
- Source mirrors/wrappers of one upstream are not independent evidence.
- Production status is derived from release evidence, not dashboard/artifact existence.
