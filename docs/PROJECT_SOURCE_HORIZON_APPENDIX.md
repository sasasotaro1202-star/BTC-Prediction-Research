# Project Source Horizon Appendix

The canonical project source keeps 5m/10m as the primary BTC classification horizons and allows additional horizon research. The implementation now instantiates that allowance explicitly as 15m, 30m, 1h, 3h, 6h, 12h, and 24h.

These seven horizons are currently RESEARCH_ONLY baselines. Their predictions are persisted, settled, audited, and monitored, but they are not treated as Production evidence and cannot alter the 5m/10m Champion.

Before promotion, each horizon must pass the same local PIT -> OOS -> robustness -> holdout -> shadow -> explicit promotion sequence.
