# Research Notes

The historical research workflow is intentionally treated as an evidence-generation pipeline, not an automatic reason to change production.

Current priorities:

- eliminate look-ahead and horizon-overlap leakage;
- keep 5m and 10m evaluation independent;
- stabilize probability outputs rather than react to every small price tick;
- compare against simple baselines;
- inspect regime-specific failures;
- test feature ablations;
- evaluate probabilistic quality, not just directional accuracy;
- include costs/slippage when assessing economic usefulness;
- retain rejected candidates for auditability;
- require the 2k/5k/10k OOS adoption gates before production replacement.

A high apparent short-horizon accuracy is not sufficient evidence of an exploitable edge. Recent BTC research demonstrates that apparently strong short-horizon classification can disappear once realistic costs and honest walk-forward evaluation are applied. 
