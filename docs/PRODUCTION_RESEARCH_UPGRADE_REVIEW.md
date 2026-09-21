# Production Research Upgrade Review

This document records external techniques reviewed against the BTC prediction system. The rule is **adopt only when the technique is independently useful, PIT-safe, and improves chronological OOS evidence**.

## Reviewed techniques

- Purged/embargoed walk-forward validation — already present in `src/research_protocol.py`.
- Probability calibration — already present through temperature calibration and chronological holdout.
- Data provenance / PIT metadata — already present through `pit_schema.py`, prediction provenance fields, and artifact audits.
- Idempotent/fault-tolerant orchestration — already present in GitHub Actions with state archives, retries, concurrency control, and conflict-safe state merging.
- Regime-aware evaluation — already present as regime diagnostics; promotion remains gated by OOS evidence.
- Online learning / ADWIN / Adaptive Random Forest — **not promoted into production yet**. It is kept as a shadow-model research candidate because online adaptation can overreact to transient microstructure and create live-training leakage if settlement timing is mishandled.

## External references

- BTC-NQ-Cross-Market-Validation: https://github.com/cliprob/BTC-NQ-Cross-Market-Validation
- River online ML: https://github.com/online-ml/river
- Crypto Flash Crash Predictor: https://github.com/sweety-mahale/Crypto-Flash-Crash-Predictor
- LeakageDetector: https://arxiv.org/html/2503.14723v1
- dataprov: https://github.com/RI-SE/dataprov
- Prefect retries: https://docs.prefect.io/v3/how-to-guides/workflows/retries
- Dagster data ingestion: https://docs.dagster.io/examples/full-pipelines/ml/data-ingestion
- Microstructure alpha paper: https://www.frontiersin.org/journals/blockchain/articles/10.3389/fbloc.2026.1811716/full
- Regime-aware Bitcoin forecasting paper: https://link.springer.com/article/10.1007/s10614-026-11338-3

## Production decision

Do not add a new model merely because an external project uses it. A candidate must beat the current production baseline on chronological OOS data across multiple regimes, while preserving PIT integrity and calibration quality. If it does not, it remains research-only.

## Current operational hardening

A broad Binance critical-data outage now causes a **safe deferred cycle** rather than a fabricated prediction or an unnecessarily red GitHub Action. Partial endpoint failures or code/data-contract errors still fail closed. This separates external transient availability from genuine correctness failures.
