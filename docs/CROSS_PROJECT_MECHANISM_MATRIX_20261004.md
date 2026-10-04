# Cross-project mechanism matrix — 2026-10-04

This document records mechanism-level references only. No performance, OOS, holdout, production, prediction, or data result from another repository is transfer evidence for BTC.

| Repository | Current main HEAD | Mechanism reference | BTC treatment |
|---|---|---|---|
| BTC-Prediction-Research | ba8412be784d4ef56846aaab5ccee6b303d4c1d1 | strict PIT, production bundle, resumable flow cache, per-horizon readiness | Canonical local implementation |
| 7-Sport-Prediction-Research | 2160f5a9538502d84249e3d4885241c388f9be62 | fail-closed pre-event enrichment, success-only checkpoints, single-writer critical state, cluster-aware evaluation | Reference mechanism; local adaptation required |
| Soccer-Prediction-Research | f2bed57d6e0ad50c888903f9df7c5a1b3339e59b | feature-level PIT lineage, mature-prior training, predictability/failure risk | Feature PIT guard added locally; other mechanisms research/extension |
| Baseball-Prediction-System | e7c785cef3d2c2a8fe250918e642b9d387cf69fe | universal source/data contract, explicit readiness, experience integrity, temporal conformal maturity | Governance reference; target semantics remain BTC-specific |
| Stock-Daily-Prediction-3000 | 0145f2daff68c0816e20bbff1aeb522a742e68e7 | nested prequential selection, contiguous prior folds, moving-block bootstrap, selection evidence, run provenance | Selection/provenance reference; local OOS required |

## Transfer contract

DISCOVER → ABSTRACT_MECHANISM → COMPATIBILITY → LOCAL_IMPLEMENTATION → TEST → LOCAL_PIT → LOCAL_OOS/WFO → ROBUSTNESS → LOCAL_FROZEN_HOLDOUT → SHADOW → PROMOTION

## Implemented in this alignment update

- Feature-level PIT firewall in src/model_compare.py with regression tests.
- Production Artifact Audit provenance manifest binding policy/config hashes to the audited SHA.
- Cross-project governance and current mechanism matrix in the canonical source layer.
- Resumable Binance Flow collector serialization and no-stale-overwrite controls remain in the current main branch.

## Still research-frontier / not claimed as locally validated

- Full event-cluster bootstrap across all BTC prediction revisions.
- Full outcome-maturity metadata propagation through every experience/failure-risk surface.
- End-to-end nested training-window/router selection with an independent BTC holdout.
- Production promotion: no change in this update.
