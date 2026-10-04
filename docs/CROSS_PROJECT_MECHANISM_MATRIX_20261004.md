# Cross-project mechanism matrix — 2026-10-04

HEADs verified against each repository's `main` ref at the time of this synchronization.

This document records mechanism-level references only. No performance, OOS, holdout, production, prediction, or data result from another repository is transfer evidence for BTC.

| Repository | Current main HEAD | Mechanism reference | BTC treatment |
|---|---|---|---|
| BTC-Prediction-Research | a20179a9612a083fd3eda50e284f4bcd0cd690b4 | strict PIT, production bundle, resumable flow cache, per-horizon readiness | Canonical local implementation |
| 7-Sport-Prediction-Research | b6a6934b3de15b58d66c7b871e8e7aed51129e22 | fail-closed pre-event enrichment, success-only checkpoints, single-writer critical state, cluster-aware evaluation | Reference mechanism; local adaptation required |
| Soccer-Prediction-Research | 87adb6d814918ca751b9084bf1efe86f35532347 | feature-level PIT lineage, mature-prior training, predictability/failure risk | Feature PIT guard added locally; other mechanisms research/extension |
| Baseball-Prediction-System | fdf665a341047cbd5fe8cac4b737b2bf1f40e4fc | universal source/data contract, explicit readiness, experience integrity, temporal conformal maturity | Governance reference; target semantics remain BTC-specific |
| Stock-Daily-Prediction-3000 | f92362fe138a8fb443da58fe91aecfbc4f4f8d70 | nested prequential selection, contiguous prior folds, moving-block bootstrap, selection evidence, run provenance | Selection/provenance reference; local OOS required |

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