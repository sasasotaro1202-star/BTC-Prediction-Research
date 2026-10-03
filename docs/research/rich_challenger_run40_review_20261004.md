# Rich Production Challenger run 40 review

- Run: 37159633464
- Head SHA: fa73ea538431f1a841b41a7e9fbe6c46d5f0b3e7
- Artifact: btc-rich-production-challenger
- Artifact SHA-256: 27c78bddb670a1ef5f2da80a9c251ebb84d758640cb129a15514900e8c1504cd
- Generated: 2026-10-03T22:50:14.988566+00:00
- Research only: true
- Production changed: false
- PIT evidence: NON_STRICT_ARCHIVE_TIMING
- Promotion evidence eligible: false

## 5m frozen holdout

| Metric | Champion | Rich Challenger | Delta |
|---|---:|---:|---:|
| Accuracy | 0.471396941 | 0.427526904 | -0.043870037 |
| LogLoss | 1.028348491 | 1.040101184 | +0.011752693 |
| Brier | 0.624098414 | 0.632342232 | +0.008243818 |
| ECE | 0.059798143 | 0.004311140 | -0.055487003 |

Stability vs Champion: 0.05 non-worse accuracy fraction across 20 blocks.

Decision: REJECTED for current 5m path. The large ECE improvement does not offset worse Accuracy, LogLoss, and Brier.

## 10m frozen holdout

| Metric | Champion | Legacy retrain | Rich Challenger |
|---|---:|---:|---:|
| Accuracy | 0.436016273 | 0.441165868 | 0.448272311 |
| LogLoss | 1.012057496 | 1.003283206 | 1.001323299 |
| Brier | 0.617756053 | 0.613357376 | 0.611191526 |
| ECE | 0.012541155 | 0.007941668 | 0.021749351 |

Rich vs Champion:
- Accuracy delta: +0.012256038
- Accuracy relative gain: +2.8109%
- LogLoss relative gain: +1.0606%
- Brier relative gain: +1.0626%
- Stability non-worse accuracy fraction: 0.75 across 20 blocks

Decision: research-interesting but not promotion-eligible. The observed gains are below the project's 3% Accuracy / 3% LogLoss relative promotion benchmarks, and PIT evidence is explicitly non-strict.

## Reproducibility note

This review is a durable summary of the GitHub Actions artifact from run 37159633464. It does not replace the original artifact, frozen holdout, or strict-PIT evidence. No production model or registry state is modified by this record.
