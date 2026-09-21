# BTC Prediction Research — Production Audit (2026-09-18)

## Scope

This audit covers the current repository state of `BTC-Prediction-Research`: market-data adapters, historical research/backtest, live features/modeling, settlement, timestamps/PIT, cache/state persistence, production artifacts, and GitHub Actions.

**No new production data source was added as part of this audit.** External candidates are listed separately as research candidates only.

## Executive assessment

The project has a strong skeleton for chronological research and operational resilience, but it is **not yet a fully point-in-time-safe research system** under a strict production definition.

The largest remaining risks are:

1. Historical data is reconstructed from current APIs/archives without persisted `available_at`, `publication_time`, `retrieved_at`, and `revision_time` provenance for every feature input.
2. The PIT audit currently validates only prediction-level timestamps and optional aggregate `market_data_cutoff_utc`; it cannot prove that every input was actually available by the cutoff.
3. Live prediction currently substitutes numeric zero values when several microstructure/funding/OI fetches fail. This violates the required fail-closed rule for data acquisition.
4. Settlement can fall back from the preferred venue to another venue. That can change the target label and makes the benchmark source-dependent unless the settlement venue is fixed and persisted.
5. The historical research engine uses a three-day publication buffer, which is conservative, but that is not equivalent to PIT provenance and does not protect against later revisions.
6. The historical research dataset currently mixes multiple Binance-derived series. Binance futures/mark/premium/funding/OI are not independent vendors; mark/premium are derivative/reference series from the same venue. ETH/SOL are different underlying assets, not independent source vendors.
7. Production artifact audit fingerprints files and metadata, but does not independently re-load the model and verify that the serialized model's classes/features exactly match the metadata.
8. The current historical report shows a useful OOS sample (64,745 rows; 52,735 evaluated model OOS observations), but checkpoint metrics are not sufficient evidence of a durable production edge by themselves.

## Current data sources / datasets

| Dataset | Current use | Independence | PIT status | Audit decision |
|---|---|---|---|---|
| Binance BTCUSDT USD-M futures 1m klines | historical + live | primary venue/dataset | partial; event time known, retrieval/provenance incomplete | production/conditional with PIT hardening |
| Binance BTCUSDT spot 1m klines | historical + live fallback | same vendor, different market | partial | conditional |
| Binance BTC mark-price klines | historical | derived/related to same Binance futures venue | partial | research/conditional; not independent source |
| Binance BTC premium-index klines | historical | derived/related to same Binance futures venue | partial | research/conditional; not independent source |
| Binance BTC funding rate | historical + live | same Binance futures venue | partial; publication availability not persisted | conditional |
| Binance BTC open interest | historical + live | same Binance futures venue | partial; observation availability not persisted | conditional |
| Binance BTC taker buy/sell volume | live | same Binance futures venue | live retrieval time not persisted per observation | conditional |
| Binance futures order book | live | same Binance futures venue | live snapshot time exists at API level, but retrieved_at is not persisted | research/conditional |
| Bybit BTCUSDT linear 1m klines | live fallback | independent exchange venue | live only; no historical PIT panel | research/conditional |
| Bybit order book | live | independent exchange venue | snapshot timestamp available from API, not persisted in prediction provenance | research/conditional |
| Bybit funding history | live | independent exchange venue | retrieval/publication provenance incomplete | research/conditional |
| Coinbase BTC-USD candles | live fallback/settlement fallback | independent exchange venue | retrieval provenance incomplete | research/conditional |
| Kraken XBTUSD OHLC | live fallback/settlement fallback | independent exchange venue | retrieval provenance incomplete | research/conditional |
| Binance Vision monthly/daily archives | historical bootstrap/research | same Binance underlying dataset; archives/mirrors are not independent sources | event timestamps known, historical retrieval/revision provenance absent | storage/transport only, not independent evidence |
| Binance Vision S3 mirror | historical fallback | **same underlying dataset as Binance Vision** | same limitation | do not count as independent |
| ETH/SOL Binance futures | cross-asset features | different underlying assets, same vendor | partial | conditional research feature |
| X research layer | research-only, no production input | external social source | strict contract exists, but no automated production fetcher | research-only |

## Current feature inventory

### Production ML model (15 features)

`ret_1m`, `ret_3m`, `ret_5m`, `ret_10m`, `acceleration`, `volatility_5m`, `volatility_10m`, `range_position_10m`, `body_1m`, `upper_wick_1m`, `lower_wick_1m`, `volume_ratio`, `volume_trend`, `ema_gap_5m`, `ema_gap_10m`.

### Live structural layer

In addition to the 15 ML features, the live structural overlay uses Binance/Bybit order-book imbalance, cross-exchange gap, taker imbalance, funding, and coverage state. The structural overlay is blended with the calibrated ML probabilities under a bounded holdout-validated weight.

### Historical research feature set

The historical research panel currently contains 41 features, including BTC futures/spot/mark/premium, ETH/SOL relative returns, funding, open interest, calendar encodings, and interaction terms.

**Important:** a larger feature count is not evidence of higher production value. The research standard correctly requires ablations and repeated OOS confirmation.

## Model / backtest audit

Strengths:

- separate 5m/10m targets;
- multiclass DOWN/FLAT/UP probabilities;
- chronological walk-forward evaluation;
- purge/embargo controls;
- nested chronological calibration in the research model comparison;
- Accuracy, LogLoss, Brier and calibration metrics;
- 2k/5k/10k OOS checkpoints;
- candidate adoption gates;
- production incumbent is preserved until a gate is passed;
- deterministic tests cover many failure modes.

Risks:

- the historical panel is not backed by per-observation PIT provenance;
- the current historical research uses current API/archives as if they were historical truth without proving when the value was observable;
- the production model feature metadata is validated mainly by metadata rather than by a full model/metadata consistency audit;
- economic usefulness is documented as a requirement, but the core historical report is still primarily predictive metrics rather than a complete execution-cost model;
- the current 10k checkpoint is a research gate, not an independent final holdout that remains untouched throughout all research iterations.

## Timestamp / PIT audit

The required production schema should distinguish at minimum:

- `event_time`: when the underlying market/network event occurred;
- `available_at`: earliest time the value was legitimately available to the prediction system;
- `publication_time`: when the source officially published/released it, when applicable;
- `retrieved_at`: when this system actually obtained it;
- `prediction_cutoff`: the hard information cutoff for the prediction;
- `revision_time`: when a source later revised/corrected the value, if applicable.

Current system has `created_at_utc`/decision metadata and target times, but does **not** persist this six-field provenance for each source/feature input. Therefore a current historical result cannot be certified as strict PIT merely from timestamps.

### Required PIT rule

For every feature value used at prediction cutoff `C`:

`event_time <= available_at <= C`

and the value must correspond to the source revision that was actually available at `C`. If the source is revision-prone and the historical vintage cannot be reconstructed, the feature must be excluded from strict OOS evaluation.

`retrieved_at` is evidence of our acquisition time, not a substitute for `available_at` or `publication_time`.

## Cache audit

The system has useful retrying and cache mechanisms, including Binance Vision archive caching and live bootstrap cache freshness checks. However, cache entries do not yet carry a complete immutable provenance envelope. A cached row therefore cannot prove which source revision was observed at which retrieval time.

Required cache metadata for strict research:

- source_id / endpoint;
- canonical request parameters;
- raw payload SHA-256;
- event-time range;
- publication_time when supplied by source;
- available_at;
- retrieved_at;
- revision_time or explicit `null` plus `revision_policy`;
- parser/schema version;
- source response timestamp when supplied;
- validation status.

## GitHub Actions audit

The main live workflow is operationally strong: 5-minute schedule, concurrency protection, retries, deterministic tests, state restore/compression round-trip, model/provenance checks, artifact upload, and conflict-safe state commits are all present.

The latest observed live run for commit `417499e...` completed successfully. All 45 job steps completed successfully.

A recent intentional fail-closed regression was also observed: commit `e607445...` failed because one PIT test expected `market_cutoff_after_decision` while the implementation returned `1:market_cutoff_after_decision`. This is a test-contract defect, not evidence that the audit incorrectly accepted a bad cutoff. The following run succeeded after the test contract was corrected.

The Actions layer therefore demonstrates useful fail-closed behavior, but reliability still depends on making test contracts deterministic and on preventing data fallbacks from silently changing semantics.

## No-new-source policy during audit

No new market/data provider was added to the production workflow during this audit. Candidate sources below are research candidates only and must not be wired into production until the PIT/provenance layer is complete.

## Candidate independent information sources — research only

| Candidate | Information | Independence | PIT / revision risk | Initial classification |
|---|---|---|---|---|
| Bybit historical trades/order book/funding/OI | independent exchange microstructure | independent venue | high for historical order-book reconstruction; manageable for trades/funding/OI with exact timestamps | research-only until forward PIT dataset exists |
| Deribit BTC options / IV / order book | options-implied volatility/skew/term structure | independent venue + different derivative market | high; must preserve quote timestamps and contract lifecycle | research-only |
| Mempool.space | mempool, fees, blocks, network activity | independent Bitcoin network source | block/event timestamps are strong; API retrieval/revision provenance still required | research-only |
| FRED/ALFRED | macro releases and vintages | independent macro source | **ALFRED vintages are valuable for revision-safe research**; publication/release timing must be stored | conditional research |
| Stablecoin supply/flows from a primary issuer or carefully versioned on-chain source | liquidity regime | independent information domain | revision/entity attribution risk | research-only |
| CryptoQuant / Glassnode-class on-chain datasets | exchange flows, holder/network metrics | independent information domain | usually paid; strong revision/PIT requirements | research-only |
| Polymarket/Kalshi BTC prediction-market prices | market-implied short-horizon probability | independent market | very high PIT requirement; must snapshot book/price before cutoff | research-only |
| X/public social data | narrative/sentiment | independent information domain | publication/edit/delete/retrieval problems | research-only |

No candidate is promoted merely because it is available, popular, or correlated with BTC.

## Recommended remediation order

1. Implement a source-neutral PIT provenance envelope with the six required timestamps.
2. Make live data acquisition fail closed for every feature that materially affects the prediction; never replace an unavailable numeric observation with `0`.
3. Persist per-source retrieval status and exact raw-response hashes in each prediction scenario.
4. Make the settlement benchmark explicit and stable; do not silently switch the target source across venues.
5. Rebuild strict historical research panels with PIT provenance before using them for candidate promotion.
6. Split the 10k research gate from a frozen independent holdout that cannot be used for feature/model selection.
7. Add ablation reports by feature group and by market regime before any feature promotion.
8. Only after the above is verified should candidate-source collection begin.

## Bottom line

The current project is operationally mature enough to continue accumulating live OOS data, but it is **not yet safe to claim that every historical feature is strictly point-in-time reproducible**. The next engineering priority is provenance/PIT hardening, not more data sources.
