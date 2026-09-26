# BTC Data Source Frontier

Research inventory for expanding the BTC Event Layer beyond the current Binance
WS kline/depth inputs. This document is research-only and does not authorize
production ingestion or promotion.

Legend:
- **A** = directly useful for 5m/10m case-level prediction
- **B** = useful mainly for regime/context or slower horizons
- **C** = useful for discovery/monitoring/secondary confirmation
- PIT = point-in-time provenance difficulty: Low / Medium / High
- Free status is based on the currently documented access model; any source
  whose free access depends on an account, limited trial, or changing website
  behavior must remain research-only until verified in the project.

## 1. Exchange microstructure

| Source | Data | Value | PIT |
|---|---|---:|---:|
| Binance USD-M Futures WS | aggTrade, depth, liquidations, klines, mark | A | Low |
| Bybit WS | L1/L50/L200/L1000 orderbook, trades, liquidations, tickers | A | Low |
| OKX WS/REST | trades, books, candles, funding, OI, mark/index | A | Low |
| Coinbase WS | BTC spot trades / L2 | A | Low |
| Deribit public API | BTC options/futures book, trades, OI, IV-related fields | A/B | Medium |
| SpiralDevelopment/crypto-hft-data | multi-exchange WS collection reference | A (implementation reference) | Low |
| emschutt/crypto-market-data-research-engine | Binance aggTrade/depth, L5 reconstruction, microstructure features | A (implementation reference) | Low |
| KhavrTrading/flowex | Binance/Bybit/Bitget streaming + 75 depth metrics | A (implementation reference) | Low |

Binance currently documents 100ms aggregate trades and 1s all-market liquidation
snapshots, including event time and trade/liquidation fields. Bybit exposes
multiple public orderbook depths with exchange timestamps and explicit
snapshot/delta sequencing. OKX documents public market data including orderbook,
trades, candles, funding, open interest and mark prices.

## 2. Derivatives / positioning

| Source | Data | Value | PIT |
|---|---|---:|---:|
| Binance Futures REST | OI history, global/top long-short ratios, taker long/short | A | Low |
| Bybit REST/WS | funding, OI, long/short, liquidation | A | Low |
| OKX REST/WS | funding history, OI, mark, orderbook/trades | A | Low |
| Deribit | options OI, volume, orderbook, IV/skew-related state | A/B | Medium |
| CryptoQuant Basic | price/OHLCV/funding/OI/liquidations; 30d and daily on free tier | B | Medium |
| CoinGlass | cross-exchange derivatives + options + liquidation + on-chain | A/B | Medium/High |
| ChainVector | normalized derivatives, options, macro reference data | B | Medium/High |

## 3. Bitcoin on-chain / network

| Source | Data | Value | PIT |
|---|---|---:|---:|
| Bitcoin Research Kit / Bitview | 8,000+ local metrics, UTXO, holder cohorts, fees, mining, mempool | B | Medium |
| mempool.space | mempool, fees, blocks, mining, Lightning, live WebSocket | B | Low/Medium |
| Blockchain.com APIs | blocks, transactions, hash rate, difficulty, fees, addresses, mempool | B | Medium |
| Blockchair | multi-chain transaction/account analytics; free testing allowance | B/C | Medium |
| Bitcoin Core / self-hosted node | raw blocks/tx/UTXO/mempool/ZMQ events | B | Lowest once captured |
| mempool.js / open-source wrappers | programmatic access to mempool/Lightning/mining state | B | Low/Medium |

Bitview is especially broad: its hosted instance exposes thousands of Bitcoin metrics
without signup, while a local Bitcoin Core node can reproduce the calculations.
mempool.space provides REST/WebSocket access to mempool, fee and network state.

## 4. News / narrative / external events

| Source | Data | Value | PIT |
|---|---|---:|---:|
| GDELT (existing project source) | global news/events/tone/mentions | A/B | High |
| cryptocurrency.cv | 662k historical crypto articles + entities/sentiment + BTC/ETH context | A | High |
| CryptoPanic-derived archives | crypto news headlines/events | A/B | High |
| SEC EDGAR | ETF/company filings, 8-K/10-K/XBRL | B/C | Low/Medium |
| Central-bank / regulator calendars | policy event timestamps | B | Low |
| FinanceCalendar | free economic-calendar JSON | B | Medium |
| XOOMAR calendar | free macro release schedule JSON | B | Medium |

The cryptocurrency.cv archive is notable for scale and includes article timestamps,
tickers/entities, sentiment, and BTC/ETH market context. Because archive
availability and enrichment timing must be proven retrospectively, use it as
research evidence until PIT capture is validated.

## 5. Macro / rates / risk appetite

| Source | Data | Value | PIT |
|---|---|---:|---:|
| FRED / ALFRED | rates, liquidity, dollar, macro releases + vintage periods | B | Low with vintage |
| Cboe VIX history | daily volatility regime | B | Low |
| CFTC COT | weekly futures positioning | B | Low/Medium |
| U.S. Treasury public data | yields/auction schedule | B | Low/Medium |
| Official Fed/ECB/BoJ calendars | meeting/speech/release timing | B | Low |

FRED/ALFRED is particularly useful for zero-leakage macro features because
the real-time period can be used to reconstruct what was known at a historical
date. The public FRED web-services API itself requires an API key, so automated
use must respect the project's free-first policy and should not assume keyless
access.

## 6. Cross-asset / crypto breadth

| Source | Data | Value | PIT |
|---|---|---:|---:|
| CoinLore | 14k+ assets, 300+ exchanges, market cap, volume, market lists, social stats | B/C | Medium |
| CoinGecko Demo | broad asset market data and metadata | B/C | Medium |
| DeFiLlama | stablecoins, TVL, perps, flows, token unlocks, ETF/DAT dashboards | B | Medium/High |
| Glassnode | very broad on-chain/market metric catalog | B | Low with PIT plan, otherwise High |
| CryptoQuant | broad cross-asset / on-chain catalog | B | Medium |

## 7. Search / crowd / sentiment

| Source | Data | Value | PIT |
|---|---|---:|---:|
| Google Trends | BTC search-demand intensity | B/C | High |
| Reddit | post/comment volume and topic shifts | B/C | High |
| Nostr public relays | real-time Bitcoin-native social events | B/C | High |
| Fear & Greed-style open implementations | composite sentiment state | C | Medium/High |

## 8. Free/open-source implementation references

- bitcoinresearchkit/brk — large local Bitcoin analytics stack.
- mempool/mempool — open-source mempool explorer/API.
- SpiralDevelopment/crypto-hft-data — multi-exchange WebSocket collector.
- emschutt/crypto-market-data-research-engine — event-level Binance ingestion + L5 reconstruction.
- KhavrTrading/flowex — multi-exchange streaming and 75 depth metrics.
- juitindev/crypto-market-data-pipeline — typed Binance event ingestion with kline/orderbook/trade/liquidation schemas.
- joaquinbejar/market2nats — normalized exchange event envelopes.
- tardis-dev/tardis-machine — local replay/caching architecture reference; historical-data availability/licensing still needs verification for free use.

## Candidate priority for this project

For the existing 5m/10m BTC system, the next research frontier is not
"more indicators" in isolation. The highest-information additions are:

1. **Cross-exchange microstructure**
   Binance + Bybit + OKX + Coinbase event alignment:
   spread, depth imbalance, trade imbalance, price dislocation,
   cross-venue lead/lag, liquidity withdrawal, and synchronization quality.

2. **Liquidation-event layer**
   Force-order events from multiple derivatives venues, clustered into
   10s/30s/60s windows with long/short notional, burst intensity and
   post-event recovery/failure signatures.

3. **Options state layer**
   Deribit BTC option IV/surface/skew/OI changes as a state variable rather
   than a single indicator.

4. **On-chain regime layer**
   Bitview/BRK + mempool.space for slower-moving state:
   fee pressure, transaction activity, holder/UTXO structure, miner/network
   conditions and mempool shocks.

5. **External-event layer**
   GDELT + cryptocurrency.cv + official SEC/regulator calendars, but only
   after exact publication/availability timestamps are captured.

6. **Macro timing layer**
   FRED/ALFRED + official central-bank calendars for "event proximity"
   and state transitions, not as high-frequency direct predictors.

Each source should first enter:
DISCOVERY -> ACQUISITION -> PIT VALIDATION -> SHADOW -> CHRONOLOGICAL OOS
before any production feature adoption.
