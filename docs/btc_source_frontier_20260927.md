# BTC Source Frontier — 2026-09-27 expansion

Research-only inventory added on a fresh branch from the latest `main`.

## Newly prioritized sources

### 1. Hyperliquid
Public WebSocket feeds expose `l2Book`, `trades`, candles and market mids. The
trade payload includes price, size, side, timestamp and trade id; book snapshots
include bid/ask levels, sizes and timestamps. Reconnect and gap recovery must be
handled explicitly.

Primary documentation:
- https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api/websocket
- https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api/websocket/subscriptions
- https://hyperliquid.gitbook.io/hyperliquid-docs/trading/liquidations

Research hypothesis:
- cross-venue price discovery
- L2 liquidity withdrawal
- trade aggression
- liquidation propagation

### 2. Bitget
Bitget's public API exposes market data without authentication. Public WebSocket
channels include order book/ticker data and a public liquidation stream. The
documented liquidation stream aggregates the prior second and can contain long
and short records.

Primary documentation:
- https://www.bitget.com/docs/classic/rest-api
- https://www.bitget.com/docs/catalog/market/market-data
- https://www.bitget.com/api-doc/uta/websocket/public/Liquidation-Channel

Research hypothesis:
- independent CEX microstructure
- cross-venue liquidation clusters
- book imbalance and liquidity loss

### 3. CME Volume / Open Interest
CME publishes daily volume and open-interest reports free of charge, with a
preliminary report after the trading day and an official Daily Bulletin the
following morning.

Primary documentation:
- https://www.cmegroup.com/market-data/volume-open-interest.html

PIT requirement:
- preliminary availability timestamp and final availability timestamp must be
stored separately.
- never use the final value for a prediction made before the final report.

### 4. Farside BTC ETF flows
Farside publishes issuer-level and total Bitcoin ETF flows. The table is updated
automatically, so represented trading date and actual observation/publication
time must be separated.

Primary source:
- https://farside.co.uk/btc/

Research hypothesis:
- capital-flow regime
- flow shock around market transitions
- slower context for 5m/10m routing rather than assumed direct predictor

### 5. Circle USDC transparency
Circle publishes USDC circulation, issuance/redemption changes and reserve
composition on its transparency page.

Primary source:
- https://www.circle.com/transparency

Research hypothesis:
- stablecoin liquidity regime
- issuance/redemption shocks
- cross-market risk-on/risk-off context

### 6. U.S. Treasury
Official Treasury data provides daily nominal and real Treasury yield curves.

Primary source:
- https://home.treasury.gov/resource-center/data-chart-center/interest-rates

Research hypothesis:
- macro state and event proximity
- slow regime/context feature, not assumed to be a direct 5m signal

## Promotion status

All sources remain **research-only**. Catalog metadata never grants production
eligibility. Required evidence path:

DISCOVERY -> ACQUISITION -> RAW PROVENANCE -> PIT VALIDATION ->
SHADOW -> CHRONOLOGICAL OOS -> ROBUSTNESS/CALIBRATION ->
LIMITED PRODUCTION -> STABLE

No production model, feature schema, calibration artifact, or promotion gate was
changed by this work.
