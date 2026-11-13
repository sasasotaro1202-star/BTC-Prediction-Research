"""Research-only BTC data-source catalog.

This module is intentionally not imported by the production prediction path.
It provides a machine-readable inventory for discovery/acquisition/PIT/OOS
gating of additional information sources.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal


AccessTier = Literal["free_public", "free_limited", "account_required", "paid", "unconfirmed"]
PITLevel = Literal["low", "medium", "high", "unknown"]
UseClass = Literal["direct", "regime", "external_event", "discovery", "implementation_reference"]


@dataclass(frozen=True)
class BTCSourceCandidate:
    source_id: str
    name: str
    category: str
    access_tier: AccessTier
    pit_level: PITLevel
    use_class: UseClass
    realtime: bool
    historical: bool
    key_required: bool
    data: tuple[str, ...]
    notes: str = ""

    def eligible_for_free_research(self) -> bool:
        return self.access_tier in {"free_public", "free_limited"}

    def production_block_reason(self) -> str | None:
        if not self.eligible_for_free_research():
            return "free_access_not_verified"
        if self.pit_level not in {"low", "medium"}:
            return "pit_evidence_incomplete"
        if self.key_required:
            return "credential_dependency"
        return None


CATALOG: tuple[BTCSourceCandidate, ...] = (
    BTCSourceCandidate(
        "binance_usdm_ws",
        "Binance USD-M Futures WS",
        "exchange_microstructure",
        "free_public",
        "low",
        "direct",
        True,
        False,
        False,
        ("aggTrade", "depth", "kline", "mark", "liquidation"),
    ),
    BTCSourceCandidate(
        "bybit_public_ws",
        "Bybit public WS",
        "exchange_microstructure",
        "free_public",
        "low",
        "direct",
        True,
        False,
        False,
        ("orderbook", "trade", "liquidation", "ticker"),
    ),
    BTCSourceCandidate(
        "okx_public_ws",
        "OKX public WS/REST",
        "exchange_microstructure",
        "free_public",
        "low",
        "direct",
        True,
        True,
        False,
        ("orderbook", "trade", "candle", "funding", "open_interest", "mark"),
    ),
    BTCSourceCandidate(
        "coinbase_exchange_ws",
        "Coinbase Exchange WS",
        "exchange_microstructure",
        "free_public",
        "low",
        "direct",
        True,
        False,
        False,
        ("l2", "match", "ticker"),
    ),
    BTCSourceCandidate(
        "kraken_futures_ws",
        "Kraken Futures WS",
        "exchange_microstructure",
        "free_public",
        "low",
        "direct",
        True,
        False,
        False,
        ("book", "trade", "ticker"),
        "Use sequence/checksum where available; preserve venue timestamps.",
    ),
    BTCSourceCandidate(
        "binance_options",
        "Binance Options public market data",
        "options",
        "free_public",
        "medium",
        "direct",
        True,
        True,
        False,
        ("open_interest", "mark_price", "IV", "greeks", "orderbook", "klines"),
        "Timestamp provenance and contract-universe snapshots must be captured.",
    ),
    BTCSourceCandidate(
        "deribit_public",
        "Deribit public API",
        "options",
        "free_public",
        "medium",
        "direct",
        True,
        True,
        False,
        ("options_book", "trades", "open_interest", "IV", "summary"),
    ),
    BTCSourceCandidate(
        "mempool_space",
        "mempool.space",
        "bitcoin_network",
        "free_public",
        "medium",
        "regime",
        True,
        True,
        False,
        ("mempool", "fees", "blocks", "difficulty", "mining", "historical_price"),
        "Rate limits apply; ingestion must cache and fail closed on missing timestamps.",
    ),
    BTCSourceCandidate(
        "brk_bitview",
        "Bitcoin Research Kit / Bitview",
        "bitcoin_onchain",
        "free_public",
        "medium",
        "regime",
        True,
        True,
        False,
        ("UTXO", "holder_cohorts", "fees", "mining", "mempool", "network_metrics"),
        "Best suited to slower state features unless point-in-time capture is proven.",
    ),
    BTCSourceCandidate(
        "farside_btc_etf",
        "Farside Bitcoin ETF flow table",
        "capital_flow",
        "free_public",
        "medium",
        "external_event",
        False,
        True,
        False,
        ("ETF_flow", "issuer_flow", "daily_total"),
        "Publication timing must be separated from the date represented by a row.",
    ),
    BTCSourceCandidate(
        "us_treasury_yields",
        "U.S. Treasury daily yield curve",
        "macro",
        "free_public",
        "low",
        "regime",
        False,
        True,
        False,
        ("nominal_yields", "real_yields"),
        "Use publication/availability time rather than the economic date alone.",
    ),
    BTCSourceCandidate(
        "blockchair",
        "Blockchair",
        "bitcoin_onchain",
        "free_limited",
        "medium",
        "regime",
        False,
        True,
        False,
        ("transactions", "addresses", "UTXO", "blocks"),
        "Free testing allowance is limited; do not assume sustained production quota.",
    ),
    BTCSourceCandidate(
        "coin_metrics_community",
        "Coin Metrics Community Data",
        "market_onchain",
        "free_limited",
        "medium",
        "regime",
        False,
        True,
        False,
        ("market_metrics", "network_metrics", "reference_rates"),
        "Community endpoints have documented rate limits and asset/history limits.",
    ),
    BTCSourceCandidate(
        "cryptoquant_basic",
        "CryptoQuant Basic",
        "derivatives_onchain",
        "free_limited",
        "medium",
        "regime",
        False,
        True,
        True,
        ("price", "OHLCV", "funding", "open_interest", "liquidations"),
        "Free Basic coverage is bounded; credential dependency keeps it research-only.",
    ),
    BTCSourceCandidate(
        "gdelt",
        "GDELT",
        "external_event",
        "free_public",
        "high",
        "external_event",
        True,
        True,
        False,
        ("news", "events", "tone", "mentions"),
        "Do not treat missing archive rows as zero information.",
    ),
    BTCSourceCandidate(
        "cryptocurrency_cv",
        "cryptocurrency.cv",
        "crypto_news",
        "free_public",
        "high",
        "external_event",
        False,
        True,
        False,
        ("news_archive", "entities", "sentiment", "BTC_context"),
        "Historical enrichment timing must be proven before PIT use.",
    ),
    BTCSourceCandidate(
        "fred_alfred",
        "FRED / ALFRED",
        "macro",
        "account_required",
        "low",
        "regime",
        False,
        True,
        True,
        ("rates", "macro_releases", "vintages"),
        "Vintage data is valuable for PIT, but API-key/account dependency remains.",
    ),
    BTCSourceCandidate(
        "coin_glass_api",
        "CoinGlass API",
        "aggregated_derivatives",
        "unconfirmed",
        "high",
        "discovery",
        True,
        True,
        True,
        ("open_interest", "funding", "liquidations", "options", "ETF", "onchain"),
        "Broad coverage is interesting, but free long-term access is not verified.",
    ),
)


IMPLEMENTATION_REFERENCES: tuple[str, ...] = (
    "bitcoinresearchkit/brk",
    "jose-donato/crypto-orderbook",
    "KhavrTrading/flowex",
    "bokiko/btc-liquidations",
    "juitindev/crypto-market-data-pipeline",
    "emschutt/crypto-market-data-research-engine",
    "SpiralDevelopment/crypto-hft-data",
    "joaquinbejar/market2nats",
)


def get_source(source_id: str) -> BTCSourceCandidate:
    for candidate in CATALOG:
        if candidate.source_id == source_id:
            return candidate
    raise KeyError(source_id)


def free_research_sources() -> tuple[BTCSourceCandidate, ...]:
    return tuple(candidate for candidate in CATALOG if candidate.eligible_for_free_research())


def production_blocked_sources() -> tuple[BTCSourceCandidate, ...]:
    return tuple(
        candidate
        for candidate in CATALOG
        if candidate.production_block_reason() is not None
    )


def with_pit_level(source_id: str, pit_level: PITLevel) -> BTCSourceCandidate:
    return replace(get_source(source_id), pit_level=pit_level)
