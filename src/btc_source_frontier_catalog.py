"""Free-first BTC research source frontier.

Research-only inventory. This module is not part of the production prediction
path and contains no network calls. Each source must pass acquisition,
provenance, PIT, chronological OOS, robustness, and calibration gates before
any feature can be promoted.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Access = Literal["public_free", "free_limited", "account_required", "unconfirmed"]
PIT = Literal["low", "medium", "high", "unknown"]


@dataclass(frozen=True)
class Source:
    source_id: str
    name: str
    family: str
    access: Access
    pit: PIT
    realtime: bool
    historical: bool
    key_required: bool
    payloads: tuple[str, ...]
    priority: int
    rationale: str


SOURCES: tuple[Source, ...] = (
    Source(
        "hyperliquid_ws",
        "Hyperliquid public WebSocket",
        "exchange_derivatives",
        "public_free",
        "low",
        True,
        True,
        False,
        ("l2Book", "trades", "candle", "allMids"),
        1,
        "Transparent DEX-style derivatives flow with explicit exchange timestamps.",
    ),
    Source(
        "bitget_public_ws",
        "Bitget public WebSocket",
        "exchange_derivatives",
        "public_free",
        "low",
        True,
        True,
        False,
        ("orderbook", "ticker", "liquidation"),
        1,
        "Adds an independent CEX venue and aggregated liquidation stream.",
    ),
    Source(
        "binance_options_public",
        "Binance Options public market data",
        "options",
        "public_free",
        "medium",
        True,
        True,
        False,
        ("open_interest", "mark", "IV", "greeks", "orderbook", "klines"),
        1,
        "Extends the existing derivatives layer from futures into the options surface.",
    ),
    Source(
        "deribit_public",
        "Deribit public API",
        "options",
        "public_free",
        "medium",
        True,
        True,
        False,
        ("book", "trades", "open_interest", "IV", "summaries"),
        1,
        "Independent BTC options venue for surface, skew, and OI state.",
    ),
    Source(
        "cme_voi",
        "CME Volume and Open Interest reports",
        "institutional_derivatives",
        "public_free",
        "medium",
        False,
        True,
        False,
        ("volume", "open_interest", "preliminary_daily", "official_daily"),
        2,
        "Institutional BTC futures/options positioning with separately published preliminary/final states.",
    ),
    Source(
        "farside_btc_etf",
        "Farside Bitcoin ETF flows",
        "capital_flow",
        "public_free",
        "medium",
        False,
        True,
        False,
        ("issuer_flow", "total_flow"),
        2,
        "Daily ETF flow state; must model publication availability separately from represented date.",
    ),
    Source(
        "circle_usdc_transparency",
        "Circle USDC transparency",
        "stablecoin_liquidity",
        "public_free",
        "medium",
        False,
        True,
        False,
        ("circulation", "issued", "redeemed", "reserve_composition"),
        2,
        "Stablecoin liquidity context and issuance/redemption regime.",
    ),
    Source(
        "us_treasury_yield_curve",
        "U.S. Treasury daily yield curve",
        "macro",
        "public_free",
        "low",
        False,
        True,
        False,
        ("nominal_yield", "real_yield"),
        2,
        "Rates/risk regime context with official publication artifacts.",
    ),
    Source(
        "mempool_space",
        "mempool.space",
        "bitcoin_network",
        "public_free",
        "medium",
        True,
        True,
        False,
        ("mempool", "fees", "blocks", "transactions", "mining"),
        2,
        "Native BTC network state and congestion shocks.",
    ),
    Source(
        "brk_bitview",
        "Bitcoin Research Kit / Bitview",
        "bitcoin_onchain",
        "public_free",
        "medium",
        True,
        True,
        False,
        ("UTXO", "holder_cohorts", "fees", "mempool", "mining", "network_metrics"),
        2,
        "Broad on-chain state; strongest use is regime/context until point-in-time capture is proven.",
    ),
    Source(
        "blockchair",
        "Blockchair",
        "bitcoin_onchain",
        "free_limited",
        "medium",
        False,
        True,
        False,
        ("transactions", "addresses", "blocks", "UTXO"),
        3,
        "Useful breadth/validation source, but sustained quota must be re-verified before production.",
    ),
    Source(
        "coin_metrics_community",
        "Coin Metrics Community Data",
        "market_onchain",
        "free_limited",
        "medium",
        False,
        True,
        False,
        ("market_metrics", "network_metrics", "reference_rates"),
        3,
        "Independent reference dataset for cross-checking exchange/network measurements.",
    ),
)


def get_source(source_id: str) -> Source:
    for source in SOURCES:
        if source.source_id == source_id:
            return source
    raise KeyError(source_id)


def free_sources() -> tuple[Source, ...]:
    return tuple(source for source in SOURCES if source.access in {"public_free", "free_limited"})


def direct_high_priority_sources() -> tuple[Source, ...]:
    return tuple(
        source
        for source in SOURCES
        if source.priority == 1 and source.realtime and source.pit in {"low", "medium"}
    )


def production_eligible_now() -> tuple[Source, ...]:
    # Intentional conservative gate: "free" alone is insufficient.
    return tuple(
        source
        for source in SOURCES
        if source.access == "public_free"
        and source.pit == "low"
        and source.realtime
        and not source.key_required
    )
