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

# Evidence-status fields are intentionally conservative. "UNVERIFIED" and
# "UNCONFIRMED" are valid registry states and must not be interpreted as proof
# of license, pricing, publication timing, schema stability, or production safety.
EvidenceStatus = Literal["UNVERIFIED", "VERIFIED"]
CostStatus = Literal["UNCONFIRMED", "FREE_VERIFIED", "PAID_VERIFIED"]


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
    upstream_id: str = "unknown"
    independence_group: str = "unknown"
    license_status: EvidenceStatus = "UNVERIFIED"
    cost_status: CostStatus = "UNCONFIRMED"
    coverage: str = "unknown"
    freshness: str = "unknown"
    revision_policy: str = "UNKNOWN"
    schema_status: EvidenceStatus = "UNVERIFIED"
    latency: str = "unknown"
    information_value: str = "research_unknown"


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
        "hyperliquid",
        "hyperliquid",
        "UNVERIFIED",
        "UNCONFIRMED",
        "realtime_derivatives",
        "realtime",
        "UNKNOWN",
        "UNVERIFIED",
        "unknown",
        "research_high",
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
        "bitget",
        "bitget",
        "UNVERIFIED",
        "UNCONFIRMED",
        "realtime_derivatives_liquidations",
        "realtime",
        "UNKNOWN",
        "UNVERIFIED",
        "unknown",
        "research_high",
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
        "binance",
        "binance",
        "UNVERIFIED",
        "UNCONFIRMED",
        "realtime_options_derivatives",
        "realtime",
        "UNKNOWN",
        "UNVERIFIED",
        "unknown",
        "research_high",
    ),
    Source(
        "binance_vision_metrics_archive",
        "Binance Vision USD-M metrics archive",
        "exchange_derivatives_archive",
        "public_free",
        "unknown",
        False,
        True,
        False,
        ("metrics", "sum_open_interest", "sum_open_interest_value"),
        2,
        "Official delayed archive for historical metrics. It shares Binance upstream lineage with the live exchange and is not independent evidence; feature-level publication timing is not PIT-proven.",
        "binance",
        "binance",
        "UNVERIFIED",
        "UNCONFIRMED",
        "historical_derivatives_metrics",
        "delayed_archive",
        "UNKNOWN",
        "UNVERIFIED",
        "unknown",
        "research_medium",
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
        "deribit",
        "deribit",
        "UNVERIFIED",
        "UNCONFIRMED",
        "realtime_options_derivatives",
        "realtime",
        "UNKNOWN",
        "UNVERIFIED",
        "unknown",
        "research_high",
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
        "cme",
        "cme",
        "UNVERIFIED",
        "UNCONFIRMED",
        "institutional_derivatives_positioning",
        "daily_preliminary_final",
        "UNKNOWN",
        "UNVERIFIED",
        "unknown",
        "research_medium",
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
        "farside",
        "farside",
        "UNVERIFIED",
        "UNCONFIRMED",
        "daily_btc_etf_flows",
        "daily",
        "UNKNOWN",
        "UNVERIFIED",
        "unknown",
        "research_medium",
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
        "circle",
        "circle",
        "UNVERIFIED",
        "UNCONFIRMED",
        "stablecoin_liquidity",
        "transparency_periodic",
        "UNKNOWN",
        "UNVERIFIED",
        "unknown",
        "research_medium",
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
        "us_treasury",
        "us_treasury",
        "UNVERIFIED",
        "UNCONFIRMED",
        "daily_macro_rates",
        "daily",
        "UNKNOWN",
        "UNVERIFIED",
        "unknown",
        "research_medium",
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
        "mempool_space",
        "mempool_space",
        "UNVERIFIED",
        "UNCONFIRMED",
        "realtime_onchain_network",
        "realtime",
        "UNKNOWN",
        "UNVERIFIED",
        "unknown",
        "research_medium",
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
        "bitcoin_research_kit",
        "bitcoin_research_kit",
        "UNVERIFIED",
        "UNCONFIRMED",
        "onchain_network_cohort_metrics",
        "realtime",
        "UNKNOWN",
        "UNVERIFIED",
        "unknown",
        "research_medium",
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
        "blockchair",
        "blockchair",
        "UNVERIFIED",
        "UNCONFIRMED",
        "historical_onchain",
        "delayed",
        "UNKNOWN",
        "UNVERIFIED",
        "unknown",
        "research_low",
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
        "coin_metrics",
        "coin_metrics",
        "UNVERIFIED",
        "UNCONFIRMED",
        "market_onchain_reference",
        "delayed",
        "UNKNOWN",
        "UNVERIFIED",
        "unknown",
        "research_low",
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
    """Return sources cleared for production.

    This catalog has no runtime capture/OOS evidence fields, so no source can
    be production-eligible from catalog metadata alone. Actual promotion must
    consume project-generated provenance, PIT, chronological OOS, robustness,
    calibration, and operational evidence.
    """
    return tuple()
