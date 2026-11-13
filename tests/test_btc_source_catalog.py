from src.btc_source_catalog import (
    CATALOG,
    IMPLEMENTATION_REFERENCES,
    free_research_sources,
    get_source,
    production_blocked_sources,
    with_pit_level,
)


def test_catalog_has_unique_ids_and_required_references():
    ids = [candidate.source_id for candidate in CATALOG]
    assert len(ids) == len(set(ids))
    assert "bitcoinresearchkit/brk" in IMPLEMENTATION_REFERENCES
    assert "jose-donato/crypto-orderbook" in IMPLEMENTATION_REFERENCES
    assert "bokiko/btc-liquidations" in IMPLEMENTATION_REFERENCES


def test_free_catalog_excludes_unconfirmed_and_paid_access():
    free = free_research_sources()
    assert free
    assert all(candidate.access_tier in {"free_public", "free_limited"} for candidate in free)
    assert "coin_glass_api" not in {candidate.source_id for candidate in free}
    assert "fred_alfred" not in {candidate.source_id for candidate in free}


def test_known_free_sources_have_no_credential_requirement():
    assert get_source("kraken_futures_ws").key_required is False
    assert get_source("binance_options").key_required is False
    assert get_source("mempool_space").key_required is False


def test_high_pit_or_credential_sources_remain_blocked():
    blocked = {candidate.source_id for candidate in production_blocked_sources()}
    assert "gdelt" in blocked
    assert "cryptocurrency_cv" in blocked
    assert "fred_alfred" in blocked
    assert "cryptoquant_basic" in blocked
    assert "coin_glass_api" in blocked


def test_pit_updates_are_immutable():
    source = get_source("farside_btc_etf")
    updated = with_pit_level("farside_btc_etf", "high")
    assert source.pit_level == "medium"
    assert updated.pit_level == "high"
    assert updated.source_id == source.source_id
