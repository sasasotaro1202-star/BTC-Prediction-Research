from src.btc_source_frontier_catalog import (
    SOURCES,
    direct_high_priority_sources,
    free_sources,
    get_source,
    production_eligible_now,
)


def test_source_ids_are_unique_and_catalog_is_nonempty():
    ids = [source.source_id for source in SOURCES]
    assert ids
    assert len(ids) == len(set(ids))


def test_free_sources_have_no_paid_or_unconfirmed_access():
    assert free_sources()
    assert all(
        source.access in {"public_free", "free_limited"}
        for source in free_sources()
    )


def test_high_priority_direct_sources_are_free_and_pit_bounded():
    sources = direct_high_priority_sources()
    ids = {source.source_id for source in sources}
    assert {"hyperliquid_ws", "bitget_public_ws"} <= ids
    assert all(source.access == "public_free" for source in sources)
    assert all(source.pit in {"low", "medium"} for source in sources)


def test_production_gate_is_fail_closed_without_evidence():
    assert production_eligible_now() == ()


def test_source_lookup():
    source = get_source("circle_usdc_transparency")
    assert source.family == "stablecoin_liquidity"
    assert "circulation" in source.payloads


def test_binance_vision_metrics_archive_is_explicitly_non_pit_and_not_independent():
    source = get_source("binance_vision_metrics_archive")
    assert source.access == "public_free"
    assert source.pit == "unknown"
    assert source.historical is True
    assert source.realtime is False
    assert "not independent evidence" in source.rationale


def test_source_registry_tracks_required_provenance_dimensions():
    for source in SOURCES:
        assert source.upstream_id.strip()
        assert source.independence_group.strip()
        assert source.license_status in {"UNVERIFIED", "VERIFIED"}
        assert source.cost_status in {"UNCONFIRMED", "FREE_VERIFIED", "PAID_VERIFIED"}
        assert source.coverage.strip()
        assert source.freshness.strip()
        assert source.revision_policy.strip()
        assert source.schema_status in {"UNVERIFIED", "VERIFIED"}
        assert source.latency.strip()
        assert source.information_value.strip()


def test_same_upstream_sources_share_independence_group():
    binance_ids = {"binance_options_public", "binance_vision_metrics_archive"}
    rows = [get_source(source_id) for source_id in binance_ids]
    assert {row.upstream_id for row in rows} == {"binance"}
    assert {row.independence_group for row in rows} == {"binance"}
