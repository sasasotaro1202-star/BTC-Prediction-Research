from pathlib import Path


def test_sentinel_recovers_deferred_cycle_when_cache_is_fresh():
    path = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "btc_production_sentinel.yml"
    text = path.read_text(encoding="utf-8")

    assert 'data/live_cycle_status.json?ref=main' in text
    assert 'data/binance_ws_1m.json?ref=binance-ws-cache' in text
    assert 'cache_age_ms=$((now*1000-cache_event_ms))' in text
    assert '[ "$cache_age_ms" -le 180000 ]' in text
    assert 'cycle_mode="$(jq -r ".mode // empty"' in text
    assert 'deferred_cache_recovery=1' in text
    assert '[ "$active" -eq 0 ]' in text
    assert '[ "$deferred_cache_recovery" -eq 1 ]' in text


def test_sentinel_does_not_make_fresh_cache_by_itself_a_recovery_signal():
    path = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "btc_production_sentinel.yml"
    text = path.read_text(encoding="utf-8")
    condition = 'if [ "$active" -eq 0 ] && { [ "$age" -ge 300 ] || [ "$head" != "$main_sha" ] || [ "$deferred_cache_recovery" -eq 1 ]; }; then'
    assert condition in text
    assert 'if [ "$cycle_mode" = "deferred" ] && [ "$cache_fresh" -eq 1 ]; then' in text
