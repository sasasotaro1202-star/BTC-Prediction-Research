from src.btc_public_venue_shadow import (
    normalize_event, parse_bitget_message, parse_hyperliquid_message, validate_event_set,
)

def test_normalize_event_is_fail_closed():
    assert normalize_event("x", "y", "z", 101, 100, {}) is None
    event = normalize_event("x", "y", "z", 100, 101, {"p": 1})
    assert event["available_at_ms"] == 101
    assert event["event_time_ms"] == 100

def test_hyperliquid_l2_and_trade():
    book = {"channel": "l2Book", "data": {"coin": "BTC", "time": 1000,
        "levels": [[{"px": "100", "sz": "2", "n": 1}], [{"px": "101", "sz": "3", "n": 1}]]}}
    trade = {"channel": "trades", "data": [
        {"coin": "BTC", "side": "B", "px": "100.5", "sz": "1.2", "time": 1001}]}
    events = parse_hyperliquid_message(book, 1100) + parse_hyperliquid_message(trade, 1100)
    assert {event["event_type"] for event in events} == {"l2_book", "trade"}

def test_hyperliquid_future_rejected():
    msg = {"channel": "trades", "data": [
        {"coin": "BTC", "side": "B", "px": "100.5", "sz": "1.2", "time": 1200}]}
    assert parse_hyperliquid_message(msg, 1100) == []

def test_bitget_topics():
    common = {"arg": {"instType": "usdt-futures", "topic": "books5", "symbol": "BTCUSDT"},
              "data": [{"symbol": "BTCUSDT", "ts": "1000", "bids": [["100", "2"]], "asks": [["101", "3"]]}]}
    trade = {"arg": {"instType": "usdt-futures", "topic": "publicTrade", "symbol": "BTCUSDT"},
             "data": [{"symbol": "BTCUSDT", "price": "100", "size": "1", "side": "buy", "ts": "1001"}]}
    liq = {"arg": {"instType": "usdt-futures", "topic": "liquidation"},
           "data": [{"symbol": "BTCUSDT", "price": "99", "amount": "5", "side": "sell", "ts": "1002"}]}
    events = parse_bitget_message(common, 1100) + parse_bitget_message(trade, 1100) + parse_bitget_message(liq, 1100)
    assert {event["event_type"] for event in events} == {"l2_book", "trade", "liquidation"}

def test_validation_deduplicates_and_rejects_future():
    event = normalize_event("x", "y", "z", 100, 110, {"p": 1})
    future = dict(event, event_id="future", event_time_ms=120)
    result = validate_event_set([event, event, future], cutoff_ms=115)
    assert result["accepted"] == 1
    assert result["duplicates"] == 1
    assert result["future"] == 1

def test_situation_card_respects_prediction_cutoff():
    from src.btc_public_venue_shadow import derive_situation_card
    hl_book = normalize_event("hyperliquid", "hyperliquid_perp", "l2_book", 1000, 1001, {
        "coin": "BTC", "levels": [
            [{"px": "100", "sz": "2", "n": 1}],
            [{"px": "101", "sz": "1", "n": 1}],
        ]
    })
    bg_trade = normalize_event("bitget", "bitget_usdt_futures", "trade", 1010, 1011, {
        "symbol": "BTCUSDT", "side": "buy", "price": "100.5", "size": "3", "ts": 1010
    })
    future_trade = normalize_event("bitget", "bitget_usdt_futures", "trade", 2000, 2001, {
        "symbol": "BTCUSDT", "side": "sell", "price": "100.5", "size": "9", "ts": 2000
    })
    assert hl_book and bg_trade and future_trade
    card = derive_situation_card([hl_book, bg_trade, future_trade], cutoff_ms=1500, window_ms=1000)
    assert card["event_count"] == 2
    assert card["trade_imbalance"]["bitget"] == 1.0
    assert future_trade["event_id"] not in card["used_event_ids"]


def test_validation_reports_provenance_quality_metrics():
    event = normalize_event("x", "venue-a", "trade", 100, 110, {"p": 1})
    result = validate_event_set([event], cutoff_ms=120)
    assert result["source_counts"] == {"x": 1}
    assert result["event_type_counts"] == {"trade": 1}
    assert result["availability_lag_ms_max"] == 10
    assert result["availability_lag_ms_p95"] == 10
    assert result["event_time_span_ms"] == 0
