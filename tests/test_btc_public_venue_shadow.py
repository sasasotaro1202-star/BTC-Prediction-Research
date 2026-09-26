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
