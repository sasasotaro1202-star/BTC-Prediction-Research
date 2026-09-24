import numpy as np
from src.binance_flow_features import derive_flow_features
from src.binance_flow_ws import ingest_event, normalize_bins, parse_agg_trade_message, parse_force_order_message


def test_aggtrade_parser_is_pit_safe():
    msg={"e":"aggTrade","E":1000,"s":"BTCUSDT","a":7,"p":"100","q":"2","T":999,"m":False}
    row=parse_agg_trade_message(msg,1010)
    assert row is not None
    assert row["aggressor_sign"]==1.0
    assert row["notional"]==200.0
    assert parse_agg_trade_message({**msg,"s":"ETHUSDT"},1010) is None
    assert parse_agg_trade_message({**msg,"E":20000},1010) is None


def test_forceorder_parser_is_strict():
    msg={"e":"forceOrder","E":2000,
         "o":{"s":"BTCUSDT","S":"SELL","i":"5","T":1999,"ap":"100","z":"3"}}
    row=parse_force_order_message(msg,2010)
    assert row is not None
    assert row["aggressor_sign"]==-1.0
    assert row["notional"]==300.0
    assert parse_force_order_message({**msg,"o":{**msg["o"],"S":"BAD"}},2010) is None


def test_finalized_bins_are_immutable():
    bins={}
    event={"kind":"agg_trade","event_time_ms":10000,"notional":100.0,"quantity":1.0,"aggressor_sign":1.0}
    assert ingest_event(bins,event)
    bins[10000]["available_at_ms"]=11000
    assert not ingest_event(bins,{**event,"notional":900.0})
    rows=normalize_bins(bins)
    assert len(rows)==1 and rows[0]["buy_notional"]==100.0


def test_features_reject_unavailable_bins():
    rows=[
        {"start_time_ms":0,"end_time_ms":5000,"available_at_ms":5500,"last_event_time_ms":4000,
         "trade_count":2,"buy_qty":2.0,"sell_qty":1.0,"buy_notional":200.0,"sell_notional":100.0,
         "max_trade_notional":150.0,"liquidation_count":0,"liquidation_buy_notional":0.0,"liquidation_sell_notional":0.0},
        {"start_time_ms":5000,"end_time_ms":10000,"available_at_ms":12000,"last_event_time_ms":9000,
         "trade_count":5,"buy_qty":5.0,"sell_qty":1.0,"buy_notional":500.0,"sell_notional":100.0,
         "max_trade_notional":200.0,"liquidation_count":1,"liquidation_buy_notional":50.0,"liquidation_sell_notional":0.0},
    ]
    feat=derive_flow_features(rows,10000)
    assert feat["flow_5s_missing"]==1.0
    assert feat["flow_5s_signed_notional"]==0.0
    assert feat["flow_30s_trade_count"]==2.0


def test_feature_output_is_finite():
    feat=derive_flow_features([],10000)
    assert np.isfinite(np.asarray(list(feat.values()),dtype=float)).all()


def test_cache_roundtrip_is_valid_json(tmp_path):
    import json
    from src.binance_flow_ws import write_cache, load_cache

    bins = {0: {
        "start_time_ms": 0, "end_time_ms": 5000, "available_at_ms": 6000,
        "last_event_time_ms": 4000, "trade_count": 1,
        "buy_qty": 1.0, "sell_qty": 0.0, "buy_notional": 100.0,
        "sell_notional": 0.0, "max_trade_notional": 100.0,
        "liquidation_count": 0, "liquidation_buy_notional": 0.0,
        "liquidation_sell_notional": 0.0,
    }}
    path = tmp_path / "flow.json"
    write_cache(bins, path)
    json.loads(path.read_text(encoding="utf-8"))
    restored = load_cache(path)
    assert 0 in restored
    assert restored[0]["buy_notional"] == 100.0
