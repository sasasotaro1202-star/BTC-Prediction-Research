from src.btc_decision_adapter_v18 import build_decision_from_situation_card, _event_id, _payload_hash
import pytest

def _events():
    payload_a={"px":"1","sz":"1"}
    sha_a=_payload_hash(payload_a)
    payload_b={"bids":[["1","1"]],"asks":[["2","1"]]}
    sha_b=_payload_hash(payload_b)
    return [
        {"event_id":_event_id("hyperliquid","hyperliquid_perp","trade",900,sha_a),"source":"hyperliquid","venue":"hyperliquid_perp","event_type":"trade","event_time_ms":900,"available_at_ms":950,"retrieved_at_ms":950,"payload_sha256":sha_a,"payload":payload_a},
        {"event_id":_event_id("bitget","bitget_usdt_futures","l2_book",800,sha_b),"source":"bitget","venue":"bitget_usdt_futures","event_type":"l2_book","event_time_ms":800,"available_at_ms":925,"retrieved_at_ms":930,"payload_sha256":sha_b,"payload":payload_b},
    ]

def test_integrity_and_pit_verified():
    events=_events(); card={"prediction_cutoff_ms":1000,"used_event_ids":[e["event_id"] for e in events],"freshness_ms":100}
    obj=build_decision_from_situation_card(cutoff_ms=1000,probability={"up":.4,"flat":.2,"down":.4},situation_card=card,events=events)
    assert obj.pit_status=="verified"; obj.require_verified_pit()

def test_payload_tamper_fails_closed():
    events=_events(); events[0]["payload"]={"px":"9","sz":"1"}
    with pytest.raises(ValueError,match="shadow_payload_hash_mismatch"):
        build_decision_from_situation_card(cutoff_ms=1000,probability={"up":.4,"flat":.2,"down":.4},situation_card={"prediction_cutoff_ms":1000,"used_event_ids":[events[0]["event_id"]]},events=events)

def test_event_id_tamper_fails_closed():
    events=_events(); events[0]["event_id"]="x"*32
    with pytest.raises(ValueError,match="shadow_event_id_mismatch"):
        build_decision_from_situation_card(cutoff_ms=1000,probability={"up":.4,"flat":.2,"down":.4},situation_card={"prediction_cutoff_ms":1000,"used_event_ids":[events[0]["event_id"]]},events=events)

def test_future_availability_fails_closed():
    events=_events(); events[0]["available_at_ms"]=1001
    with pytest.raises(ValueError,match="future_shadow_information"):
        build_decision_from_situation_card(cutoff_ms=1000,probability={"up":.4,"flat":.2,"down":.4},situation_card={"prediction_cutoff_ms":1000,"used_event_ids":[events[0]["event_id"]]},events=events)