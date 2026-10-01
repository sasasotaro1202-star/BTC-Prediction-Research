from datetime import datetime, timedelta, timezone

from src import experience_policy_stability_oos as mod


def _row(i: int, correct: int):
    ts=datetime(2026,1,1,tzinfo=timezone.utc)+timedelta(minutes=i)
    return {
        "experience_id":i+1,
        "correct":correct,
        "settled_at_utc":ts.isoformat(),
        "regime":"TREND" if i%2 else "RANGE",
        "predicted_direction":"UP" if i%3 else "DOWN",
        "confidence":0.55 if i%2 else 0.45,
        "production_mode":"primary",
    }


def test_evaluate_rows_defers_when_too_few_rows():
    rows=[_row(i,i%2) for i in range(100)]
    out=mod.evaluate_rows(rows,min_train_rows=100,test_block_rows=50,min_blocks=2)
    assert out["status"]=="DEFERRED"


def test_evaluate_rows_has_multiple_prequential_blocks_and_latest_block():
    rows=[_row(i,int(i%3!=0)) for i in range(270)]
    out=mod.evaluate_rows(rows,min_train_rows=100,test_block_rows=50,min_blocks=3)
    assert out["status"]=="OK"
    assert out["block_count"]==4
    assert out["non_worse_block_count"]+sum(not b["non_worse"] for b in out["blocks"])==4
    assert out["latest_block"]["block_id"]==4
    assert all(
        b["test_start_settled_at_utc"] > b["train_end_settled_at_utc"]
        or b["test_start_settled_at_utc"] == b["train_end_settled_at_utc"]
        for b in out["blocks"]
    )


def test_evaluate_rows_is_deterministic():
    rows=[_row(i,int(i%5!=0)) for i in range(260)]
    a=mod.evaluate_rows(rows,min_train_rows=100,test_block_rows=40,min_blocks=3)
    b=mod.evaluate_rows(rows,min_train_rows=100,test_block_rows=40,min_blocks=3)
    assert a==b
