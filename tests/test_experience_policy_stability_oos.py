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


def test_stability_moves_initial_split_to_settlement_boundary(monkeypatch):
    rows = [_row(i, 1) for i in range(8)]
    shared_ts = rows[1]["settled_at_utc"]
    rows[2]["settled_at_utc"] = shared_ts
    seen = []

    monkeypatch.setattr(mod, "_baseline", lambda train_rows: 0.5)
    monkeypatch.setattr(mod, "_hierarchical_memory_predict", lambda train_rows, test_rows: [0.5] * len(test_rows))
    monkeypatch.setattr(mod, "_metrics", lambda y_true, probabilities: {"n": len(y_true), "logloss": 0.5, "brier": 0.25, "error_rate": 0.5})

    original = mod._hierarchical_memory_predict
    def record(train_rows, test_rows):
        seen.append((len(train_rows), [r["experience_id"] for r in test_rows]))
        return original(train_rows, test_rows)
    monkeypatch.setattr(mod, "_hierarchical_memory_predict", record)

    out = mod.evaluate_rows(rows, min_train_rows=2, test_block_rows=1, min_blocks=1)
    assert out["status"] == "OK"
    assert seen[0] == (3, [4])


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


def test_stability_does_not_split_settlement_timestamp_at_block_boundary(monkeypatch):
    rows = [_row(i, 1) for i in range(8)]
    shared_ts = rows[1]["settled_at_utc"]
    rows[2]["settled_at_utc"] = shared_ts
    seen = []

    monkeypatch.setattr(mod, "_baseline", lambda train_rows: 0.5)
    def memory(train_rows, test_rows):
        seen.append(([r["experience_id"] for r in train_rows], [r["experience_id"] for r in test_rows]))
        return [0.5] * len(test_rows)
    monkeypatch.setattr(mod, "_hierarchical_memory_predict", memory)
    monkeypatch.setattr(mod, "_metrics", lambda y_true, probabilities: {"n": len(y_true), "logloss": 0.5, "brier": 0.25, "error_rate": 0.5})
    out = mod.evaluate_rows(rows, min_train_rows=3, test_block_rows=1, min_blocks=1)
    assert out["status"] == "OK"
    assert any(test_ids == [4] and train_ids[:3] == [1, 2, 3] for train_ids, test_ids in seen)
