from src.research_ledger import LANES, compact

def test_compact_excludes_internal_blobs():
    out = compact({
        "status": "OK",
        "research_only": True,
        "production_changed": False,
        "final_holdout_protected": True,
        "summary": {"mean_logloss_delta": -0.01},
        "huge_internal_blob": "secret",
    })
    assert out["summary"]["mean_logloss_delta"] == -0.01
    assert "huge_internal_blob" not in out

def test_ledger_defines_situation_meta_lane():
    assert LANES["situation_meta"] == "situation_meta_oos.json"


def test_ledger_defines_online_expert_lane():
    assert LANES["online_expert"] == "online_expert_oos.json"
