from src.research_ledger import compact
def test_compact_is_small_and_does_not_copy_internal_blobs():
    out=compact({"status":"OK","research_only":True,"production_changed":False,"final_holdout_protected":True,
                 "summary":{"mean_logloss_delta":-0.01},"final_holdout":{"n":100},"huge_internal_blob":"secret"})
    assert out["summary"]["mean_logloss_delta"] == -0.01
    assert "huge_internal_blob" not in out
def test_compact_preserves_gate_boundary():
    out=compact({"status":"OK","promotion_evidence_eligible":False,"production_changed":False})
    assert out["production_changed"] is False
    assert out["promotion_evidence_eligible"] is False
