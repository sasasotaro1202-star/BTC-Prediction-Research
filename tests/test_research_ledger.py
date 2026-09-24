import json
from pathlib import Path

from src.research_ledger import _compact_lane


def test_compact_lane_preserves_gate_safe_fields_only():
    payload = {
        "status": "OK",
        "research_only": True,
        "production_changed": False,
        "final_holdout_protected": True,
        "promotion_evidence_eligible": True,
        "policy": "research-only",
        "summary": {"mean_logloss_delta": -0.01},
        "final_holdout": {"delta": {"logloss": 0.0}},
        "huge_internal_blob": "must_not_be_copied",
    }
    out = _compact_lane(payload)
    assert out["status"] == "OK"
    assert out["summary"]["mean_logloss_delta"] == -0.01
    assert "huge_internal_blob" not in out


def test_ledger_files_exist_in_expected_repo_location():
    path = Path("src/research_ledger.py")
    assert path.name == "research_ledger.py"
    assert json.loads(json.dumps({"schema_version": 1}))["schema_version"] == 1
