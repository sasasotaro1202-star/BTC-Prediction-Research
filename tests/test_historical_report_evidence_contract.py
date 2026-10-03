from pathlib import Path


def test_historical_report_is_research_only_and_promotion_ineligible():
    source = Path("src/historical_research.py").read_text(encoding="utf-8")
    assert '"research_only":True' in source
    assert '"production_changed":False' in source
    assert '"promotion_evidence_eligible":False' in source
    assert '"pit_evidence_status":"NON_STRICT_ARCHIVE_TIMING"' in source
