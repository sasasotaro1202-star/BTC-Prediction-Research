from pathlib import Path
from unittest.mock import patch
import tempfile
from src import btc_data_frontier_autoselect as mod

def test_gap_is_fail_closed_without_audit():
    with tempfile.TemporaryDirectory() as td:
        with patch.object(mod,"ROOT",Path(td)):
            assert mod.current_gap()["gap"]==300
            assert mod.current_gap()["pit_verified"] is False

def test_selection_prefers_free_sources_and_diversifies():
    frontier={"source_state":{},"candidates":{},"history":[]}
    selected=mod.select_sources(frontier,{"strict_primary":141,"target":300,"gap":159,"pit_verified":False},{})
    assert selected
    families={s.family for s in mod.SOURCES if s.source_id in selected}
    assert len(families)>=4
    assert all(s.access in {"public_free","free_limited"} for s in mod.SOURCES if s.source_id in selected)

def test_probe_failure_never_becomes_success():
    with patch.object(mod,"_get",side_effect=RuntimeError("offline")):
        result=mod.probe("mempool_space")
    assert result["status"]=="ERROR"
    assert "payload" not in result

def test_probe_snapshot_uses_retrieval_basis_when_source_time_unknown():
    with patch.object(mod,"_get",return_value={"foo":"bar"}):
        result=mod.probe("mempool_space")
    assert result["status"]=="OK"
    assert result["temporal_basis"]=="retrieval_snapshot"
    assert result["available_at"]==result["retrieved_at"]

def test_discovery_candidate_is_never_production_eligible():
    with patch.object(mod,"_get",return_value={"items":[{"full_name":"example/btc-data","html_url":"https://github.com/example/btc-data"}]}):
        rows=mod.discover_public_sources()
    assert rows[0]["pit_status"]=="UNVERIFIED"
    assert rows[0]["production_eligible"] is False
