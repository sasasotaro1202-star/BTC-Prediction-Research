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
        rows, failures=mod.discover_public_sources()
    assert rows[0]["pit_status"]=="UNVERIFIED"
    assert rows[0]["production_eligible"] is False
    assert not failures

def test_current_gap_keeps_collection_active_for_secondary_coverage():
    with tempfile.TemporaryDirectory() as td:
        root=Path(td)
        (root/"data/historical_research").mkdir(parents=True,exist_ok=True)
        (root/"data/historical_research/pit_oos_audit.json").write_text(
            '{"verified_primary_predictions":300,"min_strict_pit_rows":300,"pit_verified":false,"coverage":{"5m":{"situation_meta_ready":136,"online_expert_ready":136},"10m":{"situation_meta_ready":140,"online_expert_ready":140}}}',
            encoding="utf-8",
        )
        with patch.object(mod,"ROOT",root):
            gap=mod.current_gap()
        assert gap["gap"] == 0
        assert gap["situation_meta_ready_min"] == 136
        assert gap["online_expert_ready_min"] == 136

def test_discovery_errors_are_recorded():
    with patch.object(mod,"_get",side_effect=RuntimeError("offline")):
        rows, failures=mod.discover_public_sources()
    assert rows == []
    assert failures

def test_select_sources_can_select_unverified_discovered_candidates():
    frontier={"source_state":{},"candidates":{
        "github:test/btc":{"candidate_id":"github:test/btc","name":"bitcoin dataset","description":"historical API timestamp","query":"bitcoin dataset","license":"MIT","status":"DISCOVERED_UNVERIFIED","production_eligible":False}
    }}
    selected=mod.select_sources(frontier,{"strict_primary":141,"target":300,"gap":159,"pit_verified":False},{})
    assert "github:test/btc" in selected

def test_workflow_continuously_recovers_missing_data():
    workflow = Path(".github/workflows/btc_autonomous_data_frontier.yml").read_text(encoding="utf-8")
    assert 'cron: "*/15 * * * *"' in workflow
    assert "actions: write" in workflow
    assert "dispatch_verified()" in workflow
    assert "dispatch verification failed" in workflow
    assert "btc_live_cycle.yml 300" in workflow
    assert "btc_binance_ws_collector.yml 900" in workflow

def test_workflow_uses_run_state_and_avoids_snapshot_commit_churn():
    workflow = Path(".github/workflows/btc_autonomous_data_frontier.yml").read_text(encoding="utf-8")
    assert "data/historical_research/data_frontier_run.json" in workflow
    assert "git add data/historical_research/data_frontier.json" in workflow
    assert "git add data/historical_research/data_frontier.json data/historical_research/source_snapshots/" not in workflow
