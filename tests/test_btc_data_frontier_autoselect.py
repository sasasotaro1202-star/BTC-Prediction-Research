from pathlib import Path
from unittest import TestCase, main
from unittest.mock import patch
import tempfile
from src import btc_data_frontier_autoselect as mod


class TestBTCDataFrontierAutoSelect(TestCase):
    def test_gap_is_fail_closed_without_audit(self):
        with tempfile.TemporaryDirectory() as td:
            with patch.object(mod,"ROOT",Path(td)):
                self.assertEqual(mod.current_gap()["gap"],300)
                self.assertFalse(mod.current_gap()["pit_verified"])

    def test_selection_prefers_free_sources_and_diversifies(self):
        frontier={"source_state":{},"candidates":{},"history":[]}
        selected=mod.select_sources(frontier,{"strict_primary":141,"target":300,"gap":159,"pit_verified":False},{})
        self.assertTrue(selected)
        families={s.family for s in mod.SOURCES if s.source_id in selected}
        self.assertGreaterEqual(len(families),4)
        self.assertTrue(all(s.access in {"public_free","free_limited"} for s in mod.SOURCES if s.source_id in selected))

    def test_probe_failure_never_becomes_success(self):
        with patch.object(mod,"_get",side_effect=RuntimeError("offline")):
            result=mod.probe("mempool_space")
        self.assertEqual(result["status"],"ERROR")
        self.assertNotIn("payload",result)

    def test_probe_snapshot_uses_retrieval_basis_when_source_time_unknown(self):
        with patch.object(mod,"_get",return_value={"foo":"bar"}):
            result=mod.probe("mempool_space")
        self.assertEqual(result["status"],"OK")
        self.assertEqual(result["temporal_basis"],"retrieval_snapshot")
        self.assertEqual(result["available_at"],result["retrieved_at"])

    def test_discovery_candidate_is_never_production_eligible(self):
        with patch.object(mod,"_get",return_value={"items":[{"full_name":"example/btc-data","html_url":"https://github.com/example/btc-data"}]}):
            rows, failures=mod.discover_public_sources()
        self.assertEqual(rows[0]["pit_status"],"UNVERIFIED")
        self.assertFalse(rows[0]["production_eligible"])
        self.assertEqual(failures,[])

    def test_current_gap_keeps_collection_active_for_secondary_coverage(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            hist=root/"data/historical_research"
            hist.mkdir(parents=True,exist_ok=True)
            (hist/"pit_oos_audit.json").write_text(
                '{"verified_primary_predictions":300,"min_strict_pit_rows":300,"pit_verified":false,"coverage":{"5m":{"situation_meta_ready":136,"online_expert_ready":136},"10m":{"situation_meta_ready":140,"online_expert_ready":140}}}',
                encoding="utf-8",
            )
            with patch.object(mod,"ROOT",root):
                gap=mod.current_gap()
            self.assertEqual(gap["gap"],0)
            self.assertEqual(gap["situation_meta_ready_min"],136)
            self.assertEqual(gap["online_expert_ready_min"],136)

    def test_discovery_errors_are_recorded(self):
        with patch.object(mod,"_get",side_effect=RuntimeError("offline")):
            rows, failures=mod.discover_public_sources()
        self.assertEqual(rows,[])
        self.assertTrue(failures)

    def test_select_sources_can_select_unverified_discovered_candidates(self):
        frontier={"source_state":{},"candidates":{
            "github:test/btc":{"candidate_id":"github:test/btc","name":"bitcoin dataset","description":"historical API timestamp","query":"bitcoin dataset","license":"MIT","status":"DISCOVERED_UNVERIFIED","production_eligible":False}
        }}
        selected=mod.select_sources(frontier,{"strict_primary":141,"target":300,"gap":159,"pit_verified":False},{})
        self.assertIn("github:test/btc",selected)

    def test_workflow_continuously_recovers_missing_data(self):
        workflow=Path(".github/workflows/btc_autonomous_data_frontier.yml").read_text(encoding="utf-8")
        self.assertIn('cron: "*/15 * * * *"',workflow)
        self.assertIn("actions: write",workflow)
        self.assertIn("dispatch_verified()",workflow)
        self.assertIn("dispatch verification failed",workflow)
        self.assertIn("btc_live_cycle.yml 300",workflow)
        self.assertIn("btc_binance_ws_collector.yml 900",workflow)

    def test_workflow_uses_run_state_and_avoids_snapshot_commit_churn(self):
        workflow=Path(".github/workflows/btc_autonomous_data_frontier.yml").read_text(encoding="utf-8")
        self.assertIn("data/historical_research/data_frontier_run.json",workflow)
        self.assertIn("git add data/historical_research/data_frontier.json",workflow)
        self.assertNotIn("git add data/historical_research/data_frontier.json data/historical_research/source_snapshots/",workflow)

    def test_github_discovery_uses_auth_token(self):
        seen={}
        def fake_get(url,method="GET",body=None,token=None):
            if "api.github.com/search/repositories" in url:
                seen["github_token"]=token
            return {"items":[]}
        with patch.object(mod,"_get",side_effect=fake_get):
            with patch.dict(__import__("os").environ,{"GITHUB_TOKEN":"test-token"}):
                rows, failures=mod.discover_public_sources()
        self.assertEqual(rows,[])
        self.assertEqual(seen["github_token"],"test-token")
        self.assertEqual(failures,[])

    def test_selector_state_is_persistent_across_cycles(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            hist=root/"data/historical_research"
            hist.mkdir(parents=True,exist_ok=True)
            (hist/"data_frontier.json").write_text(
                '{"schema_version":1,"candidates":{}}',encoding="utf-8"
            )
            (hist/"data_frontier_state.json").write_text(
                '{"schema_version":1,"source_state":{"mempool_space":{"successful_probes":7,"consecutive_failures":0}},"history":[{"cycle":12}]}',
                encoding="utf-8",
            )
            with patch.object(mod,"ROOT",root), patch.object(mod,"OUT",hist/"data_frontier.json"), patch.object(mod,"STATE_OUT",hist/"data_frontier_state.json"):
                frontier=mod.load_frontier()
            self.assertEqual(frontier["source_state"]["mempool_space"]["successful_probes"],7)
            self.assertEqual(frontier["history"][-1]["cycle"],12)

    def test_workflow_persists_selector_state_on_dedicated_branch(self):
        workflow=Path(".github/workflows/btc_autonomous_data_frontier.yml").read_text(encoding="utf-8")
        self.assertIn("btc-data-frontier-state",workflow)
        self.assertIn("Persist frontier selector state",workflow)
        self.assertIn("data/historical_research/data_frontier_state.json",workflow)
        self.assertIn("refusing to discard selection memory",workflow)

    def test_empty_durable_frontier_recovers_as_fresh_state(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            hist=root/"data/historical_research"
            hist.mkdir(parents=True,exist_ok=True)
            out=hist/"data_frontier.json"
            out.write_text("",encoding="utf-8")
            with patch.object(mod,"ROOT",root), patch.object(mod,"OUT",out), patch.object(mod,"STATE_OUT",hist/"data_frontier_state.json"):
                frontier=mod.load_frontier()
            self.assertEqual(frontier["candidates"],{})
            self.assertIn("empty_durable_frontier_reset",frontier["_recovery_events"])

    def test_selector_always_reserves_one_qualified_discovery_slot(self):
        frontier={"source_state":{},"candidates":{
            "github:test/btc":{"candidate_id":"github:test/btc","name":"bitcoin historical dataset","description":"bitcoin API timestamps historical csv","query":"bitcoin dataset","license":"MIT","status":"DISCOVERED_UNVERIFIED","production_eligible":False}
        }}
        selected=mod.select_sources(frontier,{"strict_primary":0,"target":300,"gap":300,"pit_verified":False},{})
        self.assertIn("github:test/btc",selected)
        self.assertEqual(sum(1 for sid in selected if sid=="github:test/btc"),1)

    def test_workflow_recovers_missing_historical_and_pit_data(self):
        workflow=Path(".github/workflows/btc_autonomous_data_frontier.yml").read_text(encoding="utf-8")
        self.assertIn("acquire_historical_archive",workflow)
        self.assertIn("refresh_pit_audit",workflow)
        self.assertIn("dispatch_verified btc_archive_refresh.yml 21600",workflow)
        self.assertIn("dispatch_verified btc_pit_oos_audit.yml 3600",workflow)

    def test_workflow_does_not_materialize_empty_selector_state_on_missing_file(self):
        workflow=Path(".github/workflows/btc_autonomous_data_frontier.yml").read_text(encoding="utf-8")
        self.assertIn("state_tmp=",workflow)
        self.assertIn('mv "$state_tmp"',workflow)
        self.assertNotIn("git show origin/btc-data-frontier-state:data/historical_research/data_frontier_state.json > data/historical_research/data_frontier_state.json",workflow)

    def test_run_records_selected_discovered_candidates(self):
        frontier={"source_state":{},"candidates":{
            "github:test/btc":{"candidate_id":"github:test/btc","name":"bitcoin historical dataset","description":"bitcoin API timestamps historical csv","query":"bitcoin dataset","license":"MIT","status":"DISCOVERED_UNVERIFIED","production_eligible":False}
        }}
        selected=mod.select_sources(frontier,{"strict_primary":0,"target":300,"gap":300,"pit_verified":False},{})
        self.assertIn("github:test/btc",selected)
        frontier["candidates"]["github:test/btc"]["last_selected_at"]="2026-10-01T00:00:00+00:00"
        frontier["candidates"]["github:test/btc"]["selection_count"]=1
        self.assertEqual(frontier["candidates"]["github:test/btc"]["selection_count"],1)


    def test_accumulation_target_stays_above_promotion_floor(self):
        secondary, repeat, action = mod.plan_for_gap({
            "strict_primary":300,
            "target":300,
            "situation_meta_ready_min":3000,
            "situation_meta_target":3000,
            "online_expert_ready_min":140,
            "online_expert_target":140,
        })
        self.assertEqual(secondary["strict_primary_gate"],0)
        self.assertEqual(secondary["strict_primary_accumulation"],300)
        self.assertTrue(repeat)
        self.assertEqual(action,"collect_live_and_refresh_pit_for_evidence_margin")

    def test_gate_shortfall_prioritizes_live_acquisition(self):
        _, repeat, action = mod.plan_for_gap({
            "strict_primary":141,
            "target":300,
            "situation_meta_ready_min":3000,
            "situation_meta_target":3000,
            "online_expert_ready_min":140,
            "online_expert_target":140,
        })
        self.assertTrue(repeat)
        self.assertEqual(action,"collect_live_and_refresh_pit")

    def test_selection_count_is_persistent_state_signal(self):
        state={"selection_count":5,"successful_probes":0,"consecutive_failures":0}
        source=next(src for src in mod.SOURCES if src.source_id=="mempool_space")
        self.assertGreaterEqual(mod.score(source,state,{"gap":159},{}),0)

    def test_workflow_persists_state_even_when_recovery_step_fails(self):
        workflow=Path(".github/workflows/btc_autonomous_data_frontier.yml").read_text(encoding="utf-8")
        start=workflow.index("      - name: Persist frontier selector state")
        end=workflow.index("      - name: Persist newly discovered frontier candidates")
        self.assertIn("if: always()",workflow[start:end])
        self.assertIn("if: always()",workflow[end:])

    def test_workflow_exposes_accumulation_loop(self):
        source=Path("src/btc_data_frontier_autoselect.py").read_text(encoding="utf-8")
        self.assertIn("STRICT_PRIMARY_ACCUMULATION_TARGET=600",source)
        self.assertIn("strict_primary_accumulation",source)
        self.assertIn("collect_live_and_refresh_pit_for_evidence_margin",source)

    def test_workflow_allows_evidence_margin_collection_action(self):
        workflow=Path(".github/workflows/btc_autonomous_data_frontier.yml").read_text(encoding="utf-8")
        self.assertIn('"collect_live_and_refresh_pit_for_evidence_margin"',workflow)

    def test_workflow_bootstraps_missing_selector_state_branch(self):
        workflow=Path(".github/workflows/btc_autonomous_data_frontier.yml").read_text(encoding="utf-8")
        section_start=workflow.index("      - name: Persist frontier selector state")
        section_end=workflow.index("      - name: Persist newly discovered frontier candidates")
        section=workflow[section_start:section_end]
        self.assertIn('state_ref="${GITHUB_SHA}"',section)
        self.assertIn("Selector-state branch absent; bootstrapping it",section)
        self.assertIn("git worktree add --detach",section)


    def test_atomic_json_write_round_trips_valid_json(self):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/"state.json"
            payload={"schema_version":1,"ok":True}
            mod._atomic_write_json(path,payload)
            loaded=__import__("json").loads(path.read_text(encoding="utf-8"))
            self.assertEqual(loaded,payload)

if __name__=="__main__":
    main()

    def test_bitget_history_acquisition_is_research_only_and_posthoc_pit_unverified(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            with patch.object(mod,"ROOT",root), patch.object(mod,"ACQUISITION_DIR",root/"data/historical_research/frontier_acquisitions"), patch.object(
                mod,"_get",return_value={"code":"00000","data":[["1700000000000","100","101","99","100.5","12","1206"]]}
            ):
                result=mod.acquire_bitget_history()
            self.assertEqual(result["status"],"OK")
            self.assertFalse(result["production_eligible"])
            self.assertEqual(result["pit_status"],"UNVERIFIED_POSTHOC")
            saved=list((root/"data/historical_research/frontier_acquisitions").glob("*.json"))
            self.assertEqual(len(saved),1)
            obj=__import__("json").loads(saved[0].read_text(encoding="utf-8"))
            self.assertEqual(obj["temporal_basis"],"posthoc_historical_endpoint")
            self.assertEqual(obj["record_count"],1)

    def test_acquisition_is_forced_when_strict_primary_gap_remains(self):
        frontier={"source_state":{},"candidates":{},"history":[]}
        with patch.object(mod,"_acquisition_due",return_value=False):
            result=mod.acquire_selected_research_data(frontier,{"gap":1},["mempool_space"])
        self.assertEqual(result[0]["source_id"],"bitget_public_ws")
        self.assertEqual(result[0]["status"],"SKIPPED_COOLDOWN")

    def test_candidate_reselection_accepts_acquired_research_only_candidates(self):
        frontier={"source_state":{},"candidates":{
            "github:test/btc":{"candidate_id":"github:test/btc","name":"bitcoin historical dataset","description":"timestamp historical csv","query":"bitcoin dataset","license":"MIT","status":"ACQUIRED_RESEARCH_ONLY","production_eligible":False}
        }}
        selected=mod.select_sources(frontier,{"strict_primary":0,"target":300,"gap":300,"pit_verified":False},{})
        self.assertIn("github:test/btc",selected)

    def test_historical_acquisition_cursor_moves_backward(self):
        captured={}
        def fake_get(url,method="GET",body=None,token=None):
            captured["url"]=url
            return {"code":"00000","data":[["1699990000000","100","101","99","100.5","12","1206"]]}
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            with patch.object(mod,"ROOT",root), patch.object(mod,"ACQUISITION_DIR",root/"data/historical_research/frontier_acquisitions"), patch.object(mod,"_get",side_effect=fake_get):
                result=mod.acquire_bitget_history(end_ms=1700000000000)
            self.assertEqual(result["status"],"OK")
            self.assertIn("endTime=1700000000000",captured["url"])
            self.assertEqual(result["next_cursor_ms"],1699990000000)
