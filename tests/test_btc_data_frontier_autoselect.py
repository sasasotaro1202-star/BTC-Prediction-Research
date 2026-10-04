from pathlib import Path
from unittest import TestCase, main
from unittest.mock import patch
import tempfile
from datetime import datetime, timedelta, timezone
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

    def test_deribit_probe_uses_current_public_ticker_endpoint(self):
        self.assertEqual(
            mod.PROBES["deribit_public"][1],
            "https://www.deribit.com/api/v2/public/ticker?instrument_name=BTC-PERPETUAL",
        )

    def test_probe_and_historical_acquisition_failures_are_separate(self):
        source=next(src for src in mod.SOURCES if src.source_id=="deribit_public")
        state={
            "selection_count":0,
            "probe_consecutive_failures":8,
            "historical_acquisition_failures":0,
        }
        score_with_probe_failures=mod.score(source,state,{"gap":100},{})
        state["historical_acquisition_failures"]=3
        score_with_acquisition_failures=mod.score(source,state,{"gap":100},{})
        self.assertLess(score_with_acquisition_failures,score_with_probe_failures)

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

    def test_current_gap_fails_closed_when_pit_audit_is_stale(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            hist=root/"data/historical_research"
            hist.mkdir(parents=True,exist_ok=True)
            old_audit={
                "generated_at_utc":"2026-09-30T00:00:00+00:00",
                "verified_primary_predictions":999,
                "min_strict_pit_rows":300,
                "pit_verified":True,
                "coverage":{
                    "5m":{"situation_meta_ready":3000,"online_expert_ready":140},
                    "10m":{"situation_meta_ready":3000,"online_expert_ready":140},
                },
            }
            (hist/"pit_oos_audit.json").write_text(__import__("json").dumps(old_audit),encoding="utf-8")
            with patch.object(mod,"ROOT",root):
                gap=mod.current_gap()
            self.assertFalse(gap["pit_audit_fresh"])
            self.assertEqual(gap["pit_audit_status"],"STALE")
            self.assertFalse(gap["pit_verified"])
            self.assertEqual(gap["strict_primary"],0)
            self.assertEqual(gap["gap"],300)

    def test_current_gap_accepts_fresh_pit_audit(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            hist=root/"data/historical_research"
            hist.mkdir(parents=True,exist_ok=True)
            now=datetime.now(timezone.utc).replace(microsecond=0)
            audit={
                "generated_at_utc":now.isoformat(),
                "verified_primary_predictions":154,
                "min_strict_pit_rows":300,
                "pit_verified":False,
                "legacy_unverified_count":169,
                "coverage":{
                    "5m":{"situation_meta_ready":148,"online_expert_ready":148},
                    "10m":{"situation_meta_ready":145,"online_expert_ready":145},
                },
            }
            (hist/"pit_oos_audit.json").write_text(__import__("json").dumps(audit),encoding="utf-8")
            with patch.object(mod,"ROOT",root):
                gap=mod.current_gap()
            self.assertTrue(gap["pit_audit_fresh"])
            self.assertEqual(gap["pit_audit_status"],"FRESH")
            self.assertEqual(gap["strict_primary"],154)
            self.assertEqual(gap["gap"],146)

    def test_current_gap_keeps_collection_active_for_secondary_coverage(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            hist=root/"data/historical_research"
            hist.mkdir(parents=True,exist_ok=True)
            audit={
                "generated_at_utc":datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
                "verified_primary_predictions":300,
                "min_strict_pit_rows":300,
                "pit_verified":False,
                "coverage":{
                    "5m":{"situation_meta_ready":136,"online_expert_ready":136},
                    "10m":{"situation_meta_ready":140,"online_expert_ready":140},
                },
            }
            (hist/"pit_oos_audit.json").write_text(__import__("json").dumps(audit),encoding="utf-8")
            with patch.object(mod,"ROOT",root):
                gap=mod.current_gap()
            self.assertEqual(gap["gap"],0)
            self.assertEqual(gap["situation_meta_ready_min"],136)
            self.assertEqual(gap["online_expert_ready_min"],136)

    def test_github_discovery_retries_rate_limit_and_records_success(self):
        from urllib.error import HTTPError
        rate_limited=HTTPError(
            "https://api.github.com/search/code?q=BTCUSDT",
            429,
            "Too Many Requests",
            {"Retry-After":"0"},
            None,
        )
        calls=[]
        def fake_get(url,method="GET",body=None,token=None):
            if "search/code" in url:
                calls.append(token)
                if len(calls)==1:
                    raise rate_limited
            return {"items":[]}
        with patch.object(mod,"_get",side_effect=fake_get), patch.object(mod.time,"sleep") as sleep:
            rows, failures=mod.discover_public_sources()
        self.assertEqual(rows,[])
        self.assertEqual(failures,[])
        self.assertGreaterEqual(len(calls),2)
        sleep.assert_any_call(0.1)

    def test_github_discovery_rate_limit_failure_is_fail_closed(self):
        from urllib.error import HTTPError
        rate_limited=HTTPError(
            "https://api.github.com/search/code?q=BTCUSDT",
            429,
            "Too Many Requests",
            {},
            None,
        )
        with patch.object(mod,"_get",side_effect=rate_limited), patch.object(mod.time,"sleep") as sleep:
            rows, failures=mod.discover_public_sources()
        self.assertEqual(rows,[])
        self.assertTrue(failures)
        self.assertGreaterEqual(sleep.call_count,1)

    def test_discovery_errors_are_recorded(self):
        with patch.object(mod,"_get",side_effect=RuntimeError("offline")):
            rows, failures=mod.discover_public_sources()
        self.assertEqual(rows,[])
        self.assertTrue(failures)

    def test_select_sources_can_select_unverified_discovered_candidates(self):
        frontier={"source_state":{},"candidates":{
            "github:test/btc":{"candidate_id":"github:test/btc","name":"bitcoin dataset","description":"historical API timestamp","query":"bitcoin dataset","license":"MIT","url":"https://github.com/test/btc","status":"DISCOVERED_UNVERIFIED","production_eligible":False}
        }}
        selected=mod.select_sources(frontier,{"strict_primary":141,"target":300,"gap":159,"pit_verified":False},{})
        self.assertIn("github:test/btc",selected)

    def test_discovery_lifecycle_is_fail_closed_for_unknown_cost_and_pit(self):
        candidate={
            "candidate_id":"github:test/unknown",
            "name":"bitcoin historical dataset",
            "url":"https://github.com/test/unknown",
            "description":"timestamped historical CSV",
            "platform":"github",
            "status":"DISCOVERED_UNVERIFIED",
            "production_eligible":False,
        }
        lifecycle=mod._candidate_lifecycle(candidate)
        self.assertEqual(lifecycle["eligibility"],"ELIGIBLE_FOR_RESEARCH_REVIEW")
        self.assertEqual(lifecycle["cost_status"],"UNCONFIRMED")
        self.assertEqual(lifecycle["pit_status"],"UNVERIFIED")
        self.assertTrue(lifecycle["research_selection_eligible"])
        self.assertEqual(lifecycle["acquisition_status"],"BLOCKED_UNTIL_VERIFIED_AND_ADAPTER")

        blocked=dict(candidate,description="FactSet bitcoin historical dataset")
        blocked_lifecycle=mod._candidate_lifecycle(blocked)
        self.assertEqual(blocked_lifecycle["eligibility"],"REJECTED_BLOCKED_PROVIDER")

    def test_discovery_selection_penalizes_repeated_same_candidate(self):
        base={
            "candidate_id":"github:test/a",
            "name":"bitcoin historical dataset A",
            "url":"https://github.com/test/a",
            "description":"bitcoin API historical csv timestamp",
            "query":"bitcoin dataset",
            "license":"MIT",
            "status":"DISCOVERED_UNVERIFIED",
            "production_eligible":False,
        }
        repeated=dict(base,selection_count=8,last_selected_at="2026-09-30T00:00:00+00:00")
        frontier={"source_state":{},"candidates":{
            base["candidate_id"]:base,
            "github:test/b":dict(base,candidate_id="github:test/b",name="bitcoin historical dataset B",url="https://github.com/test/b",selection_count=0,last_selected_at=None),
        }}
        selected=mod.select_sources(frontier,{"strict_primary":0,"target":300,"gap":300,"pit_verified":False},{})
        self.assertIn("github:test/b",selected)

    def test_discovery_debt_is_exposed_until_candidates_are_verified(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            hist=root/"data/historical_research"
            hist.mkdir(parents=True,exist_ok=True)
            (hist/"data_frontier.json").write_text(
                '{"schema_version":1,"candidates":{"github:test/btc":{"candidate_id":"github:test/btc","url":"https://github.com/test/btc","status":"DISCOVERED_UNVERIFIED","production_eligible":false}}}',
                encoding="utf-8",
            )
            with patch.object(mod,"ROOT",root), patch.object(mod,"OUT",hist/"data_frontier.json"):
                self.assertEqual(mod._discovery_debt(),1)

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

    def test_legacy_selector_state_migrates_failure_count_to_cumulative_total(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            hist=root/"data/historical_research"
            hist.mkdir(parents=True,exist_ok=True)
            (hist/"data_frontier.json").write_text(
                '{"schema_version":1,"candidates":{}}',encoding="utf-8"
            )
            (hist/"data_frontier_state.json").write_text(
                '{"schema_version":1,"source_state":{"hyperliquid_ws":{"historical_acquisition_failures":58}},"history":[]}',
                encoding="utf-8",
            )
            with patch.object(mod,"ROOT",root), patch.object(mod,"OUT",hist/"data_frontier.json"), patch.object(mod,"STATE_OUT",hist/"data_frontier_state.json"):
                frontier=mod.load_frontier()
            self.assertEqual(
                frontier["source_state"]["hyperliquid_ws"]["historical_acquisition_failures_total"],
                58,
            )

    def test_historical_failure_circuit_breaker_uses_bounded_exponential_backoff(self):
        now=datetime(2026,10,4,8,0,0,tzinfo=timezone.utc)
        state={"historical_acquisition_failures":4}
        until=mod._set_historical_failure_backoff(state,now=now)
        self.assertEqual(
            until,
            datetime(2026,10,4,8,15,0,tzinfo=timezone.utc).isoformat(),
        )
        state["historical_acquisition_failures"]=12
        until=mod._set_historical_failure_backoff(state,now=now)
        self.assertEqual(
            until,
            datetime(2026,10,4,14,0,0,tzinfo=timezone.utc).isoformat(),
        )

    def test_historical_circuit_breaker_skips_source_without_hiding_last_failure(self):
        frontier={
            "source_state":{
                "hyperliquid_ws":{
                    "historical_acquisition_failures":8,
                    "historical_acquisition_backoff_until":(datetime.now(timezone.utc)+timedelta(minutes=5)).replace(microsecond=0).isoformat(),
                    "last_historical_acquisition_error":"HTTPError:429:Too Many Requests",
                    "last_historical_acquisition_error_at":"2026-10-04T07:20:00+00:00",
                }
            },
            "candidates":{},
            "history":[],
        }
        with patch.object(mod,"_acquisition_due",return_value=True), patch.object(
            mod,"acquire_hyperliquid_history",side_effect=AssertionError("circuit breaker did not skip")
        ):
            out=mod.acquire_selected_research_data(
                frontier,{"gap":1,"strict_primary":299},["hyperliquid_ws"]
            )
        self.assertEqual(out[0]["status"],"SKIPPED_CIRCUIT_BREAKER")
        self.assertEqual(out[0]["consecutive_failures"],8)
        self.assertIn("429",out[0]["last_historical_acquisition_error"])

    def test_integrity_repair_bypasses_historical_circuit_breaker(self):
        frontier={
            "source_state":{
                "hyperliquid_ws":{
                    "historical_acquisition_failures":8,
                    "historical_acquisition_backoff_until":"2026-10-04T09:00:00+00:00",
                    "historical_batches_acquired":24,
                    "historical_total_records_acquired":4688,
                }
            },
            "candidates":{},
            "history":[],
        }
        failed={
            "source_id":"hyperliquid_ws",
            "status":"ERROR",
            "retrieved_at":"2026-10-04T08:10:00+00:00",
            "production_eligible":False,
            "error":"integrity_test_failure",
        }
        with patch.object(mod,"_acquisition_due",return_value=True), patch.object(
            mod,"acquire_hyperliquid_history",return_value=failed
        ):
            out=mod.acquire_selected_research_data(
                frontier,
                {"gap":0,"strict_primary":300,"historical_bounds_invalid":[{"reason":"earliest_after_latest"}]},
                ["hyperliquid_ws"],
            )
        self.assertEqual(out[0]["status"],"ERROR")
        self.assertEqual(frontier["source_state"]["hyperliquid_ws"]["historical_acquisition_failures"],9)

    def test_success_resets_consecutive_failures_but_retains_cumulative_total(self):
        frontier={
            "source_state":{
                "bitget_public_ws":{
                    "historical_acquisition_failures":3,
                    "historical_acquisition_failures_total":9,
                    "historical_batches_acquired":2,
                }
            },
            "candidates":{},
            "history":[],
        }
        success={
            "source_id":"bitget_public_ws",
            "status":"OK",
            "retrieved_at":"2026-10-04T08:10:00+00:00",
            "record_count":200,
            "next_cursor_ms":100,
            "payload_sha256":"abc",
            "first_event_time":"2026-10-04T00:00:00+00:00",
            "last_event_time":"2026-10-04T01:00:00+00:00",
            "production_eligible":False,
        }
        with patch.object(mod,"_acquisition_due",return_value=True), patch.object(
            mod,"acquire_bitget_history",return_value=success
        ):
            out=mod.acquire_selected_research_data(
                frontier,{"gap":1,"strict_primary":299},["bitget_public_ws"]
            )
        self.assertEqual(out[0]["status"],"OK")
        state=frontier["source_state"]["bitget_public_ws"]
        self.assertEqual(state["historical_acquisition_failures"],0)
        self.assertEqual(state["historical_acquisition_failures_total"],9)

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
            "github:test/btc":{"candidate_id":"github:test/btc","name":"bitcoin historical dataset","description":"bitcoin API timestamps historical csv","query":"bitcoin dataset","license":"MIT","url":"https://github.com/test/btc","status":"DISCOVERED_UNVERIFIED","production_eligible":False}
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
            "github:test/btc":{"candidate_id":"github:test/btc","name":"bitcoin historical dataset","description":"bitcoin API timestamps historical csv","query":"bitcoin dataset","license":"MIT","url":"https://github.com/test/btc","status":"DISCOVERED_UNVERIFIED","production_eligible":False}
        }}
        selected=mod.select_sources(frontier,{"strict_primary":0,"target":300,"gap":300,"pit_verified":False},{})
        self.assertIn("github:test/btc",selected)
        frontier["candidates"]["github:test/btc"]["last_selected_at"]="2026-10-01T00:00:00+00:00"
        frontier["candidates"]["github:test/btc"]["selection_count"]=1
        self.assertEqual(frontier["candidates"]["github:test/btc"]["selection_count"],1)


    def test_historical_bounds_invalid_is_exposed_and_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            hist=root/"data/historical_research"
            hist.mkdir(parents=True,exist_ok=True)
            state={
                "schema_version":1,
                "source_state":{
                    "bitget_public_ws":{
                        "historical_earliest_event_time":"2026-09-30T11:50:00+00:00",
                        "last_historical_event_time":"2026-08-16T19:40:00+00:00",
                    }
                }
            }
            (hist/"data_frontier_state.json").write_text(__import__("json").dumps(state),encoding="utf-8")
            with patch.object(mod,"ROOT",root), patch.object(mod,"STATE_OUT",hist/"data_frontier_state.json"):
                issues=mod._historical_bounds_issues_from_state()
                self.assertEqual(issues[0]["source_id"],"bitget_public_ws")
                self.assertEqual(issues[0]["reason"],"earliest_after_latest")
                secondary,repeat,action=mod.plan_for_gap({
                    "strict_primary":300,
                    "target":300,
                    "situation_meta_ready_min":3000,
                    "situation_meta_target":3000,
                    "online_expert_ready_min":140,
                    "online_expert_target":140,
                    "historical_bounds_invalid":issues,
                })
            self.assertTrue(repeat)
            self.assertEqual(secondary["historical_bounds_invalid"],issues)
            self.assertEqual(action,"repair_historical_frontier_bounds")

    def test_historical_bounds_extension_recovers_stale_latest_bound(self):
        state={
            "historical_earliest_event_time":"2026-09-30T11:50:00+00:00",
            "last_historical_event_time":"2026-08-16T19:40:00+00:00",
        }
        mod._extend_historical_time_bounds(state,{
            "first_event_time":"2026-08-16T18:00:00+00:00",
            "last_event_time":"2026-08-16T20:00:00+00:00",
        })
        self.assertEqual(state["historical_earliest_event_time"],"2026-08-16T18:00:00+00:00")
        self.assertEqual(state["last_historical_event_time"],"2026-08-16T20:00:00+00:00")


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