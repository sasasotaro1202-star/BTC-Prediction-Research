import json
import tempfile
import unittest
from pathlib import Path

from src.external_method_research import JsonClient, choose_next_candidate, process_one


class FakeGitHubClient(JsonClient):
    def __init__(self, payloads):
        self.payloads = payloads

    def get(self, url):
        return self.payloads.get(url)


class ExternalMethodResearchTests(unittest.TestCase):
    def _queue(self):
        return {
            "schema_version": 3,
            "candidates": [
                {"queue_rank": 1, "repository": "agent/first", "next_gate": "LOCAL_ONLY"},
                {"queue_rank": 2, "repository": "model/second", "next_gate": "MODEL_OOS_GATE"},
            ],
            "priority_gate": {
                "immediate_local_reproduction": [{"repository": "model/second"}]
            },
        }

    def test_priority_gate_beats_queue_rank(self):
        self.assertEqual(
            choose_next_candidate(self._queue(), {})["repository"],
            "model/second",
        )

    def test_source_verification_becomes_local_gate_ready(self):
        payloads = {
            "https://api.github.com/repos/model/second": {
                "archived": False,
                "default_branch": "main",
                "html_url": "https://github.com/model/second",
                "license": {"spdx_id": "MIT"},
                "stargazers_count": 1,
            },
            "https://api.github.com/repos/model/second/commits/main": {"sha": "abc123"},
        }
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            path = root / "data" / "external_research_method_queue.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(self._queue()), encoding="utf-8")
            result = process_one(root, FakeGitHubClient(payloads), "TEST_SHA", "1")
            self.assertEqual(result["status"], "LOCAL_GATE_READY")
            self.assertFalse(result["production_changed"])
            self.assertFalse(result["promotion_allowed"])
            self.assertFalse(result["external_performance_transfer_allowed"])

    def test_source_contract_failure_is_hold(self):
        queue = self._queue()
        queue["candidates"][1]["source_contracts"] = [
            {"kind": "huggingface_model", "id": "unknown/model"}
        ]
        payloads = {
            "https://api.github.com/repos/model/second": {
                "archived": False,
                "default_branch": "main",
                "license": {"spdx_id": "MIT"},
            },
            "https://api.github.com/repos/model/second/commits/main": {"sha": "abc"},
            "https://huggingface.co/api/models/unknown/model": None,
        }
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            path = root / "data" / "external_research_method_queue.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(queue), encoding="utf-8")
            result = process_one(root, FakeGitHubClient(payloads), "TEST_SHA", "2")
            self.assertEqual(result["status"], "HOLD")
            self.assertTrue(
                any(
                    "MODEL_REPOSITORY_NOT_FOUND" in item
                    for item in result["source_contract_verification"]["failures"]
                )
            )

    def test_missing_repo_is_hold(self):
        queue = self._queue()
        queue["candidates"][1]["repository"] = "missing/repo"
        queue["priority_gate"] = {}
        runtime = {
            "results": {
                "agent/first": {
                    "status": "LOCAL_GATE_READY",
                }
            }
        }
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            path = root / "data" / "external_research_method_queue.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(queue), encoding="utf-8")
            result = process_one(
                root,
                FakeGitHubClient({"https://api.github.com/repos/missing/repo": None}),
                "TEST_SHA",
                "3",
            )
            self.assertEqual(result["status"], "HOLD")
            self.assertEqual(
                result["primary_source_verification"]["hold_reason"],
                "REPOSITORY_NOT_FOUND",
            )




    def test_transient_source_error_is_deferred_and_persisted(self):
        class FailingClient(JsonClient):
            def __init__(self):
                pass

            def get(self, url):
                raise RuntimeError(f"http_401:{url}")

        queue = self._queue()
        queue["priority_gate"] = {}
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            path = root / "data" / "external_research_method_queue.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(queue), encoding="utf-8")
            (root / "data" / "external_research_runtime.json").write_text(
                json.dumps(runtime),
                encoding="utf-8",
            )
            result = process_one(root, FailingClient(), "TEST_SHA", "4")
            self.assertEqual(result["status"], "DEFERRED")
            self.assertFalse(result["production_changed"])
            self.assertFalse(result["promotion_allowed"])
            self.assertEqual(result["next_action"], "RETRY_SOURCE_VERIFICATION")
            self.assertTrue(result["retry_after_utc"])
            self.assertEqual(result["retry_count"], 1)
            persisted = json.loads(
                (root / "data" / "external_research_runtime.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                persisted["results"]["model/second"]["status"],
                "DEFERRED",
            )

    def test_deferred_candidate_does_not_starve_other_candidates(self):
        queue = self._queue()
        queue["priority_gate"] = {
            "immediate_local_reproduction": [
                {"repository": "model/second"},
                {"repository": "model/third"},
            ]
        }
        queue["candidates"].append(
            {"queue_rank": 3, "repository": "model/third", "next_gate": "LOCAL_ONLY"}
        )
        runtime = {
            "results": {
                "model/second": {
                    "status": "DEFERRED",
                    "retry_after_utc": "2999-01-01T00:00:00Z",
                },
            }
        }
        self.assertEqual(
            choose_next_candidate(queue, runtime)["repository"],
            "model/third",
        )

    def test_deferred_candidate_becomes_retryable_after_deadline(self):
        queue = self._queue()
        runtime = {
            "results": {
                "model/second": {
                    "status": "DEFERRED",
                    "retry_after_utc": "2000-01-01T00:00:00Z",
                },
            }
        }
        self.assertEqual(
            choose_next_candidate(queue, runtime)["repository"],
            "model/second",
        )

    def test_external_workflow_uses_live_github_expressions(self):
        workflow = Path('.github/workflows/btc_external_method_research.yml').read_text(encoding='utf-8')
        self.assertIn('GITHUB_TOKEN: ${{ github.token }}', workflow)
        self.assertIn('EXPECTED_SHA: ${{ github.sha }}', workflow)
        self.assertIn('btc-external-method-research-${{ github.run_id }}', workflow)
        self.assertIn('research: advance external method queue (${candidate})', workflow)
        self.assertNotIn('\\${{', workflow)
