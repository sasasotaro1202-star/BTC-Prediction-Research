import json
import tempfile
import unittest
from pathlib import Path

from src.external_method_research import (
    GitHubClient,
    choose_next_candidate,
    process_one,
)


class FakeGitHubClient(GitHubClient):
    def __init__(self, payloads):
        self.payloads = payloads

    def get(self, path):
        return self.payloads.get(path)


class ExternalMethodResearchTests(unittest.TestCase):
    def _queue(self):
        return {
            "schema_version": 2,
            "candidates": [
                {
                    "queue_rank": 1,
                    "repository": "agent/first",
                    "mechanism": "first mechanism",
                    "method_abstract": "first abstract",
                    "btc_relevance": "HIGH",
                    "next_gate": "LOCAL_ONLY",
                },
                {
                    "queue_rank": 2,
                    "repository": "model/second",
                    "mechanism": "second mechanism",
                    "method_abstract": "second abstract",
                    "btc_relevance": "HIGH",
                    "next_gate": "MODEL_OOS_GATE",
                },
            ],
            "priority_gate": {
                "immediate_local_reproduction": [
                    {"repository": "model/second", "gate": "MODEL_OOS_GATE"}
                ]
            },
        }

    def test_priority_gate_beats_queue_rank(self):
        candidate = choose_next_candidate(self._queue(), {})
        self.assertEqual(candidate["repository"], "model/second")

    def test_processed_terminal_candidate_is_skipped(self):
        runtime = {"results": {"model/second": {"status": "SOURCE_VERIFIED"}}}
        candidate = choose_next_candidate(self._queue(), runtime)
        self.assertEqual(candidate["repository"], "agent/first")

    def test_missing_repo_is_explicit_hold(self):
        payloads = {"/repos/missing/repo": None}
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            path = root / "data" / "external_research_method_queue.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(
                    {
                        **self._queue(),
                        "candidates": [
                            {
                                "queue_rank": 1,
                                "repository": "missing/repo",
                                "mechanism": "m",
                                "method_abstract": "a",
                                "btc_relevance": "HIGH",
                                "next_gate": "LOCAL_ONLY",
                            }
                        ],
                        "priority_gate": {},
                    }
                ),
                encoding="utf-8",
            )
            result = process_one(
                root,
                FakeGitHubClient(payloads),
                analysis_sha="TEST_SHA",
                run_id="1",
            )
            self.assertEqual(result["status"], "HOLD")
            self.assertFalse(result["production_changed"])
            saved = json.loads(
                (
                    root
                    / "data"
                    / "external_research_results"
                    / "missing__repo.json"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(
                saved["source_verification"]["hold_reason"],
                "REPOSITORY_NOT_FOUND",
            )

    def test_active_known_license_is_source_verified_without_performance_claim(self):
        payloads = {
            "/repos/model/second": {
                "archived": False,
                "default_branch": "main",
                "html_url": "https://github.com/model/second",
                "license": {"spdx_id": "MIT"},
                "updated_at": "2026-10-08T00:00:00Z",
                "pushed_at": "2026-10-08T00:00:00Z",
                "stargazers_count": 10,
            },
            "/repos/model/second/commits/main": {
                "sha": "abc123",
                "html_url": "https://github.com/model/second/commit/abc123",
            },
            "/repos/model/second/readme": {"sha": "readme123"},
        }
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            path = root / "data" / "external_research_method_queue.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(self._queue()), encoding="utf-8")
            result = process_one(
                root,
                FakeGitHubClient(payloads),
                analysis_sha="TEST_SHA",
                run_id="2",
            )
            self.assertEqual(result["status"], "SOURCE_VERIFIED")
            self.assertTrue(result["source_verification"]["source_verified"])
            self.assertFalse(result["external_performance_transfer_allowed"])
            self.assertFalse(result["production_changed"])
            self.assertEqual(
                result["source_verification"]["source_commit_sha"],
                "abc123",
            )


    def test_unknown_license_fails_closed(self):
        payloads = {
            "/repos/model/second": {
                "archived": False,
                "default_branch": "main",
                "html_url": "https://github.com/model/second",
                "license": None,
            }
        }
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            path = root / "data" / "external_research_method_queue.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(self._queue()), encoding="utf-8")
            result = process_one(
                root,
                FakeGitHubClient(payloads),
                analysis_sha="TEST_SHA",
                run_id="3",
            )
            self.assertEqual(result["status"], "HOLD")
            self.assertEqual(
                result["source_verification"]["hold_reason"],
                "LICENSE_UNVERIFIED",
            )

    def test_exhausted_queue_remains_research_only(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            path = root / "data" / "external_research_method_queue.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(self._queue()), encoding="utf-8")
            runtime = {
                "results": {
                    "model/second": {"status": "SOURCE_VERIFIED"},
                    "agent/first": {"status": "HOLD"},
                }
            }
            (root / "data" / "external_research_runtime.json").write_text(
                json.dumps(runtime),
                encoding="utf-8",
            )
            result = process_one(
                root,
                FakeGitHubClient({}),
                analysis_sha="TEST_SHA",
                run_id="4",
            )
            self.assertEqual(result["status"], "EXHAUSTED")
            self.assertTrue(result["research_only"])
            self.assertFalse(result["production_changed"])
            saved = json.loads(
                (root / "data" / "external_research_runtime.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertTrue(saved["queue_exhausted"])



    def test_source_verified_candidate_advances_to_local_gate(self):
        payloads = {
            "/repos/model/second": {
                "archived": False,
                "default_branch": "main",
                "html_url": "https://github.com/model/second",
                "license": {"spdx_id": "MIT"},
                "updated_at": "2026-10-08T00:00:00Z",
                "pushed_at": "2026-10-08T00:00:00Z",
                "stargazers_count": 10,
            },
            "/repos/model/second/commits/main": {"sha": "abc123", "html_url": "https://github.com/model/second/commit/abc123"},
            "/repos/model/second/readme": {"sha": "readme123"},
        }
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            path = root / "data" / "external_research_method_queue.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            queue = self._queue()
            queue["candidates"][1]["source_contracts"] = []
            path.write_text(json.dumps(queue), encoding="utf-8")
            result = process_one(
                root,
                FakeGitHubClient(payloads),
                analysis_sha="TEST_SHA",
                run_id="5",
            )
            self.assertEqual(result["status"], "LOCAL_GATE_READY")
            self.assertTrue(result["source_contract_verification"]["all_verified"])
            self.assertFalse(result["production_changed"])

    def test_source_contract_unknown_license_fails_closed(self):
        payloads = {
            "/repos/model/second": {
                "archived": False,
                "default_branch": "main",
                "html_url": "https://github.com/model/second",
                "license": {"spdx_id": "MIT"},
            },
            "/repos/model/second/commits/main": {"sha": "abc123"},
            "https://huggingface.co/api/models/unknown/model": None,
        }
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            path = root / "data" / "external_research_method_queue.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            queue = self._queue()
            queue["candidates"][1]["source_contracts"] = [
                {"kind": "github", "id": "model/second"},
                {"kind": "huggingface_model", "id": "unknown/model"},
            ]
            path.write_text(json.dumps(queue), encoding="utf-8")
            result = process_one(
                root,
                FakeGitHubClient(payloads),
                analysis_sha="TEST_SHA",
                run_id="6",
            )
            self.assertEqual(result["status"], "HOLD")
            self.assertIn("MODEL_REPOSITORY_NOT_FOUND", result["source_contract_verification"]["failures"])

