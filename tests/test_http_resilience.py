import unittest
from urllib.error import HTTPError

from src import historical_research, historical_research_runner, http_resilience


class TestHTTPResilience(unittest.TestCase):
    def test_retries_timeout_then_succeeds(self):
        class Response:
            def __enter__(self): return self
            def __exit__(self, *args): return False
            def read(self): return b'{"ok": true}'

        calls = {"n": 0}
        original = http_resilience.urlopen
        original_sleep = http_resilience.time.sleep
        try:
            def fake(_req, timeout=None):
                calls["n"] += 1
                if calls["n"] < 3:
                    raise TimeoutError("simulated timeout")
                return Response()
            http_resilience.urlopen = fake
            http_resilience.time.sleep = lambda _: None
            payload = http_resilience.request_json(
                "https://example.invalid",
                attempts=4,
                timeout=1,
                total_timeout=10,
            )
        finally:
            http_resilience.urlopen = original
            http_resilience.time.sleep = original_sleep

        self.assertEqual(calls["n"], 3)
        self.assertEqual(payload, {"ok": True})

    def test_non_retryable_http_error_is_not_retried(self):
        calls = {"n": 0}
        original = http_resilience.urlopen
        original_sleep = http_resilience.time.sleep
        try:
            def fake(_req, timeout=None):
                calls["n"] += 1
                raise HTTPError(
                    "https://example.invalid",
                    404,
                    "not found",
                    hdrs=None,
                    fp=None,
                )
            http_resilience.urlopen = fake
            http_resilience.time.sleep = lambda _: None
            with self.assertRaises(HTTPError):
                http_resilience.request_bytes(
                    "https://example.invalid",
                    attempts=5,
                    timeout=1,
                    total_timeout=10,
                )
        finally:
            http_resilience.urlopen = original
            http_resilience.time.sleep = original_sleep

        self.assertEqual(calls["n"], 1)

    def test_historical_research_routes_json_through_shared_transport(self):
        calls = []

        def fake_request_json(url, *, headers, timeout, attempts, total_timeout):
            calls.append((url, headers, timeout, attempts, total_timeout))
            return {"ok": True}

        original = historical_research.request_json
        try:
            historical_research.request_json = fake_request_json
            result = historical_research.req_json(
                "https://example.invalid/api",
                timeout=20,
                retries=4,
            )
        finally:
            historical_research.request_json = original

        self.assertEqual(result, {"ok": True})
        self.assertEqual(calls[0][0], "https://example.invalid/api")
        self.assertEqual(calls[0][1]["Accept"], "application/json")
        self.assertEqual(calls[0][3], 4)
        self.assertGreaterEqual(calls[0][4], 120.0)

    def test_historical_runner_routes_archives_through_shared_transport(self):
        calls = []

        def fake_request_bytes(url, *, headers, timeout, attempts, total_timeout):
            calls.append((url, headers, timeout, attempts, total_timeout))
            return b"archive"

        original = historical_research_runner.request_bytes
        try:
            historical_research_runner.request_bytes = fake_request_bytes
            result = historical_research_runner._download(
                "https://example.invalid/archive.zip",
                timeout=40,
            )
        finally:
            historical_research_runner.request_bytes = original

        self.assertEqual(result, b"archive")
        self.assertEqual(calls[0][0], "https://example.invalid/archive.zip")
        self.assertEqual(calls[0][1]["User-Agent"], historical_research_runner.USER_AGENT)
        self.assertEqual(calls[0][3], 5)
        self.assertGreaterEqual(calls[0][4], 120.0)


if __name__ == "__main__":
    unittest.main()
