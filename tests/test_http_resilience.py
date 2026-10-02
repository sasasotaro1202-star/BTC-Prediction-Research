import unittest
from urllib.error import HTTPError

from src import http_resilience


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


if __name__ == "__main__":
    unittest.main()
