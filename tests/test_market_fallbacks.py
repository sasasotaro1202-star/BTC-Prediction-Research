import sys
import unittest
from urllib.error import HTTPError
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import market_data  # noqa: E402
import predict  # noqa: E402
from binance_ws import load_depth_cache  # noqa: E402


class TestMarketFallbacks(unittest.TestCase):
    def test_bybit_closed_parser_filters_open_candle_and_sorts(self):
        now = 1_800_000_000_000
        original = market_data.time.time
        try:
            market_data.time.time = lambda: now / 1000
            payload = {
                'result': {'list': [
                    [str(now - 60_000), '100', '101', '99', '100.5', '12', '0'],
                    [str(now), '101', '102', '100', '101.5', '13', '0'],
                    [str(now - 120_000), '98', '100', '97', '99', '11', '0'],
                ]}
            }
            rows = market_data.closed_bybit(payload)
            self.assertEqual([r[0] for r in rows], [now - 120_000, now - 60_000])
            self.assertEqual(rows[-1][4], 100.5)
        finally:
            market_data.time.time = original

    def test_cache_fallback_is_optional_and_well_shaped(self):
        result = market_data.cache_rows(120)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 4)
        rows, created, age_ms, fresh = result
        self.assertIsInstance(rows, list)
        self.assertIsInstance(created, str)
        self.assertTrue(age_ms is None or isinstance(age_ms, int))
        self.assertIsInstance(fresh, bool)
        if rows:
            self.assertEqual(len(rows[0]), 6)
            self.assertTrue(all(len(r) == 6 for r in rows))

    def test_binance_taker_uses_futures_api_host(self):
        original = market_data._get
        seen = []
        try:
            market_data._get = lambda url: seen.append(url) or [
                {'takerBuyVol': '2', 'takerSellVol': '1'}
            ]
            rows = market_data.binance_taker()
            self.assertEqual(rows[0]['takerBuyVol'], '2')
            self.assertEqual(len(seen), 1)
            self.assertTrue(seen[0].startswith(
                'https://fapi.binance.com/futures/data/takerBuySellVol?'
            ))
        finally:
            market_data._get = original

    def test_bybit_orderbook_imbalance_uses_v5_shape(self):
        payload = {
            'result': {
                'b': [['100', '5'], ['99', '3']],
                'a': [['101', '1'], ['102', '1']],
            }
        }
        self.assertAlmostEqual(predict.imbalance(payload, levels=25), 0.60, places=8)

    def test_fallback_model_loader_selects_source_specific_artifact(self):
        original_exists = predict.Path.exists
        original_load = predict.joblib.load
        seen = []
        try:
            predict.Path.exists = lambda self: True
            predict.joblib.load = lambda path: seen.append(str(path)) or object()
            predict.load_model('5m', 'coinbase')
            self.assertTrue(seen[-1].endswith('/models/coinbase_5m.joblib'))
        finally:
            predict.Path.exists = original_exists
            predict.joblib.load = original_load

    def test_evidence_gated_fallback_requires_matching_metadata_and_holdout_gain(self):
        import json
        import tempfile

        original_dir = predict.MODEL_DIR
        try:
            with tempfile.TemporaryDirectory() as td:
                predict.MODEL_DIR = Path(td)
                meta = {
                    "model_version": "coinbase_fallback.rf.v1",
                    "source": "Coinbase Exchange BTC-USD 1m candles",
                    "horizon": "5m",
                    "classes": ["DOWN", "FLAT", "UP"],
                    "features": predict.FEATURES,
                    "artifact": "coinbase_5m.joblib",
                    "holdout_n": 5394,
                    "holdout_metrics": {"logloss": 1.0302},
                    "baseline_metrics": {"logloss": 1.0482},
                }
                (predict.MODEL_DIR / "coinbase_5m.json").write_text(
                    json.dumps(meta), encoding="utf-8"
                )
                (predict.MODEL_DIR / "coinbase_5m.joblib").write_bytes(b"placeholder")
                class FakeModel:
                    classes_ = ["DOWN", "FLAT", "UP"]

                    def predict_proba(self, X):
                        return [[0.4, 0.2, 0.4] for _ in X]

                from unittest.mock import patch
                with patch.object(predict.joblib, "load", return_value=FakeModel()):
                    self.assertTrue(predict._fallback_model_ready("coinbase", "5m"))
                self.assertFalse(predict._fallback_model_ready("coinbase", "10m"))
        finally:
            predict.MODEL_DIR = original_dir

    def test_select_fallback_source_is_none_when_either_horizon_is_not_ready(self):
        original = predict._fallback_model_ready
        try:
            predict._fallback_model_ready = lambda source, horizon: horizon == "5m"
            self.assertIsNone(
                predict._select_fallback_source({"price_feature_fallback": "coinbase"})
            )
        finally:
            predict._fallback_model_ready = original

    def test_predict_model_directory_matches_production_artifacts(self):
        self.assertTrue(str(predict.MODEL_DIR).endswith('/models'))
        self.assertNotIn('/data/models', str(predict.MODEL_DIR))

    def test_historical_runner_treats_binance_418_as_recoverable(self):
        import historical_research_runner as runner
        self.assertIn(418, runner.RETRYABLE_HTTP)
        self.assertIn(429, runner.RETRYABLE_HTTP)
        self.assertIn(503, runner.RETRYABLE_HTTP)

    def test_historical_runner_routes_binance_418_to_archive_fallback(self):
        import historical_research_runner as runner
        original_request = runner._ORIGINAL_REQ_JSON
        original_fallback = runner._archive_fallback
        try:
            runner._ORIGINAL_REQ_JSON = lambda *args, **kwargs: (_ for _ in ()).throw(
                RuntimeError("HTTP Error 418: I'm a teapot")
            )
            seen = []
            runner._archive_fallback = lambda url, optional=False: seen.append((url, optional)) or [["archive-row"]]
            url = "https://fapi.binance.com/fapi/v1/klines?symbol=BTCUSDT&interval=1m&startTime=1000&endTime=2000"
            self.assertEqual(runner.resilient_req_json(url), [["archive-row"]])
            self.assertEqual(seen, [(url, False)])
        finally:
            runner._ORIGINAL_REQ_JSON = original_request
            runner._archive_fallback = original_fallback

    def test_ws_freshness_rejects_old_event_with_fresh_retrieval(self):
        now = 1_800_000_000_000
        original = market_data.time.time
        try:
            market_data.time.time = lambda: now / 1000
            rows = []
            for i in range(40):
                event = now - (39 - i) * 60_000
                rows.append({
                    "open_time_ms": event,
                    "event_time_ms": event,
                    "retrieved_at_ms": now,
                })
            self.assertEqual(len(market_data._fresh_ws_suffix(rows, minimum=40)), 40)

            stale = [dict(row) for row in rows]
            stale[-1]["event_time_ms"] = now - 181_000
            stale[-1]["retrieved_at_ms"] = now
            self.assertEqual(market_data._fresh_ws_suffix(stale, minimum=40), [])
        finally:
            market_data.time.time = original

    def test_depth_cache_rejects_stale_event_with_fresh_retrieval(self):
        import json
        import tempfile
        now = 1_800_000_000_000
        original = market_data.time.time
        try:
            market_data.time.time = lambda: now / 1000
            payload = {
                'schema_version': 1,
                'source': 'Binance USD-M Futures WebSocket',
                'stream': 'btcusdt@depth20@100ms',
                'snapshot': {
                    'e': 'depthUpdate',
                    'E': now - 181_000,
                    'u': 1,
                    'b': [[100, 2] for _ in range(20)],
                    'a': [[101, 2] for _ in range(20)],
                    'retrieved_at_ms': now,
                },
            }
            with tempfile.TemporaryDirectory() as td:
                path = Path(td) / 'depth.json'
                path.write_text(json.dumps(payload), encoding='utf-8')
                self.assertIsNone(load_depth_cache(path))
        finally:
            market_data.time.time = original
    def test_parallel_result_calls_preserve_success_and_failure(self):
        calls = {
            "ok": lambda: {"value": 1},
            "bad": lambda: (_ for _ in ()).throw(RuntimeError("boom")),
        }
        result = market_data._parallel_result_calls(calls)
        self.assertEqual(result["ok"], {"value": 1})
        self.assertIsInstance(result["bad"], RuntimeError)

    def test_error_label_exposes_http_status_without_response_body(self):
        err = HTTPError("https://fapi.binance.com/fapi/v1/klines", 403, "Forbidden", {}, None)
        self.assertEqual(market_data._error_label(err), "HTTPError:403")


if __name__ == '__main__':
    unittest.main()
