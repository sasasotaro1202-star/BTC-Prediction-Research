"""BTC-only live predictor with resilient multi-venue data and conservative probability fusion."""
from __future__ import annotations
import asyncio, json, math, sqlite3
from datetime import datetime, timezone, timedelta
from pathlib import Path
import joblib, numpy as np
from concurrent.futures import ThreadPoolExecutor
from db import DB, init_db
from live_data_policy import validate_live_inputs
from feature_schema import FEATURES
from market_data import BINANCE_WS_CACHE, resilient_1m_series, binance_depth, bybit_depth, binance_premium, binance_oi, binance_taker, bybit_funding, bybit_mark_price
from binance_ws import capture_depth_snapshot, capture_mark_price, load_cache as load_binance_ws_cache, load_depth_cache, taker_imbalance as ws_taker_imbalance
from microstructure_features import derive_market_flow_features
from runtime_production_model import resolve_production_model
from situation import summarize_situation

ROOT=Path(__file__).resolve().parents[1]
MODEL_DIR=ROOT/'models'
INTERVAL=300
MAX_LIVE_EVENT_AGE_SECONDS=180
CLASSES=["DOWN","FLAT","UP"]
PRODUCTION_FALLBACKS_ENABLED=False

def utcnow(): return datetime.now(timezone.utc)
def enforce_primary_production_source(status):
    """Production predictions use one benchmark venue; fallback venues remain research-only."""
    if not isinstance(status, dict):
        raise ValueError("live_prediction_inputs_incomplete:status_not_dict")
    source = str(status.get("price_feature_fallback", "none"))
    if source != "none" and not PRODUCTION_FALLBACKS_ENABLED:
        raise ValueError(
            "live_prediction_inputs_incomplete:price_feature_fallback="
            f"{source},production_fallback_disabled"
        )
def jst(dt): return dt.astimezone(timezone(timedelta(hours=9))).isoformat()
def next_grid(dt,steps=1):
    ts=int(dt.timestamp()); return datetime.fromtimestamp(((ts//INTERVAL)+steps)*INTERVAL,timezone.utc)
def validate_latest_event_time(event_ms, *, now=None):
    """Reject stale/future market candles before a directional prediction."""
    current=utcnow() if now is None else now
    event=datetime.fromtimestamp(int(event_ms)/1000,timezone.utc)
    age=(current-event).total_seconds()
    if age > MAX_LIVE_EVENT_AGE_SECONDS:
        raise ValueError(f"stale_live_market_event:{age:.0f}s")
    if age < -60:
        raise ValueError(f"future_live_market_event:{age:.0f}s")
    return event
def _ema(v,span):
    a=2/(span+1); e=float(v[0])
    for x in v[1:]: e=a*float(x)+(1-a)*e
    return e