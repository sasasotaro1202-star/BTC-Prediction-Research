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
            f"{source},fallback_not_allowed;production_fallback_disabled"
        )
def jst(dt): return dt.astimezone(timezone(timedelta(hours=9))).isoformat()