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
from market_data import BINANCE_WS_CACHE, resilient_1m_series, derive_binance_taker_from_closed_klines, binance_archive_daily_taker_rows, binance_depth, bybit_depth, binance_premium, binance_oi, binance_taker, bybit_funding, bybit_mark_price
from binance_ws import capture_depth_snapshot, capture_mark_price, load_cache as load_binance_ws_cache, load_depth_cache, taker_imbalance as ws_taker_imbalance
from microstructure_features import derive_market_flow_features
from runtime_production_model import resolve_production_model
from situation import summarize_situation

ROOT=Path(__file__).resolve().parents[1]
MODEL_DIR=ROOT/'models'
INTERVAL=300
MAX_LIVE_EVENT_AGE_SECONDS=180
CLASSES=["DOWN","FLAT","UP"]

def utcnow(): return datetime.now(timezone.utc)
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
def latest_market_event_ms(fut, status):
    if not fut:
        raise ValueError('missing_live_market_rows')
    latest_open_ms = int(fut[-1][0])
    if status.get('binance_futures_transport') == 'websocket':
        return int(status.get('binance_futures_ws_event_time_ms', latest_open_ms + 60_000 - 1))
    # Closed 1m REST/archive rows expose candle-open time. Use the scheduled
    # close boundary rather than the older open time to avoid false stale rejects.
    return latest_open_ms + 60_000 - 1

def _ema(v,span):
    a=2/(span+1); e=float(v[0])
    for x in v[1:]: e=a*float(x)+(1-a)*e
    return e
def _ret(c,n):
    """Return a past-to-current return, but fail closed if history is insufficient."""
    if len(c) <= n:
        raise ValueError(f"insufficient_price_history_for_return:{n}")
    return c[-1]/c[-1-n]-1
def features(rows):
    if len(rows) < 31:
        raise ValueError(f"insufficient_price_history_for_features:{len(rows)}")
    c=np.asarray([float(x[4]) for x in rows],dtype=float); o=np.asarray([float(x[1]) for x in rows],dtype=float); h=np.asarray([float(x[2]) for x in rows],dtype=float); l=np.asarray([float(x[3]) for x in rows],dtype=float); v=np.asarray([float(x[5]) for x in rows],dtype=float)
    if not all(np.all(np.isfinite(a)) for a in (c,o,h,l,v)):
        raise ValueError('invalid_price_history_nonfinite')
    if np.any(c<=0) or np.any(o<=0) or np.any(h<=0) or np.any(l<=0) or np.any(v<0):
        raise ValueError('invalid_price_history_domain')
    if np.any(h < np.maximum(o,c)) or np.any(l > np.minimum(o,c)):
        raise ValueError('invalid_ohlc_relationship')
    p=c[-1]
    r1,r3,r5,r10,r15,r30=[_ret(c,n) for n in (1,3,5,10,15,30)]; a=r1-r3/3
    rv5=float(np.std(np.diff(c[-6:])/c[-6:-1])); rv10=float(np.std(np.diff(c[-11:])/c[-11:-1])); hi,lo=max(h[-10:]),min(l[-10:]); rp=(p-lo)/(hi-lo) if hi>lo else .5
    hi30,lo30=max(h[-30:]),min(l[-30:]); rp30=(p-lo30)/(hi30-lo30) if hi30>lo30 else .5
    body=(p-o[-1])/p; up=(h[-1]-max(o[-1],p))/p; low=(min(o[-1],p)-l[-1])/p; rvol=float(np.mean(v[-5:]))/max(1e-12,float(np.mean(v[-15:-5]))) if np.mean(v[-15:-5]) else 1.; vtrend=float(np.mean(v[-5:]))/max(1e-12,float(np.mean(v[-10:]))) if np.mean(v[-10:]) else 1.
    trend_alignment=(0.50*r5+0.30*r15+0.20*r30)
    return {'ret_1m':r1,'ret_3m':r3,'ret_5m':r5,'ret_10m':r10,'ret_15m':r15,'ret_30m':r30,'acceleration':a,'volatility_5m':rv5,'volatility_10m':rv10,'range_position_10m':rp,'range_position_30m':rp30,'body_1m':body,'upper_wick_1m':up,'lower_wick_1m':low,'volume_ratio':rvol,'volume_trend':vtrend,'ema_gap_5m':p/_ema(c[-20:],5)-1,'ema_gap_10m':p/_ema(c[-30:],10)-1,'trend_alignment':trend_alignment}
def imbalance(book,levels=25):
    if not isinstance(book,dict): raise ValueError('order_book_not_dict')
    bids=book.get('bids'); asks=book.get('asks')
    if (not bids or not asks) and isinstance(book.get('result'),dict):
        result=book['result']; bids=result.get('b') or result.get('bids'); asks=result.get('a') or result.get('asks')
    if not bids or not asks: raise ValueError('order_book_missing_bids_or_asks')
    bids=bids[:levels]; asks=asks[:levels]; b=sum(float(x[1]) for x in bids); a=sum(float(x[1]) for x in asks)
    if not math.isfinite(b) or not math.isfinite(a) or b+a<=0: raise ValueError('order_book_nonfinite_or_empty')
    return (b-a)/(b+a)
def structural(f,m):
    vol=max(.00025,f['volatility_10m'])
    score=(2.2*f['ret_1m']+1.6*f['ret_3m']+f['ret_5m']+.45*f['ret_10m']+.35*f['ret_15m']+.20*f['ret_30m'])/vol
    score+=.15*f['acceleration']/vol+.10*(1.2*f['ema_gap_5m']+.4*f['ema_gap_10m'])/vol
    score+=.07*math.log(max(.25,min(4,f['volume_ratio'])))+.05*(f['range_position_10m']-.5)+.04*(f['range_position_30m']-.5)
    # Secondary venue signals are additive only when freshly available.
    score+=.12*m.get('book_imbalance',0.0)
    if m.get('cross_exchange_gap') is not None:
        score+=.08*m['cross_exchange_gap']/vol
    score+=.08*m.get('bybit_book_imbalance',0.0)
    taker=m.get('taker_imbalance',0.0)
    if taker>.55: score+=.10
    elif taker<-.55: score-=.10
    crowd=max(-1.0,min(1.0,m.get('funding_binance',m.get('funding_bybit',0.0))/0.0003)); score-=.05*crowd
    score=max(-2.5,min(2.5,score)); up=1/(1+math.exp(-score)); flat=max(.08,min(.40,.25-.055*min(2.5,abs(score)))); up=(1-flat)*up; return {'DOWN':1-up-flat,'FLAT':flat,'UP':up}
def _fallback_model_ready(source, horizon):
    """Allow only a separately trained, evidence-bearing venue fallback."""
    if source not in {'bybit', 'coinbase'} or horizon not in {'5m', '10m'}:
        return False
    meta_path = MODEL_DIR / f'{source}_{horizon}.json'
    artifact_path = MODEL_DIR / f'{source}_{horizon}.joblib'
    if not meta_path.is_file() or meta_path.stat().st_size <= 0:
        return False
    if not artifact_path.is_file() or artifact_path.stat().st_size <= 0:
        return False
    try:
        meta = json.loads(meta_path.read_text(encoding='utf-8'))
        if meta.get('horizon') != horizon:
            return False
        if meta.get('artifact') != artifact_path.name:
            return False
        if meta.get('classes') != CLASSES:
            return False
        if meta.get('features') != list(FEATURES):
            return False
        if not str(meta.get('model_version','')).startswith(f'{source}_fallback.'):
            return False
        holdout = meta.get('holdout_metrics') or {}
        baseline = meta.get('baseline_metrics') or {}
        n = int(meta.get('holdout_n', 0))
        ll = float(holdout.get('logloss'))
        base_ll = float(baseline.get('logloss'))
        if n < 5000 or not math.isfinite(ll) or not math.isfinite(base_ll):
            return False
        if ll > base_ll - 0.005:
            return False
        # The metadata contract alone is insufficient for production fallback.
        # Validate the serialized estimator before allowing it to become the
        # selected source; malformed/partial artifacts must fail closed.
        model = joblib.load(artifact_path)
        if [str(x) for x in getattr(model, 'classes_', [])] != CLASSES:
            return False
        if not callable(getattr(model, 'predict_proba', None)):
            return False
        return True
    except (OSError, TypeError, ValueError, json.JSONDecodeError, EOFError, ImportError):
        return False


def _select_fallback_source(status):
    """Return a gated venue fallback source, or None to remain fail-closed."""
    if not isinstance(status, dict):
        return None
    source = str(status.get('price_feature_fallback', ''))
    if source not in {'bybit', 'coinbase'}:
        return None
    if all(_fallback_model_ready(source, h) for h in ('5m', '10m')):
        return source
    return None


def load_model(h, source='primary'):
    """Load a validated model; production resolves only the primary bundle normally."""
    if source == 'primary':
        return resolve_production_model(h).load()
    prefix = f'{source}_{h}'
    p=MODEL_DIR/f'{prefix}.joblib'
    if not p.exists(): raise FileNotFoundError(f'model_missing:{source}:{h}')
    try:
        return joblib.load(p)
    except Exception as exc:
        raise RuntimeError(f'model_load_failed:{source}:{h}:{type(exc).__name__}') from exc
def model_probs(model,f):
    raw=model.predict_proba(np.array([[f[k] for k in FEATURES]]))[0]; out={c:1e-6 for c in CLASSES}
    for c,p in zip(model.classes_,raw):out[str(c)]=float(p)
    if not all(math.isfinite(v) and v>=0 for v in out.values()): raise ValueError('model_probability_nonfinite')
    s=sum(out.values())
    if not math.isfinite(s) or s<=0: raise ValueError('model_probability_invalid_sum')
    return {k:v/s for k,v in out.items()}
def regver(h):
    return resolve_production_model(h).model_version
def load_temperature(h):
    p=MODEL_DIR/f'{h}.calibration.json'
    try:
        obj=json.loads(p.read_text(encoding='utf-8')); t=float(obj.get('temperature',1.0)); n=int(obj.get('n_settled',0)); calibrated_version=str(obj.get('model_version','')); current_version=regver(h)
        if calibrated_version != current_version:return 1.0
        if not (0.5<=t<=3.0) or n<300:return 1.0
        return t
    except Exception:return 1.0
def load_blend_weight(h):
    p=MODEL_DIR/f'{h}.blend.json'
    try:
        obj=json.loads(p.read_text(encoding='utf-8')); w=float(obj.get('base_weight',0.20)); n=int(obj.get('n',0)); status=str(obj.get('status','')); calibrated_version=str(obj.get('model_version','')); current_version=regver(h)
        if calibrated_version != current_version:return 0.0
        if n<400 or status != 'accepted': return 0.0
        if not math.isfinite(w) or not (0.0<=w<=0.45): return 0.0
        return w
    except Exception:return 0.0
def load_fallback_calibration(source, h):
    """Load source-native fallback calibration only when it matches the model generation."""
    if source not in {"bybit", "coinbase"} or h not in {"5m", "10m"}:
        return {"temperature": 1.0, "blend_weight": 0.0, "status": "invalid_request", "n": 0}
    path = MODEL_DIR / f"{source}_{h}.calibration.json"
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        meta = json.loads((MODEL_DIR / f"{source}_{h}.json").read_text(encoding="utf-8"))
        if obj.get("model_version") != meta.get("model_version"):
            return {"temperature": 1.0, "blend_weight": 0.0, "status": "stale_model_version", "n": 0}
        n = int(obj.get("n", 0))
        status = str(obj.get("status", ""))
        t = float(obj.get("temperature", 1.0))
        w = float(obj.get("blend_weight", 0.0))
        if n < 80 or status != "accepted":
            return {"temperature": 1.0, "blend_weight": 0.0, "status": "not_accepted", "n": n}
        if not (0.5 <= t <= 3.0) or not (0.0 <= w <= 0.45):
            return {"temperature": 1.0, "blend_weight": 0.0, "status": "invalid_bounds", "n": n}
        return {"temperature": t, "blend_weight": w, "status": status, "n": n}
    except Exception:
        return {"temperature": 1.0, "blend_weight": 0.0, "status": "unavailable", "n": 0}


def calibrate_fallback_probs(probs, source, h):
    c = load_fallback_calibration(source, h)
    t = c["temperature"]
    if t == 1.0:
        return probs
    p = np.clip(np.asarray([probs["DOWN"], probs["FLAT"], probs["UP"]], float), 1e-6, 1-1e-6)
    p /= p.sum()
    z = np.log(p) / t
    z -= z.max()
    q = np.exp(z)
    q /= q.sum()
    return {"DOWN": float(q[0]), "FLAT": float(q[1]), "UP": float(q[2])}


def apply_structural_weight(base, structural, weight):
    w = max(0.0, min(0.45, float(weight)))
    if w <= 0.0:
        return base
    p = np.asarray([base["DOWN"], base["FLAT"], base["UP"]], dtype=float)
    q = np.asarray([structural["DOWN"], structural["FLAT"], structural["UP"]], dtype=float)
    out = np.clip((1.0 - w) * p + w * q, 1e-6, 1.0)
    out /= out.sum()
    return {"DOWN": float(out[0]), "FLAT": float(out[1]), "UP": float(out[2])}


def calibrate_probs(probs,h):
    t=load_temperature(h)
    if t==1.0:return probs
    p=np.clip(np.asarray([probs['DOWN'],probs['FLAT'],probs['UP']],float),1e-6,1-1e-6); p/=p.sum(); z=np.log(p)/t; z-=z.max(); q=np.exp(z); q/=q.sum()
    return {'DOWN':float(q[0]),'FLAT':float(q[1]),'UP':float(q[2])}
def fuse(base,struct,m,data_complete,horizon):
    p=np.array([base['DOWN'],base['FLAT'],base['UP']]); q=np.array([struct['DOWN'],struct['FLAT'],struct['UP']])
    base_w=load_blend_weight(horizon)
    # Contextual agreement weighting is research-only until it has its own
    # chronological OOS evidence. A rejected/unvalidated blend therefore cannot
    # re-enter production merely because secondary venues are available.
    if data_complete:
        w=base_w
    else:
        w=min(base_w,.15)
    w=max(0.0,min(.45,w)); out=(1-w)*p+w*q; out=np.clip(out,.03,.94); out/=out.sum()
    return {'DOWN':float(out[0]),'FLAT':float(out[1]),'UP':float(out[2])},float(w)
def _parallel_market_calls(calls):
    """Fetch independent public market-data endpoints concurrently."""
    if not calls:
        return {}
    results = {}
    with ThreadPoolExecutor(max_workers=min(6, len(calls))) as pool:
        future_map = {pool.submit(fn): key for key, fn in calls.items()}
        for future, key in ((f, future_map[f]) for f in future_map):
            try:
                results[key] = future.result()
            except Exception as exc:
                results[key] = exc
    return results

def _validate_persisted_provenance(scenario, now):
    """Fail closed before persisting any new prediction without auditable PIT metadata."""
    if not isinstance(scenario, dict):
        raise ValueError("prediction_provenance_missing")
    provenance = scenario.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError("prediction_provenance_missing")
    decision = scenario.get("decision_time_utc")
    available = provenance.get("available_at")
    retrieved = provenance.get("retrieved_at")
    cutoff = provenance.get("prediction_cutoff")
    if not all(isinstance(v, str) and v for v in (decision, available, retrieved, cutoff)):
        raise ValueError("prediction_provenance_incomplete")
    try:
        decision_dt = datetime.fromisoformat(decision.replace("Z", "+00:00"))
        available_dt = datetime.fromisoformat(available.replace("Z", "+00:00"))
        retrieved_dt = datetime.fromisoformat(retrieved.replace("Z", "+00:00"))
        cutoff_dt = datetime.fromisoformat(cutoff.replace("Z", "+00:00"))
    except (TypeError, ValueError):
        raise ValueError("prediction_provenance_invalid_timestamp")
    if decision_dt.tzinfo is None or available_dt.tzinfo is None or retrieved_dt.tzinfo is None or cutoff_dt.tzinfo is None:
        raise ValueError("prediction_provenance_timezone_required")
    # Acquisition/retrieval must not occur after the prediction cutoff; allowing
    # otherwise would create a post-decision feature snapshot.
    if available_dt > cutoff_dt or retrieved_dt > cutoff_dt or cutoff_dt > decision_dt + timedelta(seconds=1):
        raise ValueError("prediction_provenance_temporal_violation")
    sources = provenance.get("sources")
    if not isinstance(sources, dict) or not sources:
        raise ValueError("prediction_provenance_sources_missing")
    valid_sources = 0
    for key, item in sources.items():
        if not isinstance(item, dict):
            continue
        status = item.get("status")
        if status in {"ok", "ok_current_only"}:
            item_available = item.get("available_at")
            item_retrieved = item.get("retrieved_at")
            item_cutoff = item.get("prediction_cutoff")
            if not all(isinstance(v, str) and v for v in (item_available, item_retrieved, item_cutoff)):
                raise ValueError(f"prediction_source_provenance_incomplete:{key}")
            try:
                source_available = datetime.fromisoformat(item_available.replace("Z", "+00:00"))
                source_retrieved = datetime.fromisoformat(item_retrieved.replace("Z", "+00:00"))
                source_cutoff = datetime.fromisoformat(item_cutoff.replace("Z", "+00:00"))
            except (TypeError, ValueError):
                raise ValueError(f"prediction_source_provenance_invalid_timestamp:{key}")
            if source_available.tzinfo is None or source_retrieved.tzinfo is None or source_cutoff.tzinfo is None:
                raise ValueError(f"prediction_source_provenance_timezone_required:{key}")
            if source_available > source_cutoff or source_retrieved > source_cutoff or source_cutoff > decision_dt + timedelta(seconds=1):
                raise ValueError(f"prediction_source_provenance_temporal_violation:{key}")
            valid_sources += 1
    if valid_sources == 0:
        raise ValueError("prediction_provenance_no_valid_sources")


def insert_prediction(now,target5,target10,price,p5,p10,model_version,features_json,scenario):
    _validate_persisted_provenance(scenario, now)
    with sqlite3.connect(DB) as c:
        c.execute('INSERT INTO predictions(created_at_utc,target_5m,target_10m,base_price,p_up_5m,p_down_5m,p_flat_5m,p_up_10m,p_down_10m,p_flat_10m,model_version,feature_json,scenario_json) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)',(now.isoformat(),target5.isoformat(),target10.isoformat(),price,p5['UP'],p5['DOWN'],p5['FLAT'],p10['UP'],p10['DOWN'],p10['FLAT'],model_version,json.dumps(features_json),json.dumps(scenario)))
def main():
    init_db(); now=utcnow(); fut,spot,by,status=resilient_1m_series()
    # Production live runs remain Binance-primary. Cross-venue fallback models
    # are research/recovery artifacts only and are never allowed to create a
    # production prediction when required Binance inputs are unavailable.
    use_bybit_fallback = False
    use_coinbase_fallback = False
    use_fallback = False
    if len(fut)<40: raise SystemExit('live_prediction_fail_closed: insufficient futures data')
    f=features(fut); price=float(fut[-1][4]); spotp=float(spot[-1][4]) if len(spot)>=40 else None
    m={}
    # Research-only derived flow features use already-closed 1m observations.
    # They never modify the 15-feature Production Champion input vector.
    ws_candidate = load_binance_ws_cache(BINANCE_WS_CACHE, 120)
    ws_fresh = False
    if ws_candidate:
        try:
            freshest = max(int(r["retrieved_at_ms"]) for r in ws_candidate)
            ws_fresh = (int(datetime.now(timezone.utc).timestamp() * 1000) - freshest) <= 180_000
        except (KeyError, TypeError, ValueError):
            ws_fresh = False
    flow = derive_market_flow_features(
        fut,
        ws_rows=ws_candidate if ws_fresh else None,
    )
    for key, value in flow.items():
        if value is not None:
            m[key] = float(value)
    status["market_flow_v2"] = "ok" if any(v is not None for v in flow.values()) else "missing"
    if flow.get("taker_imbalance_15m") is not None and ws_fresh:
        latest_ws = max(ws_candidate, key=lambda row: int(row["open_time_ms"]))
        status["binance_taker_window_transport"] = "websocket_closed_klines"
        status["binance_taker_window_rows"] = len(ws_candidate)
        status["binance_taker_window_event_time_ms"] = int(latest_ws["event_time_ms"])
        status["binance_taker_window_retrieved_at_ms"] = max(
            int(r["retrieved_at_ms"]) for r in ws_candidate
        )
    ws_taker_result = ws_taker_imbalance(ws_candidate, 5) if ws_fresh else None
    kline_taker_result = None
    if ws_taker_result is None and status.get("binance_futures_transport") in {"rest", "websocket"}:
        kline_taker_result = derive_binance_taker_from_closed_klines(
            fut,
            window=5,
            cutoff_ms=int(datetime.now(timezone.utc).timestamp() * 1000),
        )
    archive_taker_result = None
    if ws_taker_result is None and kline_taker_result is None and status.get("binance_futures_transport") == "binance_vision_daily_archive":
        try:
            archive_taker_rows = binance_archive_daily_taker_rows(5)
            archive_taker_result = derive_binance_taker_from_closed_klines(
                archive_taker_rows,
                window=5,
                cutoff_ms=int(datetime.now(timezone.utc).timestamp() * 1000),
            )
        except Exception:
            archive_taker_result = None

    # Prefer a fresh WS depth snapshot over Binance REST. A short direct-WS
    # recovery is attempted only when the rolling cache is unavailable.
    ws_book = load_depth_cache()
    if ws_book is None:
        try:
            ws_book = asyncio.run(capture_depth_snapshot(6.0))
        except Exception:
            ws_book = None

    # Prefer Binance mark-price WS for funding; REST is a bounded recovery path.
    ws_mark = None
    try:
        ws_mark = asyncio.run(capture_mark_price(6.0))
    except Exception:
        ws_mark = None

    market_call_defs = {
        "bybit_depth": bybit_depth,
        "binance_oi": binance_oi,
        "bybit_funding": bybit_funding,
    }
    if ws_book is None:
        market_call_defs["binance_depth"] = binance_depth
    if ws_mark is None:
        market_call_defs["binance_premium"] = binance_premium
    if ws_taker_result is None and kline_taker_result is None and archive_taker_result is None:
        market_call_defs["binance_taker"] = binance_taker
    market_calls = _parallel_market_calls(market_call_defs)

    depth_result = ws_book if ws_book is not None else market_calls.get("binance_depth")
    try:
        if isinstance(depth_result, Exception) or depth_result is None:
            raise RuntimeError('binance_depth_unavailable')
        m['book_imbalance']=imbalance(depth_result)
        status['binance_depth']='ok'
        if ws_book is not None:
            status['binance_depth_transport']='websocket_cache_or_direct'
            status['binance_depth_event_time_ms']=int(ws_book['event_time_ms'])
            status['binance_depth_ws_retrieved_at_ms']=int(ws_book['retrieved_at_ms'])
            status['binance_depth_ws_levels']=min(len(ws_book['bids']),len(ws_book['asks']))
        else:
            status['binance_depth_transport']='rest'
    except Exception as exc:
        status['binance_depth']=f'error:{type(exc).__name__}'

    bybit_book = market_calls.get("bybit_depth")
    try:
        if isinstance(bybit_book, Exception):
            raise bybit_book
        m['bybit_book_imbalance']=imbalance(bybit_book)
        status['bybit_depth']='ok'
    except Exception as exc:
        status['bybit_depth']=f'error:{type(exc).__name__}'

    # Bybit is a secondary cross-venue signal. It is useful when available,
    # but an outage must not block an otherwise valid production prediction.
    # Prefer the latest closed candle when available; otherwise query the
    # current linear-market ticker. This preserves the venue-divergence signal
    # without requiring an unrelated 40-bar contiguous Bybit history.
    def _valid_price(value):
        try:
            value=float(value)
            return value if math.isfinite(value) and value>0 else None
        except (TypeError,ValueError):
            return None

    def _book_midprice(book):
        if not isinstance(book,dict):
            return None
        result=book.get('result',{}) if isinstance(book.get('result',{}),dict) else {}
        bids=result.get('b') or result.get('bids') or book.get('bids')
        asks=result.get('a') or result.get('asks') or book.get('asks')
        if not bids or not asks:
            return None
        try:
            bid=_valid_price(bids[0][0]); ask=_valid_price(asks[0][0])
            if bid is None or ask is None or ask < bid:
                return None
            return (bid+ask)/2.0
        except (TypeError,ValueError,IndexError):
            return None

    # Prefer the already-fetched Bybit candle/current row. If that is not
    # usable, try the ticker once; if that response is malformed or unavailable,
    # fall back to the already-required order book.
    byp=_valid_price(by[-1][4]) if by else None
    if byp is not None:
        status['bybit_futures']='ok_current_only' if len(by) == 1 else status.get('bybit_futures','ok')
    else:
        try:
            ticker=bybit_mark_price()
            rows=ticker.get('result',{}).get('list',[]) if isinstance(ticker,dict) else []
            if rows:
                row=rows[0]
                byp=_valid_price(row.get('lastPrice') or row.get('markPrice'))
                if byp is None:
                    bid=_valid_price(row.get('bid1Price'))
                    ask=_valid_price(row.get('ask1Price'))
                    if bid is not None and ask is not None and ask >= bid:
                        byp=(bid+ask)/2.0
            if byp is not None:
                status['bybit_futures']='ok_current_only'
        except Exception:
            pass
        if byp is None:
            byp=_book_midprice(bybit_book) if not isinstance(bybit_book, Exception) else None
            if byp is not None:
                status['bybit_futures']='ok_current_only'
            else:
                status['bybit_futures']='error:missing_current_price'
    if byp is not None and math.isfinite(byp) and byp>0:
        m['cross_exchange_gap']=byp/price-1
    else:
        m['cross_exchange_gap']=None
        status['bybit_futures']=status.get('bybit_futures','error:missing_current_price')

    premium_result = ws_mark if ws_mark is not None else market_calls.get("binance_premium")
    try:
        if premium_result is None or isinstance(premium_result, Exception):
            raise RuntimeError('binance_premium_unavailable')
        if ws_mark is not None:
            m['funding_binance']=float(ws_mark['funding_rate'])
            status['binance_premium']='ok'
            status['binance_premium_transport']='websocket'
            status['binance_premium_event_time_ms']=int(ws_mark['event_time_ms'])
            status['binance_premium_retrieved_at_ms']=int(ws_mark['retrieved_at_ms'])
        else:
            m['funding_binance']=float(premium_result.get('lastFundingRate'))
            status['binance_premium']='ok'
            status['binance_premium_transport']='rest'
            if premium_result.get('time') is not None:
                status['binance_premium_event_time_ms']=int(premium_result['time'])
    except Exception as exc:
        status['binance_premium']=f'error:{type(exc).__name__}'

    oi_result = market_calls.get("binance_oi")
    try:
        if isinstance(oi_result, Exception):
            raise oi_result
        m['oi']=float(oi_result.get('openInterest'))
        status['binance_oi']='ok'
    except Exception as exc:
        status['binance_oi']=f'error:{type(exc).__name__}'

    taker_result = (
        ws_taker_result
        if ws_taker_result is not None
        else kline_taker_result
        if kline_taker_result is not None
        else archive_taker_result
        if archive_taker_result is not None
        else market_calls.get("binance_taker")
    )
    try:
        if ws_taker_result is not None:
            m['taker_imbalance'], event_ms = ws_taker_result
            latest_rows=[r for r in ws_candidate if int(r['open_time_ms']) >= int(ws_candidate[-1]['open_time_ms'])-4*60_000]
            freshest_retrieved=max(int(r['retrieved_at_ms']) for r in latest_rows) if latest_rows else 0
            now_ms=int(datetime.now(timezone.utc).timestamp()*1000)
            if now_ms - freshest_retrieved > 180_000 or freshest_retrieved > now_ms + 60_000:
                raise RuntimeError('websocket_taker_cache_stale_or_future')
            m['taker_imbalance']=float(m['taker_imbalance'])
            status['binance_taker']='ok'
            status['binance_taker_transport']='websocket_derived_from_closed_klines'
            status['binance_taker_event_time_ms']=int(event_ms)
            status['binance_taker_retrieved_at_ms']=int(freshest_retrieved)
        elif kline_taker_result is not None:
            m['taker_imbalance']=float(kline_taker_result['taker_imbalance'])
            status['binance_taker']='ok'
            status['binance_taker_transport']='closed_kline_derived'
            status['binance_taker_event_time_ms']=int(kline_taker_result['event_time_ms'])
            status['binance_taker_retrieved_at_ms']=int(kline_taker_result['retrieved_at_ms'])
            status['binance_taker_rows']=int(kline_taker_result['rows'])
        elif archive_taker_result is not None:
            m['taker_imbalance']=float(archive_taker_result['taker_imbalance'])
            status['binance_taker']='ok'
            status['binance_taker_transport']='binance_vision_closed_klines'
            status['binance_taker_event_time_ms']=int(archive_taker_result['event_time_ms'])
            status['binance_taker_retrieved_at_ms']=int(archive_taker_result['retrieved_at_ms'])
            status['binance_taker_rows']=int(archive_taker_result['rows'])
        else:
            if taker_result is None or isinstance(taker_result, Exception):
                raise RuntimeError('binance_taker_unavailable')
            t=taker_result
            t=t[-1] if isinstance(t,list) and t else t
            tb=float(t['takerBuyVol']); ts=float(t['takerSellVol'])
            m['taker_imbalance']=(tb-ts)/max(1e-12,tb+ts)
            status['binance_taker']='ok'
            status['binance_taker_transport']='rest'
    except Exception as exc:
        status['binance_taker']=f'error:{type(exc).__name__}'

    funding_result = market_calls.get("bybit_funding")
    try:
        if isinstance(funding_result, Exception):
            raise funding_result
        m['funding_bybit']=float(funding_result.get('result',{}).get('list',[{}])[0]['fundingRate'])
        status['bybit_funding']='ok'
    except Exception as exc:
        status['bybit_funding']=f'error:{type(exc).__name__}'

    if spotp is not None:m['spot_futures_gap']=spotp/price-1
    # Binance remains the normal production venue. A separately trained,
    # evidence-gated fallback may be used only when the data adapter explicitly
    # reports a Bybit/Coinbase price-history fallback.
    fallback_source = _select_fallback_source(status)
    use_bybit_fallback = fallback_source == 'bybit'
    use_coinbase_fallback = fallback_source == 'coinbase'
    use_fallback = use_bybit_fallback or use_coinbase_fallback
    if use_bybit_fallback:
        # Use neutral Binance-only microstructure because the Binance-native
        # signals are unavailable in this degraded venue mode.
        m['book_imbalance'] = m.get('bybit_book_imbalance', 0.0)
        m['taker_imbalance'] = 0.0
        m['funding_binance'] = 0.0
    validate_live_inputs(
        status,
        fut_rows=len(fut),
        spot_rows=len(spot),
        bybit_rows=len(by),
        allow_bybit_fallback=use_bybit_fallback,
        allow_coinbase_fallback=use_coinbase_fallback,
    )
    prediction_cutoff=utcnow()
    now=prediction_cutoff
    latest_event_ms=latest_market_event_ms(fut, status)
    latest_event=validate_latest_event_time(latest_event_ms, now=prediction_cutoff)
    s5=structural(f,m)
    gap=m.get('cross_exchange_gap')
    s10=structural(f,{**m,'cross_exchange_gap':(gap*.8 if gap is not None else None)})
    model_source = 'bybit' if use_bybit_fallback else ('coinbase' if use_coinbase_fallback else 'primary')
    base5=model_probs(load_model('5m', model_source),f); base10=model_probs(load_model('10m', model_source),f)
    # Secondary venue completeness is descriptive only unless the feature has
    # source-native timing/provenance. Bybit current snapshots may be useful for
    # diagnostics, but they must not silently increase model weight merely because
    # a response exists.
    data_complete = (
        byp is not None
        and 'bybit_book_imbalance' in m
        and status.get('bybit_futures') in {'ok', 'ok_current_only'}
        and status.get('bybit_depth') == 'ok'
    )
    if use_fallback:
        prefix = 'bybit' if use_bybit_fallback else 'coinbase'
        fallback5 = load_fallback_calibration(prefix, '5m')
        fallback10 = load_fallback_calibration(prefix, '10m')
        w5 = float(fallback5.get('blend_weight', 0.0))
        w10 = float(fallback10.get('blend_weight', 0.0))
        raw5 = calibrate_fallback_probs(base5, prefix, '5m')
        raw10 = calibrate_fallback_probs(base10, prefix, '10m')
        p5 = apply_structural_weight(raw5, s5, w5)
        p10 = apply_structural_weight(raw10, s10, w10)
    else:
        raw5,w5=fuse(base5,s5,m,data_complete,'5m'); raw10,w10=fuse(base10,s10,m,data_complete,'10m')
        p5=calibrate_probs(raw5,'5m'); p10=calibrate_probs(raw10,'10m')
    target5=next_grid(now,1); target10=next_grid(now,2); direction=max(p5,key=p5.get); regime='TREND' if abs(f['trend_alignment'])>max(.0007,1.5*f['volatility_10m']) else 'RANGE'; warnings=[]
    situation = summarize_situation(f, m, p5, p10, data_quality=status)
    if m.get('cross_exchange_gap') is not None and abs(m['cross_exchange_gap'])>.0005:warnings.append('cross-exchange divergence')
    if abs(m.get('book_imbalance',0.0))>.45 or abs(m.get('bybit_book_imbalance',0.0))>.45:warnings.append('order-book imbalance')
    if abs(m.get('taker_imbalance',0.0))>.55:warnings.append('taker-flow imbalance')
    if abs(m.get('funding_binance',0.0))>.0002:warnings.append('elevated funding')
    if abs(f['ret_15m'])>.003 or abs(f['ret_30m'])>.005:warnings.append('higher-timeframe impulse')
    retrieved=prediction_cutoff.isoformat()
    # Conservative live provenance: the system's observation is treated as
    # available no later than the end-of-acquisition cutoff. Source-native
    # publication/revision timestamps are unknown unless the adapter exposes
    # them, so they remain explicit nulls rather than invented values.
    source_provenance={}
    def _iso_ms(value):
        return None if value in (None, '') else datetime.fromtimestamp(int(value)/1000, timezone.utc).isoformat()

    binance_futures_event = _iso_ms(status.get('binance_futures_ws_event_time_ms')) if status.get('binance_futures_transport') == 'websocket' else latest_event.isoformat()
    binance_futures_available = _iso_ms(status.get('binance_futures_ws_retrieved_at_ms')) if status.get('binance_futures_transport') == 'websocket' else retrieved
    source_provenance['binance_futures']={
        'information_origin':'Binance',
        'transport':status.get('binance_futures_transport','rest'),
        'event_time':binance_futures_event,
        'available_at':binance_futures_available,
        'publication_time':None,
        'retrieved_at':retrieved,
        'revision_time':None,
        'prediction_cutoff':retrieved,
        'status':status.get('binance_futures'),
    }

    depth_event=_iso_ms(status.get('binance_depth_event_time_ms')) if status.get('binance_depth_event_time_ms') else None
    depth_available=_iso_ms(status.get('binance_depth_ws_retrieved_at_ms')) if str(status.get('binance_depth_transport','')).startswith('websocket') else retrieved
    source_provenance['binance_depth']={
        'information_origin':'Binance',
        'transport':status.get('binance_depth_transport','rest'),
        'event_time':depth_event,
        'available_at':depth_available,
        'publication_time':None,
        'retrieved_at':retrieved,
        'revision_time':None,
        'prediction_cutoff':retrieved,
        'status':status.get('binance_depth'),
        'levels':status.get('binance_depth_ws_levels'),
    }

    taker_available=_iso_ms(status.get('binance_taker_retrieved_at_ms')) if status.get('binance_taker_transport','').startswith('websocket') else retrieved
    source_provenance['binance_taker']={
        'information_origin':'Binance',
        'transport':status.get('binance_taker_transport','rest'),
        'event_time':_iso_ms(status.get('binance_taker_event_time_ms')),
        'available_at':taker_available,
        'publication_time':None,
        'retrieved_at':retrieved,
        'revision_time':None,
        'prediction_cutoff':retrieved,
        'status':status.get('binance_taker'),
    }
    if status.get('binance_taker_window_transport') == 'websocket_closed_klines':
        window_retrieved = _iso_ms(status.get('binance_taker_window_retrieved_at_ms'))
        source_provenance['binance_taker_window']={
            'information_origin':'Binance',
            'transport':'websocket_closed_klines',
            'event_time':_iso_ms(status.get('binance_taker_window_event_time_ms')),
            'available_at':window_retrieved,
            'publication_time':None,
            'retrieved_at':window_retrieved,
            'revision_time':None,
            'prediction_cutoff':retrieved,
            'status':'ok',
        }

    premium_available=_iso_ms(status.get('binance_premium_retrieved_at_ms')) if status.get('binance_premium_transport') == 'websocket' else retrieved
    source_provenance['binance_premium']={
        'information_origin':'Binance',
        'transport':status.get('binance_premium_transport','rest'),
        'event_time':_iso_ms(status.get('binance_premium_event_time_ms')),
        'available_at':premium_available,
        'publication_time':None,
        'retrieved_at':retrieved,
        'revision_time':None,
        'prediction_cutoff':retrieved,
        'status':status.get('binance_premium'),
    }
    # Bybit adapters used here expose current market snapshots but do not expose
    # a trustworthy source-native event/publication timestamp. Never borrow the
    # Binance candle timestamp for another venue: that would falsely imply PIT
    # alignment. Keep event_time explicitly unknown until the adapter supplies it.
    for source_key in ('bybit_futures','bybit_depth','bybit_funding'):
        source_provenance[source_key]={
            'information_origin':'Bybit',
            'event_time':None,
            'available_at':retrieved if status.get(source_key) in {'ok','ok_current_only'} else None,
            'publication_time':None,
            'retrieved_at':retrieved,
            'prediction_cutoff':retrieved if status.get(source_key) in {'ok','ok_current_only'} else None,
            'revision_time':None,
            'status':status.get(source_key),
        }
    if use_coinbase_fallback:
        source_provenance['coinbase_futures']={
            'information_origin':'Coinbase Exchange BTC-USD',
            'event_time':latest_event.isoformat(),
            'available_at':retrieved,
            'publication_time':None,
            'retrieved_at':retrieved,
            'revision_time':None,
            'prediction_cutoff':retrieved,
            'status':'ok',
        }
    scenario={'decision_time_utc':prediction_cutoff.isoformat(),'features':f,'microstructure':m,'regime':regime,'warnings':warnings,'data_quality':status,'provenance':{'event_time':latest_event.isoformat(),'available_at':retrieved,'publication_time':None,'retrieved_at':retrieved,'prediction_cutoff':retrieved,'revision_time':None,'policy':'live_acquisition_end_is_conservative_available_at; source_native_publication_and_revision_are_unknown_unless_adapter_provides_them','sources':source_provenance},'calibration':{'5m_temperature':load_temperature('5m'),'10m_temperature':load_temperature('10m'),'5m_blend_weight':w5,'10m_blend_weight':w10},'components':{'model_raw_5m':base5,'structural_5m':s5,'fused_raw_5m':raw5,'calibrated_5m':p5,'model_raw_10m':base10,'structural_10m':s10,'fused_raw_10m':raw10,'calibrated_10m':p10},'situation':situation,'policy':('bybit_fallback_model+fallback_oos_calibration' if use_bybit_fallback else ('coinbase_fallback_model+fallback_oos_calibration' if use_coinbase_fallback else 'production+structural+multi-timeframe+cross_exchange_microstructure+holdout_calibrated_blend')),'production_mode':('bybit_fallback' if use_bybit_fallback else ('coinbase_fallback' if use_coinbase_fallback else 'binance_primary'))}
    if use_bybit_fallback or use_coinbase_fallback:
        prefix='bybit' if use_bybit_fallback else 'coinbase'
        by5=json.loads((MODEL_DIR/f'{prefix}_5m.json').read_text(encoding='utf-8'))['model_version']
        by10=json.loads((MODEL_DIR/f'{prefix}_10m.json').read_text(encoding='utf-8'))['model_version']
        model_version=f'5m:{by5}|10m:{by10}'
    else:
        model_version=f'5m:{regver("5m")}|10m:{regver("10m")}'
    insert_prediction(now,target5,target10,price,p5,p10,model_version,f,scenario)
    print(json.dumps({'timestamp_jst':jst(now),'btc_price':price,'direction_5m':direction,'probabilities_5m':p5,'probabilities_10m':p10,'situation':situation,'confidence':max(p5.values()),'regime':regime,'warnings':warnings,'target_5m_jst':jst(target5),'model_5m':(json.loads((MODEL_DIR/(('bybit_5m.json' if use_bybit_fallback else 'coinbase_5m.json'))).read_text(encoding='utf-8'))['model_version'] if use_fallback else regver('5m')),'model_10m':(json.loads((MODEL_DIR/(('bybit_10m.json' if use_bybit_fallback else 'coinbase_10m.json'))).read_text(encoding='utf-8'))['model_version'] if use_fallback else regver('10m')),'calibration':scenario['calibration'],'data_quality':status},ensure_ascii=False))
if __name__=='__main__':main()
