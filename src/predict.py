"""BTC-only live predictor with resilient multi-venue data and conservative probability fusion."""
from __future__ import annotations
import json, math, sqlite3
from datetime import datetime, timezone, timedelta
from pathlib import Path
import joblib, numpy as np
from db import DB, init_db
from live_data_policy import validate_live_inputs
from market_data import resilient_1m_series, binance_depth, bybit_depth, binance_premium, binance_oi, binance_taker, bybit_funding, bybit_mark_price

ROOT=Path(__file__).resolve().parents[1]
MODEL_DIR=ROOT/'models'
INTERVAL=300
CLASSES=["DOWN","FLAT","UP"]
FEATURES=['ret_1m','ret_3m','ret_5m','ret_10m','acceleration','volatility_5m','volatility_10m','range_position_10m','body_1m','upper_wick_1m','lower_wick_1m','volume_ratio','volume_trend','ema_gap_5m','ema_gap_10m']

def utcnow(): return datetime.now(timezone.utc)
def jst(dt): return dt.astimezone(timezone(timedelta(hours=9))).isoformat()
def next_grid(dt,steps=1):
    ts=int(dt.timestamp()); return datetime.fromtimestamp(((ts//INTERVAL)+steps)*INTERVAL,timezone.utc)
def _ema(v,span):
    a=2/(span+1); e=float(v[0])
    for x in v[1:]: e=a*float(x)+(1-a)*e
    return e
def _ret(c,n): return c[-1]/c[-1-n]-1 if len(c)>n else 0.0
def features(rows):
    c=np.asarray([float(x[4]) for x in rows]); o=np.asarray([float(x[1]) for x in rows]); h=np.asarray([float(x[2]) for x in rows]); l=np.asarray([float(x[3]) for x in rows]); v=np.asarray([float(x[5]) for x in rows]); p=c[-1]
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
def load_model(h, source='primary'):
    """Load the model artifact for the selected production source."""
    prefix = h if source == 'primary' else f'{source}_{h}'
    p=MODEL_DIR/f'{prefix}.joblib'
    if not p.exists(): raise FileNotFoundError(f'model_missing:{source}:{h}')
    try:return joblib.load(p)
    except Exception as exc: raise RuntimeError(f'model_load_failed:{source}:{h}:{type(exc).__name__}') from exc
def model_probs(model,f):
    raw=model.predict_proba(np.array([[f[k] for k in FEATURES]]))[0]; out={c:1e-6 for c in CLASSES}
    for c,p in zip(model.classes_,raw):out[str(c)]=float(p)
    if not all(math.isfinite(v) and v>=0 for v in out.values()): raise ValueError('model_probability_nonfinite')
    s=sum(out.values())
    if not math.isfinite(s) or s<=0: raise ValueError('model_probability_invalid_sum')
    return {k:v/s for k,v in out.items()}
def regver(h):
    with sqlite3.connect(DB) as con:r=con.execute('SELECT production_version FROM model_registry WHERE horizon=?',(h,)).fetchone()
    if not r or not r[0]: raise RuntimeError(f'model_registry_missing:{h}')
    return r[0]
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
        if calibrated_version != current_version:return 0.20
        if n<400 or status not in {'accepted','rejected','insufficient_history'}: return 0.20
        if not math.isfinite(w) or not (0.0<=w<=0.45): return 0.20
        return w
    except Exception:return 0.20
def calibrate_probs(probs,h):
    t=load_temperature(h)
    if t==1.0:return probs
    p=np.clip(np.asarray([probs['DOWN'],probs['FLAT'],probs['UP']],float),1e-6,1-1e-6); p/=p.sum(); z=np.log(p)/t; z-=z.max(); q=np.exp(z); q/=q.sum()
    return {'DOWN':float(q[0]),'FLAT':float(q[1]),'UP':float(q[2])}
def fuse(base,struct,m,data_complete,horizon):
    p=np.array([base['DOWN'],base['FLAT'],base['UP']]); q=np.array([struct['DOWN'],struct['FLAT'],struct['UP']]); gap=m.get('cross_exchange_gap'); agree=max(0,1-4*abs(gap)) if gap is not None else 0.0; base_w=load_blend_weight(horizon); w=(base_w+.08*agree) if data_complete else min(base_w,.15); w=max(0.0,min(.45,w)); out=(1-w)*p+w*q; out=np.clip(out,.03,.94); out/=out.sum(); return {'DOWN':float(out[0]),'FLAT':float(out[1]),'UP':float(out[2])},float(w)
def insert_prediction(now,target5,target10,price,p5,p10,model_version,features_json,scenario):
    with sqlite3.connect(DB) as c:
        c.execute('INSERT INTO predictions(created_at_utc,target_5m,target_10m,base_price,p_up_5m,p_down_5m,p_flat_5m,p_up_10m,p_down_10m,p_flat_10m,model_version,feature_json,scenario_json) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)',(now.isoformat(),target5.isoformat(),target10.isoformat(),price,p5['UP'],p5['DOWN'],p5['FLAT'],p10['UP'],p10['DOWN'],p10['FLAT'],model_version,json.dumps(features_json),json.dumps(scenario)))
def main():
    init_db(); now=utcnow(); fut,spot,by,status=resilient_1m_series()
    use_bybit_fallback = status.get('price_feature_fallback') == 'bybit'
    use_coinbase_fallback = status.get('price_feature_fallback') == 'coinbase'
    use_fallback = use_bybit_fallback or use_coinbase_fallback
    if len(fut)<40: raise SystemExit('live_prediction_fail_closed: insufficient futures data')
    f=features(fut); price=float(fut[-1][4]); spotp=float(spot[-1][4]) if len(spot)>=40 else None
    m={}
    try:m['book_imbalance']=imbalance(binance_depth()); status['binance_depth']='ok'
    except Exception as exc:status['binance_depth']=f'error:{type(exc).__name__}'
    bybit_book=None
    try:
        bybit_book=bybit_depth(); m['bybit_book_imbalance']=imbalance(bybit_book); status['bybit_depth']='ok'
    except Exception as exc:status['bybit_depth']=f'error:{type(exc).__name__}'
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
    # fall back to the already-required order book. This closes the prior
    # failure mode where a malformed ticker response prevented the order-book
    # fallback from running.
    byp=_valid_price(by[-1][4]) if by else None
    if byp is not None:
        status['bybit_futures']='ok_current_only' if len(by) == 1 else status.get('bybit_futures','ok')
    else:
        try:
            ticker=bybit_mark_price()
            rows=ticker.get('result',{}).get('list',[]) if isinstance(ticker,dict) else []
            if rows:
                row=rows[0]
                # Bybit can occasionally return an empty lastPrice/markPrice
                # while still exposing live top-of-book prices. Use those only
                # as a bounded current-price fallback; never synthesize from
                # stale/future data.
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
            byp=_book_midprice(bybit_book)
            if byp is not None:
                status['bybit_futures']='ok_current_only'
            else:
                status['bybit_futures']='error:missing_current_price'
    if byp is not None and math.isfinite(byp) and byp>0:
        m['cross_exchange_gap']=byp/price-1
    else:
        # Explicitly record missing secondary data; do not fabricate a gap.
        m['cross_exchange_gap']=None
        status['bybit_futures']=status.get('bybit_futures','error:missing_current_price')
    try:m['funding_binance']=float(binance_premium().get('lastFundingRate')); status['binance_premium']='ok'
    except Exception as exc:status['binance_premium']=f'error:{type(exc).__name__}'
    try:m['oi']=float(binance_oi().get('openInterest')); status['binance_oi']='ok'
    except Exception as exc:status['binance_oi']=f'error:{type(exc).__name__}'
    try:
        t=binance_taker(); t=t[-1] if isinstance(t,list) and t else t; tb=float(t['takerBuyVol']); ts=float(t['takerSellVol']); m['taker_imbalance']=(tb-ts)/max(1e-12,tb+ts); status['binance_taker']='ok'
    except Exception as exc:status['binance_taker']=f'error:{type(exc).__name__}'
    try:m['funding_bybit']=float(bybit_funding().get('result',{}).get('list',[{}])[0]['fundingRate']); status['bybit_funding']='ok'
    except Exception as exc:status['bybit_funding']=f'error:{type(exc).__name__}'
    if spotp is not None:m['spot_futures_gap']=spotp/price-1
    # Binance is the production price/target venue. Bybit is optional
    # secondary information and is handled fail-safe by the policy layer.
    if use_bybit_fallback:
        # Binance is geo-blocked on GitHub-hosted runners; use the explicitly
        # trained Bybit fallback model with neutral Binance-only microstructure.
        m['book_imbalance'] = m.get('bybit_book_imbalance', 0.0)
        m['taker_imbalance'] = 0.0
        m['funding_binance'] = 0.0
    validate_live_inputs(status,fut_rows=len(fut),spot_rows=len(spot),bybit_rows=len(by),allow_bybit_fallback=use_bybit_fallback,allow_coinbase_fallback=use_coinbase_fallback)
    prediction_cutoff=utcnow()
    latest_event_ms=int(fut[-1][0]); latest_event=datetime.fromtimestamp(latest_event_ms/1000,timezone.utc)
    s5=structural(f,m)
    gap=m.get('cross_exchange_gap')
    s10=structural(f,{**m,'cross_exchange_gap':(gap*.8 if gap is not None else None)})
    model_source = 'bybit' if use_bybit_fallback else ('coinbase' if use_coinbase_fallback else 'primary')
    base5=model_probs(load_model('5m', model_source),f); base10=model_probs(load_model('10m', model_source),f)
    data_complete = byp is not None and 'bybit_book_imbalance' in m
    if use_fallback:
        raw5,raw10=base5,base10; w5=w10=0.0
    else:
        raw5,w5=fuse(base5,s5,m,data_complete,'5m'); raw10,w10=fuse(base10,s10,m,data_complete,'10m')
    # Fallback artifacts are deliberately uncalibrated until their own
    # chronological calibration is independently validated. Never apply the
    # primary Binance model's calibration to a different-source artifact.
    if use_fallback:
        p5,p10=raw5,raw10
    else:
        p5=calibrate_probs(raw5,'5m'); p10=calibrate_probs(raw10,'10m')
    target5=next_grid(now,1); target10=next_grid(now,2); direction=max(p5,key=p5.get); regime='TREND' if abs(f['trend_alignment'])>max(.0007,1.5*f['volatility_10m']) else 'RANGE'; warnings=[]
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
    for source_key in ('binance_futures','binance_depth','binance_taker','binance_premium'):
        source_provenance[source_key]={
            'information_origin':'Binance',
            'event_time':latest_event.isoformat(),
            'available_at':retrieved,
            'publication_time':None,
            'retrieved_at':retrieved,
            'revision_time':None,
            'status':status.get(source_key),
        }
    for source_key in ('bybit_futures','bybit_depth','bybit_funding'):
        source_provenance[source_key]={
            'information_origin':'Bybit',
            'event_time':latest_event.isoformat(),
            'available_at':retrieved if status.get(source_key) in {'ok','ok_current_only'} else None,
            'publication_time':None,
            'retrieved_at':retrieved,
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
            'status':'ok',
        }
    scenario={'features':f,'microstructure':m,'regime':regime,'warnings':warnings,'data_quality':status,'provenance':{'event_time':latest_event.isoformat(),'available_at':retrieved,'publication_time':None,'retrieved_at':retrieved,'prediction_cutoff':retrieved,'revision_time':None,'policy':'live_acquisition_end_is_conservative_available_at; source_native_publication_and_revision_are_unknown_unless_adapter_provides_them','sources':source_provenance},'calibration':{'5m_temperature':load_temperature('5m'),'10m_temperature':load_temperature('10m'),'5m_blend_weight':w5,'10m_blend_weight':w10},'components':{'model_raw_5m':base5,'structural_5m':s5,'fused_raw_5m':raw5,'calibrated_5m':p5,'model_raw_10m':base10,'structural_10m':s10,'fused_raw_10m':raw10,'calibrated_10m':p10},'policy':('bybit_fallback_model_only_uncalibrated' if use_bybit_fallback else ('coinbase_fallback_model_only_uncalibrated' if use_coinbase_fallback else 'production+structural+multi-timeframe+cross_exchange_microstructure+holdout_calibrated_blend')),'production_mode':('bybit_fallback' if use_bybit_fallback else ('coinbase_fallback' if use_coinbase_fallback else 'binance_primary'))}
    if use_bybit_fallback or use_coinbase_fallback:
        prefix='bybit' if use_bybit_fallback else 'coinbase'
        by5=json.loads((MODEL_DIR/f'{prefix}_5m.json').read_text(encoding='utf-8'))['model_version']
        by10=json.loads((MODEL_DIR/f'{prefix}_10m.json').read_text(encoding='utf-8'))['model_version']
        model_version=f'5m:{by5}|10m:{by10}'
    else:
        model_version=f'5m:{regver("5m")}|10m:{regver("10m")}'
    insert_prediction(now,target5,target10,price,p5,p10,model_version,f,scenario)
    print(json.dumps({'timestamp_jst':jst(now),'btc_price':price,'direction_5m':direction,'probabilities_5m':p5,'probabilities_10m':p10,'confidence':max(p5.values()),'regime':regime,'warnings':warnings,'target_5m_jst':jst(target5),'model_5m':(json.loads((MODEL_DIR/(('bybit_5m.json' if use_bybit_fallback else 'coinbase_5m.json'))).read_text(encoding='utf-8'))['model_version'] if use_fallback else regver('5m')),'model_10m':(json.loads((MODEL_DIR/(('bybit_10m.json' if use_bybit_fallback else 'coinbase_10m.json'))).read_text(encoding='utf-8'))['model_version'] if use_fallback else regver('10m')),'calibration':scenario['calibration'],'data_quality':status},ensure_ascii=False))
if __name__=='__main__':main()
