import json, math, sqlite3
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen, Request
import joblib, numpy as np
from db import DB, init_db

INTERVAL=300
FEATURES=[
    'ret_1m','ret_3m','ret_5m','ret_10m',
    'acceleration','volatility_5m','volatility_10m',
    'range_position_10m','body_1m','upper_wick_1m','lower_wick_1m',
    'volume_ratio','volume_trend','ema_gap_5m','ema_gap_10m'
]


def utcnow(): return datetime.now(timezone.utc)

def next_grid(dt,steps=1):
    ts=int(dt.timestamp())
    return datetime.fromtimestamp(((ts//INTERVAL)+steps)*INTERVAL,timezone.utc)

def fetch_klines(limit=60):
    url='https://api.exchange.coinbase.com/products/BTC-USD/candles?granularity=60'
    req=Request(url,headers={'User-Agent':'btc-prediction-research/3.0','Accept':'application/json'})
    with urlopen(req,timeout=15) as r: rows=json.loads(r.read())
    return sorted(rows,key=lambda x:x[0])[-limit:]

def _ema(values, span):
    a=2.0/(span+1.0); e=float(values[0])
    for v in values[1:]: e=a*float(v)+(1-a)*e
    return e

def _ret(closes,n):
    return closes[-1]/closes[-1-n]-1.0 if len(closes)>n else 0.0

def features(rows):
    closes=np.asarray([float(x[4]) for x in rows],float)
    opens=np.asarray([float(x[3]) for x in rows],float)
    highs=np.asarray([float(x[2]) for x in rows],float)
    lows=np.asarray([float(x[1]) for x in rows],float)
    vols=np.asarray([float(x[5]) for x in rows],float)
    c=closes[-1]
    r1=_ret(closes,1); r3=_ret(closes,3); r5=_ret(closes,5); r10=_ret(closes,10)
    acceleration=r1-r3/3.0
    rets=np.diff(closes[-11:])/closes[-11:-1] if len(closes)>=11 else np.array([0.0])
    volatility_10m=float(np.std(rets))
    rets5=np.diff(closes[-6:])/closes[-6:-1] if len(closes)>=6 else np.array([0.0])
    volatility_5m=float(np.std(rets5))
    hi=float(np.max(highs[-10:])); lo=float(np.min(lows[-10:]));
    range_position_10m=(c-lo)/(hi-lo) if hi>lo else 0.5
    body=(c-opens[-1])/c
    upper=(highs[-1]-max(opens[-1],c))/c
    lower=(min(opens[-1],c)-lows[-1])/c
    recent5=float(np.mean(vols[-5:])); prior10=float(np.mean(vols[-15:-5])) if len(vols)>=15 else recent5
    volume_ratio=recent5/(prior10 if prior10>0 else 1.0)
    volume_trend=recent5/(float(np.mean(vols[-10:])) if len(vols)>=10 else recent5)
    ema5=_ema(closes[-20:],5); ema10=_ema(closes[-30:],10)
    return {
        'ret_1m':float(r1),'ret_3m':float(r3),'ret_5m':float(r5),'ret_10m':float(r10),
        'acceleration':float(acceleration),'volatility_5m':volatility_5m,'volatility_10m':volatility_10m,
        'range_position_10m':float(range_position_10m),'body_1m':float(body),
        'upper_wick_1m':float(upper),'lower_wick_1m':float(lower),
        'volume_ratio':float(volume_ratio),'volume_trend':float(volume_trend),
        'ema_gap_5m':float(c/ema5-1.0),'ema_gap_10m':float(c/ema10-1.0)
    }

def structural_probs(f):
    vol=max(0.00025,f['volatility_10m'])
    trend=(2.0*f['ret_1m']+1.5*f['ret_3m']+1.2*f['ret_5m']+0.8*f['ret_10m'])/vol
    accel=f['acceleration']/vol
    ema=(f['ema_gap_5m']+f['ema_gap_10m'])/(2*vol)
    flow=math.log(max(0.25,min(4.0,f['volume_ratio'])))
    candle=f['body_1m']/vol
    score=max(-2.5,min(2.5,0.58*trend+0.16*accel+0.12*ema+0.09*flow+0.05*candle))
    directional=1/(1+math.exp(-score))
    activity=min(1.0,max(0.0,(f['volume_ratio']-0.7)/1.3))
    trend_abs=min(2.5,abs(trend))
    flat=max(0.08,min(0.32,0.27-0.07*trend_abs-0.03*activity+0.05*(1-f['volume_trend'])))
    up=(1-flat)*directional; down=(1-flat)-up
    return {'DOWN':float(down),'FLAT':float(flat),'UP':float(up)}

def registry_version(h):
    try:
        with sqlite3.connect(DB) as con:r=con.execute('SELECT production_version FROM model_registry WHERE horizon=?',(h,)).fetchone()
        return r[0] if r else 'v1.0'
    except Exception:return 'v1.0'

def load_production_model(h):
    path=Path(DB).parent/'models'/f'{h}.joblib'
    if not path.exists():return None
    try:return joblib.load(path)
    except Exception:return None

def model_probs(model,f):
    if model is None:return None
    try:
        X=np.array([[f[k] for k in FEATURES]],dtype=float); raw=model.predict_proba(X)[0]
        out={'DOWN':1e-6,'FLAT':1e-6,'UP':1e-6}
        for cls,p in zip(model.classes_,raw):out[str(cls)]=float(p)
        s=sum(out.values()); return {k:v/s for k,v in out.items()}
    except Exception:return None

def stabilize(probs,structural):
    conservative={'DOWN':0.36,'FLAT':0.28,'UP':0.36}
    p=np.array([probs['DOWN'],probs['FLAT'],probs['UP']],float)
    q=np.array([structural['DOWN'],structural['FLAT'],structural['UP']],float)
    r=np.array([conservative['DOWN'],conservative['FLAT'],conservative['UP']],float)
    out=0.60*p+0.25*q+0.15*r
    out=np.clip(out,0.05,0.90); out/=out.sum()
    return {'DOWN':float(out[0]),'FLAT':float(out[1]),'UP':float(out[2])}

def scenario_paths(f,base):
    vol=max(0.00025,f['volatility_10m'])
    trend=abs(2*f['ret_1m']+1.5*f['ret_3m']+1.2*f['ret_5m']+0.8*f['ret_10m'])/vol
    continuation=min(0.62,max(0.25,0.38+0.09*min(2.5,trend)))
    reversal=max(0.14,min(0.38,0.28-0.05*min(2.5,trend)))
    range_prob=max(0.10,1-continuation-reversal)
    return {'continuation':float(continuation),'reversal':float(reversal),'range':float(range_prob),
            'directional_bias':'UP' if base['UP']>base['DOWN'] else 'DOWN' if base['DOWN']>base['UP'] else 'FLAT'}

def predict():
    init_db(); rows=fetch_klines()
    if len(rows)<31: raise RuntimeError(f'Insufficient BTC candles: {len(rows)}')
    f=features(rows); structural=structural_probs(f); now=utcnow()
    t5=next_grid(now,1); t10=next_grid(now,2); price=float(rows[-1][4])
    p5=stabilize(model_probs(load_production_model('5m'),f) or structural,structural)
    p10=stabilize(model_probs(load_production_model('10m'),f) or structural,structural)
    v5=registry_version('5m'); v10=registry_version('10m')
    scenario={'5m':p5,'10m':p10,'structural_prior':structural,'forward_paths':scenario_paths(f,p5),
              'price_source':'coinbase_btc_usd','probability_policy':'stable_scenario_blend_v4','feature_version':'v4'}
    with sqlite3.connect(DB) as con:
        con.execute('INSERT INTO predictions(created_at_utc,target_5m,target_10m,base_price,p_up_5m,p_down_5m,p_flat_5m,p_up_10m,p_down_10m,p_flat_10m,model_version,feature_json,scenario_json) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)',
                    (now.isoformat(),t5.isoformat(),t10.isoformat(),price,p5['UP'],p5['DOWN'],p5['FLAT'],p10['UP'],p10['DOWN'],p10['FLAT'],
                     f'5m:{v5}|10m:{v10}',json.dumps(f),json.dumps(scenario)))
    print({'price':price,'target_5m':t5.isoformat(),'target_10m':t10.isoformat(),'p5':p5,'p10':p10,'model_5m':v5,'model_10m':v10,'policy':'stable_scenario_blend_v4'})

if __name__=='__main__':predict()
