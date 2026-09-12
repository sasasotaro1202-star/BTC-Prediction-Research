import json, math, sqlite3
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen, Request
import joblib, numpy as np
from db import DB, init_db

INTERVAL=300
FEATURES=['ret_1m','ret_3m','ret_5m','ret_10m','volatility_10m','volume_ratio']

def utcnow(): return datetime.now(timezone.utc)
def next_grid(dt,steps=1):
    ts=int(dt.timestamp())
    return datetime.fromtimestamp(((ts//INTERVAL)+steps)*INTERVAL,timezone.utc)

def fetch_klines(limit=30):
    url='https://api.exchange.coinbase.com/products/BTC-USD/candles?granularity=60'
    req=Request(url,headers={'User-Agent':'btc-prediction-research/2.1','Accept':'application/json'})
    with urlopen(req,timeout=15) as r: rows=json.loads(r.read())
    return sorted(rows,key=lambda x:x[0])[-limit:]

def features(rows):
    closes=[float(x[4]) for x in rows]; vols=[float(x[5]) for x in rows]
    def ret(n): return closes[-1]/closes[-1-n]-1 if len(closes)>n else 0.0
    window=closes[-10:]
    vol=(max(window)-min(window))/closes[-1] if window else 0.0
    return {'ret_1m':ret(1),'ret_3m':ret(3),'ret_5m':ret(5),'ret_10m':ret(10),'volatility_10m':vol,'volume_ratio':sum(vols[-5:])/(sum(vols[-20:-5])/3 if sum(vols[-20:-5]) else 1)}

def structural_probs(f):
    vol=max(0.0015,f['volatility_10m']/2)
    trend=(2.0*f['ret_1m']+1.5*f['ret_3m']+1.2*f['ret_5m']+0.8*f['ret_10m'])/vol
    acceleration=(f['ret_1m']-f['ret_3m']/3)/vol
    flow=math.log(max(0.25,min(4.0,f['volume_ratio'])))
    score=max(-2.5,min(2.5,0.78*trend+0.22*acceleration+0.25*flow))
    directional=1/(1+math.exp(-score))
    flat=max(0.10,min(0.32,0.23-0.18*abs(trend)+0.08*max(0.0,1.0-f['volume_ratio'])))
    flat=max(0.08,min(0.32,flat))
    up=(1-flat)*directional
    down=(1-flat)-up
    return {'DOWN':down,'FLAT':flat,'UP':up}

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
    # Fixed pre-declared blend: small feature changes should not create large probability jumps.
    conservative={'DOWN':0.36,'FLAT':0.28,'UP':0.36}
    p=np.array([probs['DOWN'],probs['FLAT'],probs['UP']],float)
    q=np.array([structural['DOWN'],structural['FLAT'],structural['UP']],float)
    r=np.array([conservative['DOWN'],conservative['FLAT'],conservative['UP']],float)
    out=0.60*p+0.25*q+0.15*r
    out=np.clip(out,0.05,0.90); out/=out.sum()
    return {'DOWN':float(out[0]),'FLAT':float(out[1]),'UP':float(out[2])}

def scenario_paths(f,base):
    vol=max(0.0015,f['volatility_10m'])
    trend=abs(2*f['ret_1m']+1.5*f['ret_3m']+1.2*f['ret_5m']+0.8*f['ret_10m'])/vol
    continuation=min(0.60,max(0.25,0.38+0.08*min(2.5,trend)))
    reversal=max(0.15,min(0.38,0.27-0.05*min(2.5,trend)))
    range_prob=max(0.10,1-continuation-reversal)
    return {'continuation':float(continuation),'reversal':float(reversal),'range':float(range_prob),'directional_bias':'UP' if base['UP']>base['DOWN'] else 'DOWN' if base['DOWN']>base['UP'] else 'FLAT'}

def predict():
    init_db(); rows=fetch_klines()
    if len(rows)<21:raise RuntimeError(f'Insufficient BTC candles: {len(rows)}')
    f=features(rows); structural=structural_probs(f); now=utcnow(); t5=next_grid(now,1); t10=next_grid(now,2); price=float(rows[-1][4])
    p5=stabilize(model_probs(load_production_model('5m'),f) or structural,structural)
    p10=stabilize(model_probs(load_production_model('10m'),f) or structural,structural)
    v5=registry_version('5m'); v10=registry_version('10m')
    scenario={'5m':p5,'10m':p10,'structural_prior':structural,'forward_paths':scenario_paths(f,p5),'price_source':'coinbase_btc_usd','probability_policy':'stable_scenario_blend_v3'}
    with sqlite3.connect(DB) as con:
        con.execute('INSERT INTO predictions(created_at_utc,target_5m,target_10m,base_price,p_up_5m,p_down_5m,p_flat_5m,p_up_10m,p_down_10m,p_flat_10m,model_version,feature_json,scenario_json) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)',(now.isoformat(),t5.isoformat(),t10.isoformat(),price,p5['UP'],p5['DOWN'],p5['FLAT'],p10['UP'],p10['DOWN'],p10['FLAT'],f'5m:{v5}|10m:{v10}',json.dumps(f),json.dumps(scenario)))
    print({'price':price,'target_5m':t5.isoformat(),'target_10m':t10.isoformat(),'p5':p5,'p10':p10,'model_5m':v5,'model_10m':v10,'policy':'stable_scenario_blend_v3'})

if __name__=='__main__':predict()
