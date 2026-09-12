import json, math, sqlite3
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen, Request
import joblib, numpy as np
from db import DB, init_db

INTERVAL=300
FEATURES=['ret_1m','ret_3m','ret_5m','ret_10m','volatility_10m','volume_ratio']


def utcnow():
    return datetime.now(timezone.utc)


def next_grid(dt,steps=1):
    ts=int(dt.timestamp())
    return datetime.fromtimestamp(((ts//INTERVAL)+steps)*INTERVAL,timezone.utc)


def fetch_klines(limit=30):
    # Coinbase Exchange is used in GitHub Actions because Binance's public API
    # returns HTTP 451 from the GitHub-hosted runner region.
    url='https://api.exchange.coinbase.com/products/BTC-USD/candles?granularity=60'
    req=Request(url,headers={'User-Agent':'btc-prediction-research/1.3','Accept':'application/json'})
    with urlopen(req,timeout=15) as r:
        rows=json.loads(r.read())
    rows=sorted(rows,key=lambda x:x[0])[-limit:]
    # Coinbase candle format: [time, low, high, open, close, volume]
    return rows


def features(rows):
    closes=[float(x[4]) for x in rows]
    vols=[float(x[5]) for x in rows]
    def ret(n):
        return closes[-1]/closes[-1-n]-1 if len(closes)>n else 0.0
    return {
        'ret_1m':ret(1),
        'ret_3m':ret(3),
        'ret_5m':ret(5),
        'ret_10m':ret(10),
        'volatility_10m':(max(closes[-10:])-min(closes[-10:]))/closes[-1],
        'volume_ratio':sum(vols[-5:])/(sum(vols[-20:-5])/3 if sum(vols[-20:-5]) else 1)
    }


def stable_probs(f):
    score=2.2*f['ret_5m']+1.3*f['ret_10m']+0.8*f['ret_3m']
    score/=max(0.0015,f['volatility_10m']/2)
    score=max(-2.0,min(2.0,score))
    up=1/(1+math.exp(-score))
    flat=max(0.10,min(0.30,0.22-0.25*abs(f['ret_5m'])/max(0.001,f['volatility_10m'])))
    up=(1-flat)*up
    down=1-flat-up
    return {'DOWN':down,'FLAT':flat,'UP':up}


def registry_version(h):
    try:
        with sqlite3.connect(DB) as con:
            r=con.execute('SELECT production_version FROM model_registry WHERE horizon=?',(h,)).fetchone()
        return r[0] if r else 'v1.0'
    except Exception:
        return 'v1.0'


def model_probs(h,f):
    path=Path(DB).parent/'models'/f'{h}.joblib'
    if not path.exists():
        return None
    try:
        model=joblib.load(path)
        X=np.array([[f[k] for k in FEATURES]],dtype=float)
        raw=model.predict_proba(X)[0]
        out={'DOWN':1e-6,'FLAT':1e-6,'UP':1e-6}
        for cls,p in zip(model.classes_,raw):
            out[str(cls)]=float(p)
        s=sum(out.values())
        return {k:v/s for k,v in out.items()}
    except Exception:
        return None


def predict():
    init_db()
    rows=fetch_klines()
    if len(rows)<21:
        raise RuntimeError(f'Insufficient BTC candles: {len(rows)}')
    f=features(rows)
    fallback=stable_probs(f)
    now=utcnow()
    t5=next_grid(now,1)
    t10=next_grid(now,2)
    price=float(rows[-1][4])
    p5=model_probs('5m',f) or fallback
    p10=model_probs('10m',f) or fallback
    v5=registry_version('5m')
    v10=registry_version('10m')
    scenario={'5m':p5,'10m':p10,'fallback_model':fallback,'price_source':'coinbase_btc_usd'}
    with sqlite3.connect(DB) as con:
        con.execute('INSERT INTO predictions(created_at_utc,target_5m,target_10m,base_price,p_up_5m,p_down_5m,p_flat_5m,p_up_10m,p_down_10m,p_flat_10m,model_version,feature_json,scenario_json) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)',(
            now.isoformat(),t5.isoformat(),t10.isoformat(),price,
            p5['UP'],p5['DOWN'],p5['FLAT'],
            p10['UP'],p10['DOWN'],p10['FLAT'],
            f'5m:{v5}|10m:{v10}',json.dumps(f),json.dumps(scenario)))
    print({'price':price,'target_5m':t5.isoformat(),'target_10m':t10.isoformat(),'p5':p5,'p10':p10,'model_5m':v5,'model_10m':v10})


if __name__=='__main__':
    predict()
