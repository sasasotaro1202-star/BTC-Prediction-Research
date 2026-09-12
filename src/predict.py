import json, math, sqlite3
from datetime import datetime, timezone, timedelta
from pathlib import Path
from urllib.request import urlopen, Request
from db import DB, init_db

INTERVAL = 300

def utcnow(): return datetime.now(timezone.utc)
def next_grid(dt, steps=1):
    ts = int(dt.timestamp())
    return datetime.fromtimestamp(((ts // INTERVAL) + steps) * INTERVAL, timezone.utc)

def fetch_klines(limit=30):
    url='https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=1m&limit=%d'%limit
    req=Request(url,headers={'User-Agent':'btc-prediction-research/1.0'})
    with urlopen(req,timeout=15) as r: return json.loads(r.read())

def features(rows):
    closes=[float(x[4]) for x in rows]; vols=[float(x[5]) for x in rows]
    def ret(n): return closes[-1]/closes[-1-n]-1 if len(closes)>n else 0.0
    return {'ret_1m':ret(1),'ret_3m':ret(3),'ret_5m':ret(5),'ret_10m':ret(10),
            'volatility_10m':(max(closes[-10:])-min(closes[-10:]))/closes[-1],
            'volume_ratio':sum(vols[-5:])/(sum(vols[-20:-5]) / 3 if sum(vols[-20:-5]) else 1)}

def stable_probs(f):
    # Scenario-first model: continuation, reversal, range. Bound the score so tiny ticks
    # cannot cause large probability jumps.
    score=2.2*f['ret_5m'] + 1.3*f['ret_10m'] + 0.8*f['ret_3m']
    score /= max(0.0015, f['volatility_10m']/2)
    score=max(-2.0,min(2.0,score))
    up=1/(1+math.exp(-score))
    flat=max(0.10,min(0.30,0.22-0.25*abs(f['ret_5m'])/max(0.001,f['volatility_10m'])))
    up=(1-flat)*up; down=1-flat-up
    return up,down,flat

def predict():
    init_db(); rows=fetch_klines(); f=features(rows); p=stable_probs(f)
    now=utcnow(); t5=next_grid(now,1); t10=next_grid(now,2); price=float(rows[-1][4])
    scenario={'continuation':max(p[0],p[1]),'range':p[2],'reversal':min(p[0],p[1])}
    with sqlite3.connect(DB) as con:
        con.execute('INSERT INTO predictions(created_at_utc,target_5m,target_10m,base_price,p_up_5m,p_down_5m,p_flat_5m,p_up_10m,p_down_10m,p_flat_10m,model_version,feature_json,scenario_json) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)',
          (now.isoformat(),t5.isoformat(),t10.isoformat(),price,*p,*p,'v1.0',json.dumps(f),json.dumps(scenario)))
    print({'price':price,'target_5m':t5.isoformat(),'target_10m':t10.isoformat(),'p':p})

if __name__=='__main__': predict()
