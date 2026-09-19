"""Train a separately gated Coinbase 1m fallback model for production continuity."""
from __future__ import annotations
import json, time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen
import joblib
from bootstrap_train import build_dataset, train_one, FEATURES

ROOT=Path(__file__).resolve().parents[1]
MODEL_DIR=ROOT/'models'
TARGET=30_000
UA='BTC-Prediction-Research/coinbase-fallback/1.0'

def get(url):
    last=None
    for i in range(4):
        try:
            req=Request(url,headers={'User-Agent':UA,'Accept':'application/json'})
            with urlopen(req,timeout=30) as r: return json.loads(r.read())
        except Exception as exc:
            last=exc
            if i<3: time.sleep(1.0*(i+1))
    raise last

def fetch_coinbase(target=TARGET):
    rows=[]
    end=int(time.time())
    for _ in range((target+299)//300+8):
        start=end-300*60
        url='https://api.exchange.coinbase.com/products/BTC-USD/candles?'+urlencode({'granularity':60,'start':start,'end':end})
        raw=get(url)
        if not raw: break
        for r in raw:
            if len(r)>=6:
                rows.append([int(r[0])*1000,float(r[3]),float(r[2]),float(r[1]),float(r[4]),float(r[5])])
        oldest=min(int(r[0]) for r in raw)
        if oldest>=end: break
        end=oldest-1
        if len(rows)>=target: break
        time.sleep(0.15)
    now=int(time.time()*1000)
    rows=[r for r in rows if r[0]+60_000<=now]
    rows=sorted({r[0]:r for r in rows}.values(),key=lambda r:r[0])
    if len(rows)<10_000: raise RuntimeError(f'coinbase_fallback_insufficient_history:{len(rows)}')
    return rows[-target:]

def main():
    rows=fetch_coinbase()
    MODEL_DIR.mkdir(parents=True,exist_ok=True)
    out=[]
    for horizon in ('5m','10m'):
        X,y=build_dataset(rows,int(horizon[:-1]))
        if len(y)<1500 or len(set(y))<3: raise RuntimeError(f'coinbase_fallback_insufficient_dataset:{horizon}:{len(y)}')
        best,baseline,holdout_n,validation_results=train_one(X,y,purge_gap=int(horizon[:-1]))
        _,_,_,name,model,_,score=best
        if score['logloss']>=baseline['logloss']-0.005: raise RuntimeError(f'coinbase_fallback_holdout_rejected:{horizon}')
        artifact=MODEL_DIR/f'coinbase_{horizon}.joblib'
        metadata=MODEL_DIR/f'coinbase_{horizon}.json'
        joblib.dump(model,artifact)
        meta={'model_version':f'coinbase_fallback.{name}.v1','source':'Coinbase Exchange BTC-USD 1m candles','horizon':horizon,'classes':list(model.classes_),'features':FEATURES,'artifact':artifact.name,'selection_method':'chronological_17pct_validation','holdout_n':holdout_n,'holdout_metrics':score,'baseline_metrics':baseline,'validation_metrics':[{'model':r[3],'logloss':r[0],'brier':r[1],'accuracy':-r[2]} for r in validation_results],'calibration':'not_calibrated_until_fallback_live_settlement','trained_at_utc':datetime.now(timezone.utc).isoformat()}
        metadata.write_text(json.dumps(meta,indent=2),encoding='utf-8')
        out.append(meta)
    print(json.dumps({'ok':True,'rows':len(rows),'models':out},indent=2))

if __name__=='__main__': main()
