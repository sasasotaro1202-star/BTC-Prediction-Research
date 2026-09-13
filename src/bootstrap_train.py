"""BTC-only robust historical bootstrap trainer.

The bootstrap is intentionally best-effort: production inference must not be
blocked by a temporary historical-data outage. Existing production models are
never replaced unless a chronological holdout gate is passed.
"""
from __future__ import annotations
import json, math, sqlite3, time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen
import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from db import DB, init_db

ROOT=Path(__file__).resolve().parents[1]
MODEL_DIR=ROOT/'models'; DATA_DIR=ROOT/'data'/'historical_research'; CACHE=DATA_DIR/'btc_bootstrap_1m.json'; STATUS=DATA_DIR/'bootstrap_status.json'
CLASSES=['DOWN','FLAT','UP']; THRESHOLD=.00020; UA='BTC-Prediction-Research/bootstrap/3.0'
FEATURES=['ret_1m','ret_3m','ret_5m','ret_10m','acceleration','volatility_5m','volatility_10m','range_position_10m','body_1m','upper_wick_1m','lower_wick_1m','volume_ratio','volume_trend','ema_gap_5m','ema_gap_10m']

def write_status(obj):
    DATA_DIR.mkdir(parents=True,exist_ok=True)
    STATUS.write_text(json.dumps({**obj,'updated_at_utc':datetime.now(timezone.utc).isoformat()},indent=2),encoding='utf-8')

def get(url, attempts=4, timeout=20):
    last=None
    for i in range(attempts):
        try:
            req=Request(url,headers={'User-Agent':UA,'Accept':'application/json'})
            with urlopen(req,timeout=timeout) as r:return json.loads(r.read())
        except Exception as e:
            last=e
            if i+1<attempts:time.sleep(min(4.0,0.75*(i+1)))
    raise last

def bybit_page(end_ms=None,limit=1000):
    p={'category':'linear','symbol':'BTCUSDT','interval':'1','limit':min(1000,limit)}
    if end_ms is not None:p['end']=int(end_ms)
    return get('https://api.bybit.com/v5/market/kline?'+urlencode(p))

def binance_page(end_ms=None,limit=1500):
    p={'symbol':'BTCUSDT','interval':'1m','limit':limit}
    if end_ms is not None:p['endTime']=end_ms
    return get('https://fapi.binance.com/fapi/v1/klines?'+urlencode(p))

def coinbase_page(start_s,end_s):
    p={'granularity':60,'start':datetime.fromtimestamp(start_s,tz=timezone.utc).isoformat(),'end':datetime.fromtimestamp(end_s,tz=timezone.utc).isoformat()}
    return get('https://api.exchange.coinbase.com/products/BTC-USD/candles?'+urlencode(p))

def fetch_bybit(target):
    rows=[]; end=None
    for _ in range(math.ceil(target/1000)+5):
        page=bybit_page(end,1000); raw=page.get('result',{}).get('list',[]) if isinstance(page,dict) else []
        if not raw:break
        for r in raw:
            if len(r)>=6: rows.append([int(r[0]),float(r[1]),float(r[2]),float(r[3]),float(r[4]),float(r[5])])
        oldest=min(int(r[0]) for r in raw); new_end=oldest-1
        if end is not None and new_end>=end:break
        end=new_end
        if len(rows)>=target:break
        time.sleep(.05)
    rows=sorted({r[0]:r for r in rows}.values(),key=lambda r:r[0]); now_ms=int(time.time()*1000)
    rows=[r for r in rows if r[0]+60000<=now_ms]
    return rows[-target:]

def fetch_binance(target):
    rows=[];end=None
    for _ in range(math.ceil(target/1500)+5):
        page=binance_page(end,1500)
        if not page:break
        rows.extend([[int(r[0]),float(r[1]),float(r[2]),float(r[3]),float(r[4]),float(r[5])] for r in page])
        oldest=min(int(r[0]) for r in page);end=oldest-1
        if len(page)<1500:break
    rows=sorted({r[0]:r for r in rows}.values(),key=lambda r:r[0]); return rows[-target:]

def fetch_coinbase(target):
    rows=[];end=int(time.time());window=300*60
    for _ in range(math.ceil(target/300)+8):
        start=max(0,end-window);page=coinbase_page(start,end)
        if not page:break
        rows.extend([[int(r[0])*1000,float(r[3]),float(r[2]),float(r[1]),float(r[4]),float(r[5])] for r in page]);end=start-1
        if len(rows)>=target:break
        time.sleep(.15)
    rows=sorted({r[0]:r for r in rows}.values(),key=lambda r:r[0]); return rows[-target:]

def fetch_history(target=30000):
    errors=[]; best=[]; best_name='none'
    for name,fn in (('bybit_futures',fetch_bybit),('binance_futures',fetch_binance),('coinbase',fetch_coinbase)):
        try:
            rows=fn(target)
            if len(rows)>len(best):best,best_name=rows,name
            if len(rows)>=target:return rows,name
            errors.append(f'{name}:only_{len(rows)}_rows')
        except Exception as e:errors.append(f'{name}:{type(e).__name}:{e}')
    if best:
        return best,best_name
    raise RuntimeError(f'no historical BTC source available: {"; ".join(errors)}')

def ema(v,span):
    a=2/(span+1);e=float(v[0])
    for x in v[1:]:e=a*float(x)+(1-a)*e
    return e

def make_features(rows):
    c=np.asarray([r[4] for r in rows],float);o=np.asarray([r[1] for r in rows],float);h=np.asarray([r[2] for r in rows],float);l=np.asarray([r[3] for r in rows],float);v=np.asarray([r[5] for r in rows],float);p=c[-1]
    ret=lambda n:c[-1]/c[-1-n]-1
    r1,r3,r5,r10=[ret(n) for n in (1,3,5,10)];a=r1-r3/3;rv5=float(np.std(np.diff(c[-6:])/c[-6:-1]));rv10=float(np.std(np.diff(c[-11:])/c[-11:-1]))
    hi,lo=max(h[-10:]),min(l[-10:]);rp=(p-lo)/(hi-lo) if hi>lo else .5;body=(p-o[-1])/p;up=(h[-1]-max(o[-1],p))/p;low=(min(o[-1],p)-l[-1])/p
    old=float(np.mean(v[-15:-5]));rvol=float(np.mean(v[-5:]))/max(1e-12,old);vt=float(np.mean(v[-5:]))/max(1e-12,float(np.mean(v[-10:])))
    return [r1,r3,r5,r10,a,rv5,rv10,rp,body,up,low,rvol,vt,p/ema(c[-20:],5)-1,p/ema(c[-30:],10)-1]

def build_dataset(rows,horizon):
    X=[];y=[]
    for i in range(30,len(rows)-horizon):
        r=rows[i+horizon][4]/rows[i][4]-1;y.append('UP' if r>THRESHOLD else 'DOWN' if r<-THRESHOLD else 'FLAT');X.append(make_features(rows[:i+1]))
    return np.asarray(X,float),np.asarray(y)

def norm(p):
    p=np.clip(np.asarray(p,float),1e-8,1);return p/p.sum(axis=1,keepdims=True)

def metrics(y,p):
    idx={c:i for i,c in enumerate(CLASSES)};yi=np.asarray([idx[z] for z in y]);p=norm(p);one=np.eye(3)[yi]
    return {'accuracy':float(np.mean(p.argmax(1)==yi)),'logloss':float(log_loss(yi,p,labels=[0,1,2])),'brier':float(np.mean(np.sum((p-one)**2,axis=1)))}

def calibrate(raw,y):
    raw=norm(raw)
    if len(y)<200 or len(set(y))<3:return raw,1.0
    yi=np.asarray([{c:i for i,c in enumerate(CLASSES)}[z] for z in y]);best=(log_loss(yi,raw,labels=[0,1,2]),1.0);logits=np.log(raw)
    for t in np.linspace(.75,2.25,61):
        z=logits/t;z-=z.max(1,keepdims=True);q=np.exp(z);q/=q.sum(1,keepdims=True);ll=log_loss(yi,q,labels=[0,1,2])
        if ll<best[0]:best=(ll,float(t))
    z=logits/best[1];z-=z.max(1,keepdims=True);q=np.exp(z);q/=q.sum(1,keepdims=True);return q,best[1]

def train_one(X,y):
    n=len(y);a=int(n*.65);b=int(n*.82);Xtr,Xcal,Xte=X[:a],X[a:b],X[b:];ytr,ycal,yte=y[:a],y[a:b],y[b:]
    counts=np.array([np.mean(ytr==c) for c in CLASSES]);base=metrics(yte,np.tile(counts,(len(yte),1)));cands=[('logreg',Pipeline([('scale',StandardScaler()),('model',LogisticRegression(C=.5,max_iter=3000))])),('rf',RandomForestClassifier(n_estimators=300,max_depth=7,min_samples_leaf=12,max_features='sqrt',random_state=42,n_jobs=-1)),('hgb',HistGradientBoostingClassifier(max_iter=220,max_leaf_nodes=15,learning_rate=.04,l2_regularization=1.5,random_state=42))]
    results=[]
    for name,model in cands:
        model.fit(Xtr,ytr);_,t=calibrate(model.predict_proba(Xcal),ycal);p=norm(model.predict_proba(Xte))
        if t!=1:
            z=np.log(p)/t;z-=z.max(1,keepdims=True);p=np.exp(z);p/=p.sum(1,keepdims=True)
        s=metrics(yte,p);results.append((s['logloss'],s['brier'],-s['accuracy'],name,model,t,s))
    results.sort(key=lambda z:z[:3]);return results[0],base,len(yte)

def publish(h,best,base,n):
    ll,br,negacc,name,model,t,s=best
    if not(s['logloss']<base['logloss']-.01 and s['brier']<base['brier']-.005):return False,{'status':'holdout_rejected','model':name,'candidate':s,'baseline':base,'holdout_n':n}
    MODEL_DIR.mkdir(parents=True,exist_ok=True);joblib.dump(model,MODEL_DIR/f'{h}.joblib');version=f'bootstrap.{name}.v3';meta={'model_version':version,'horizon':h,'classes':list(model.classes_),'features':FEATURES,'artifact':f'{h}.joblib','candidate':False,'bootstrap':True,'holdout_n':n,'holdout_metrics':s,'baseline_metrics':base,'temperature':float(t),'trained_at_utc':datetime.now(timezone.utc).isoformat()};(MODEL_DIR/f'{h}.json').write_text(json.dumps(meta,indent=2),encoding='utf-8');init_db()
    with sqlite3.connect(DB) as con:con.execute('INSERT INTO model_registry(horizon,production_version,updated_at_utc) VALUES(?,?,?) ON CONFLICT(horizon) DO UPDATE SET production_version=excluded.production_version,updated_at_utc=excluded.updated_at_utc',(h,version,datetime.now(timezone.utc).isoformat()))
    return True,meta

def main():
    MODEL_DIR.mkdir(parents=True,exist_ok=True);DATA_DIR.mkdir(parents=True,exist_ok=True);init_db()
    if all((MODEL_DIR/f'{h}.joblib').exists() and (MODEL_DIR/f'{h}.json').exists() for h in ('5m','10m')):
        write_status({'status':'skipped_existing_production_models','required_models_present':True})
        print('BTC bootstrap skipped: production models already exist');return
    try:
        rows,source=fetch_history(30000)
    except Exception as e:
        write_status({'status':'historical_sources_unavailable','error':f'{type(e).__name__}: {e}','production_models_present':False,'inference_policy':'continue_with_structural_fallback'})
        print(f'BTC bootstrap non-fatal: historical sources unavailable: {e}');return
    CACHE.write_text(json.dumps({'source':source,'rows':rows,'created_at_utc':datetime.now(timezone.utc).isoformat()},separators=(',',':')),encoding='utf-8')
    write_status({'status':'history_fetched','source':source,'rows':len(rows)})
    # Do not fail the live cycle merely because a public archive currently
    # exposes fewer rows than the research target. Existing production models
    # remain untouched; prediction.py has an independently tested structural
    # fallback. A bootstrap model is published only after the chronological
    # holdout gate below passes.
    if len(rows)<10000:
        write_status({'status':'insufficient_history_for_bootstrap','source':source,'rows':len(rows),'minimum_rows':10000,'inference_policy':'continue_with_structural_fallback'})
        print(f'BTC bootstrap non-fatal: got {len(rows)} rows; need >=10000 for safe retraining');return
    for h in (5,10):
        X,y=build_dataset(rows,h)
        if len(y)<7000:
            write_status({'status':'insufficient_training_rows','horizon':h,'rows':len(y),'minimum_rows':7000})
            print(f'BTC bootstrap non-fatal: dataset too small for {h}m: {len(y)}');continue
        best,base,n=train_one(X,y);ok,detail=publish(f'{h}m',best,base,n);print(json.dumps({'horizon':h,'source':source,'rows':len(y),'published':ok,'detail':detail},ensure_ascii=False))
        if ok:write_status({'status':'completed','last_published_horizon':f'{h}m','source':source,'rows':len(y),'detail':detail})
if __name__=='__main__':main()
