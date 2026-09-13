"""BTC-only historical bootstrap trainer.

Builds production 5m/10m classifiers from historical 1m OHLCV when the
live database does not yet contain enough settled predictions for walk-forward
model comparison. Fails closed: a model is published only when an unseen
holdout beats a simple class-frequency baseline on both LogLoss and Brier.
"""
from __future__ import annotations
import json, math, time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss

ROOT=Path(__file__).resolve().parents[1]
MODEL_DIR=ROOT/'models'; DATA_DIR=ROOT/'data'/'historical_research'; CACHE=DATA_DIR/'btc_bootstrap_1m.json'
CLASSES=['DOWN','FLAT','UP']
FEATURES=['ret_1m','ret_3m','ret_5m','ret_10m','acceleration','volatility_5m','volatility_10m','range_position_10m','body_1m','upper_wick_1m','lower_wick_1m','volume_ratio','volume_trend','ema_gap_5m','ema_gap_10m']
THRESHOLD=.00020
UA='BTC-Prediction-Research/bootstrap/1.0'

def get(url, attempts=3, timeout=15):
    last=None
    for i in range(attempts):
        try:
            req=Request(url,headers={'User-Agent':UA,'Accept':'application/json'})
            with urlopen(req,timeout=timeout) as r:return json.loads(r.read())
        except Exception as e:
            last=e
            if i+1<attempts: time.sleep(1.2*(i+1))
    raise last

def binance_page(end_ms=None, limit=1500):
    p={'symbol':'BTCUSDT','interval':'1m','limit':limit}
    if end_ms is not None:p['endTime']=end_ms
    return get('https://fapi.binance.com/fapi/v1/klines?'+urlencode(p))

def kraken_page(since=None):
    p={'pair':'XBTUSD','interval':1}
    if since is not None:p['since']=since
    return get('https://api.kraken.com/0/public/OHLC?'+urlencode(p))

def fetch_history(minutes=12000):
    target=max(3000,minutes); rows=[]
    try:
        end=None
        while len(rows)<target:
            page=binance_page(end,1500)
            if not page:break
            converted=[[int(r[0]),float(r[1]),float(r[2]),float(r[3]),float(r[4]),float(r[5])] for r in page]
            rows.extend(converted); oldest=min(r[0] for r in converted); end=oldest-1
            if len(converted)<1500:break
        rows=sorted({r[0]:r for r in rows}.values(),key=lambda r:r[0])
        if len(rows)>=target:return rows[-target:],'binance_futures'
    except Exception: rows=[]
    rows=[]; since=int(time.time())-target*60
    try:
        for _ in range(max(8,math.ceil(target/650)+3)):
            payload=kraken_page(since); result=payload.get('result',{}); key=next((k for k in result if k!='last'),None); page=result.get(key,[]) if key else []
            if not page:break
            converted=[[int(r[0])*1000,float(r[1]),float(r[2]),float(r[3]),float(r[4]),float(r[6])] for r in page]
            rows.extend(converted); last=max(r[0] for r in converted); since=last//1000+60
            if len(rows)>=target:break
            time.sleep(.15)
        rows=sorted({r[0]:r for r in rows}.values(),key=lambda r:r[0])
        if len(rows)>=target:return rows[-target:],'kraken'
    except Exception: pass
    raise RuntimeError(f'bootstrap history unavailable: got {len(rows)} rows, need {target}')

def ema(v,span):
    a=2/(span+1); e=float(v[0])
    for x in v[1:]:e=a*float(x)+(1-a)*e
    return e

def ret(c,n):return c[-1]/c[-1-n]-1 if len(c)>n else 0.0

def make_features(rows):
    c=np.asarray([r[4] for r in rows],float);o=np.asarray([r[1] for r in rows],float);h=np.asarray([r[2] for r in rows],float);l=np.asarray([r[3] for r in rows],float);v=np.asarray([r[5] for r in rows],float);p=c[-1]
    r1,r3,r5,r10=[ret(c,n) for n in (1,3,5,10)]; a=r1-r3/3
    rv5=float(np.std(np.diff(c[-6:])/c[-6:-1]));rv10=float(np.std(np.diff(c[-11:])/c[-11:-1]))
    hi,lo=max(h[-10:]),min(l[-10:]);rp=(p-lo)/(hi-lo) if hi>lo else .5
    body=(p-o[-1])/p;up=(h[-1]-max(o[-1],p))/p;low=(min(o[-1],p)-l[-1])/p
    rvol=float(np.mean(v[-5:]))/max(1e-12,float(np.mean(v[-15:-5]))) if np.mean(v[-15:-5]) else 1.
    vt=float(np.mean(v[-5:]))/max(1e-12,float(np.mean(v[-10:]))) if np.mean(v[-10:]) else 1.
    return [r1,r3,r5,r10,a,rv5,rv10,rp,body,up,low,rvol,vt,p/ema(c[-20:],5)-1,p/ema(c[-30:],10)-1]

def build_dataset(rows,horizon):
    X=[];y=[]
    for i in range(30,len(rows)-horizon):
        f=make_features(rows[:i+1]); r=rows[i+horizon][4]/rows[i][4]-1
        y.append('UP' if r>THRESHOLD else 'DOWN' if r<-THRESHOLD else 'FLAT'); X.append(f)
    return np.asarray(X,float),np.asarray(y)

def norm(p):
    p=np.clip(np.asarray(p,float),1e-7,1-1e-7);return p/p.sum(axis=1,keepdims=True)

def score(y,p):
    idx={c:i for i,c in enumerate(CLASSES)};yi=np.array([idx[v] for v in y]);p=norm(p);pred=p.argmax(1);one=np.eye(3)[yi]
    return {'accuracy':float(np.mean(pred==yi)),'logloss':float(log_loss(yi,p,labels=[0,1,2])),'brier':float(np.mean(np.sum((p-one)**2,axis=1)))}

def calibrate(p,y):
    if len(y)<100 or len(set(y))<3:return norm(p),1.
    idx={c:i for i,c in enumerate(CLASSES)};yi=np.array([idx[v] for v in y]);raw=norm(p);logits=np.log(raw);best_t=1.;best=float('inf')
    for t in np.linspace(.7,2.5,73):
        z=logits/t;z-=z.max(1,keepdims=True);q=np.exp(z);q/=q.sum(1,keepdims=True);ll=log_loss(yi,q,labels=[0,1,2])
        if ll<best:best=float(ll);best_t=float(t)
    z=logits/best_t;z-=z.max(1,keepdims=True);q=np.exp(z);q/=q.sum(1,keepdims=True)
    if log_loss(yi,q,labels=[0,1,2])>=log_loss(yi,raw,labels=[0,1,2])-.001:return raw,1.
    return q,best_t

def train_one(X,y):
    n=len(y);a=int(n*.65);b=int(n*.82);Xtr,Xcal,Xte=X[:a],X[a:b],X[b:];ytr,ycal,yte=y[:a],y[a:b],y[b:]
    candidates={
      'bootstrap_logreg':lambda:Pipeline([('scale',StandardScaler()),('model',LogisticRegression(C=1,max_iter=3000))]),
      'bootstrap_rf':lambda:RandomForestClassifier(n_estimators=400,max_depth=8,min_samples_leaf=10,max_features='sqrt',random_state=42,n_jobs=-1),
      'bootstrap_hgb':lambda:HistGradientBoostingClassifier(max_iter=250,max_leaf_nodes=15,learning_rate=.04,l2_regularization=1.0,random_state=42)}
    counts={c:float(np.mean(ytr==c)) for c in CLASSES};baseline=np.tile([counts['DOWN'],counts['FLAT'],counts['UP']],(len(yte),1));base_score=score(yte,baseline);results=[]
    for name,factory in candidates.items():
        model=factory();model.fit(Xtr,ytr);cal_p,t=calibrate(model.predict_proba(Xcal),ycal);raw=norm(model.predict_proba(Xte))
        if t!=1.:z=np.log(raw)/t;z-=z.max(1,keepdims=True);raw=np.exp(z);raw/=raw.sum(1,keepdims=True)
        s=score(yte,raw);results.append((s['logloss'],s['brier'],s['accuracy'],name,model,t,s))
    results.sort(key=lambda z:(z[0],z[1],-z[2]));return results[0],base_score,len(yte)

def publish(h,best,base_score,n_test):
    ll,br,acc,name,model,t,s=best
    if not (s['logloss']<base_score['logloss']-.01 and s['brier']<base_score['brier']-.005):
        return False,{'status':'holdout_rejected','model':name,'test_n':n_test,'candidate':s,'baseline':base_score}
    MODEL_DIR.mkdir(parents=True,exist_ok=True);joblib.dump(model,MODEL_DIR/f'{h}.joblib')
    meta={'model_version':f'bootstrap.{name}','horizon':h,'classes':list(model.classes_),'features':FEATURES,'artifact':f'{h}.joblib','candidate':False,'bootstrap':True,'holdout_n':n_test,'holdout_metrics':s,'baseline_metrics':base_score,'temperature':float(t),'trained_at_utc':datetime.now(timezone.utc).isoformat()}
    (MODEL_DIR/f'{h}.json').write_text(json.dumps(meta,indent=2),encoding='utf-8');return True,meta

def main():
    MODEL_DIR.mkdir(parents=True,exist_ok=True);DATA_DIR.mkdir(parents=True,exist_ok=True)
    if all((MODEL_DIR/f'{h}.joblib').exists() and (MODEL_DIR/f'{h}.json').exists() for h in ('5m','10m')):
        print('BTC bootstrap skipped: production models already exist');return
    rows,source=fetch_history(12000);CACHE.write_text(json.dumps({'source':source,'rows':rows,'created_at_utc':datetime.now(timezone.utc).isoformat()}),encoding='utf-8')
    for h in (5,10):
        X,y=build_dataset(rows,h)
        if len(y)<2000:raise RuntimeError(f'bootstrap dataset too small for {h}m: {len(y)}')
        best,base,n=train_one(X,y);ok,detail=publish(f'{h}m',best,base,n)
        print(json.dumps({'horizon':h,'source':source,'rows':len(y),'published':ok,'detail':detail},ensure_ascii=False))
        if not ok:raise RuntimeError(f'No safe bootstrap model passed holdout for {h}m: {detail}')

if __name__=='__main__':main()
