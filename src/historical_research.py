import csv, json, math, time, urllib.parse, urllib.request
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

OUT = Path('data/historical_research')
OUT.mkdir(parents=True, exist_ok=True)
CLASSES = ['DOWN', 'FLAT', 'UP']
FEATURES = ['ret_1m','ret_3m','ret_5m','ret_10m','acceleration','volatility_5m','volatility_10m','range_position_10m','body_1m','upper_wick_1m','lower_wick_1m','volume_ratio','volume_trend','ema_gap_5m','ema_gap_10m']
GRANULARITY = 60
CANDLE_LIMIT = 300
DAYS = 45
MIN_TRAIN = 10000
TEST_BLOCK = 100
TARGETS = {'5m': 5, '10m': 10}
THRESHOLD = 0.00020


def fetch_chunk(start, end):
    q = urllib.parse.urlencode({'granularity': GRANULARITY, 'start': start.isoformat(), 'end': end.isoformat()})
    url = 'https://api.exchange.coinbase.com/products/BTC-USD/candles?' + q
    req = urllib.request.Request(url, headers={'User-Agent':'btc-prediction-research/historical-v1','Accept':'application/json'})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def fetch_history(days=DAYS):
    end = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    start = end - timedelta(days=days)
    rows = []
    cursor = start
    while cursor < end:
        nxt = min(cursor + timedelta(minutes=CANDLE_LIMIT), end)
        last_error = None
        for attempt in range(4):
            try:
                chunk = fetch_chunk(cursor, nxt)
                rows.extend(chunk)
                break
            except Exception as e:
                last_error = e
                time.sleep(1.0 * (attempt + 1))
        else:
            raise RuntimeError(f'Coinbase candle fetch failed at {cursor}: {last_error}')
        cursor = nxt
        time.sleep(0.12)
    # Coinbase rows are [timestamp, low, high, open, close, volume].
    by_ts = {}
    for r in rows:
        if len(r) >= 6:
            by_ts[int(r[0])] = [int(r[0]), float(r[3]), float(r[2]), float(r[1]), float(r[4]), float(r[5])]
    clean = [by_ts[k] for k in sorted(by_ts)]
    if len(clean) < MIN_TRAIN + 1000:
        raise RuntimeError(f'Only {len(clean)} candles downloaded; need more history')
    return clean


def ema(values, span):
    a = 2.0 / (span + 1.0)
    e = float(values[0])
    for v in values[1:]:
        e = a * float(v) + (1.0-a) * e
    return e


def row_features(rows, i):
    closes = np.array([r[4] for r in rows[max(0,i-35):i+1]], float)
    opens = np.array([r[3] for r in rows[max(0,i-35):i+1]], float)
    highs = np.array([r[2] for r in rows[max(0,i-35):i+1]], float)
    lows = np.array([r[3] if False else r[3] for r in rows[max(0,i-35):i+1]], float)
    lows = np.array([r[3] if False else r[3] for r in rows[max(0,i-35):i+1]], float)
    # Correct low extraction: r[3] is low after normalization below.
    lows = np.array([r[3] for r in rows[max(0,i-35):i+1]], float)
    vols = np.array([r[5] for r in rows[max(0,i-35):i+1]], float)
    c = closes[-1]
    def ret(n): return closes[-1]/closes[-1-n]-1.0
    r1,r3,r5,r10 = ret(1),ret(3),ret(5),ret(10)
    acceleration = r1-r3/3.0
    rr10 = np.diff(closes[-11:])/closes[-11:-1]
    rr5 = np.diff(closes[-6:])/closes[-6:-1]
    vol10,vol5 = float(np.std(rr10)),float(np.std(rr5))
    hi,lo = float(np.max(highs[-10:])),float(np.min(lows[-10:]))
    rp = (c-lo)/(hi-lo) if hi>lo else 0.5
    body = (c-opens[-1])/c
    upper = (highs[-1]-max(opens[-1],c))/c
    lower = (min(opens[-1],c)-lows[-1])/c
    recent = float(np.mean(vols[-5:])); prior = float(np.mean(vols[-15:-5]))
    vr = recent/(prior if prior>0 else 1.0)
    vt = recent/(float(np.mean(vols[-10:])) if np.mean(vols[-10:])>0 else recent)
    e5,e10 = ema(closes[-20:],5),ema(closes[-30:],10)
    return [r1,r3,r5,r10,acceleration,vol5,vol10,rp,body,upper,lower,vr,vt,c/e5-1.0,c/e10-1.0]


def build_dataset(rows, horizon):
    n = len(rows)-horizon
    X=[]; y=[]; ts=[]
    for i in range(35,n):
        x=row_features(rows,i)
        if not all(math.isfinite(v) for v in x): continue
        base=rows[i][4]; future=rows[i+horizon][4]; r=future/base-1.0
        label='UP' if r>THRESHOLD else 'DOWN' if r<-THRESHOLD else 'FLAT'
        X.append(x); y.append(label); ts.append(rows[i][0])
    return np.asarray(X,float),np.asarray(y),np.asarray(ts,dtype=np.int64)


def normalize(p):
    p=np.clip(np.asarray(p,float),1e-7,1-1e-7)
    return p/p.sum(axis=1,keepdims=True)


def metrics(y,p):
    idx={c:i for i,c in enumerate(CLASSES)}; yi=np.array([idx[v] for v in y]); p=normalize(p); pred=p.argmax(1)
    one=np.eye(3)[yi]; conf=p.max(1); hit=(pred==yi).astype(float); ece=0.0
    for b in range(10):
        lo=b/10; hi=(b+1)/10; m=(conf>=lo)&((conf<hi) if hi<1 else (conf<=hi))
        if m.any(): ece += float(m.mean())*abs(float(hit[m].mean())-float(conf[m].mean()))
    return {'n':int(len(y)),'accuracy':float((pred==yi).mean()),'logloss':float(log_loss(yi,p,labels=[0,1,2])),'brier':float(np.mean(np.sum((p-one)**2,axis=1))),'ece':float(ece)}


def model_factories():
    return {
      'logreg_c0.1': lambda: Pipeline([('scale',StandardScaler()),('model',LogisticRegression(C=.1,max_iter=3000))]),
      'logreg_c1': lambda: Pipeline([('scale',StandardScaler()),('model',LogisticRegression(C=1,max_iter=3000))]),
      'logreg_c10': lambda: Pipeline([('scale',StandardScaler()),('model',LogisticRegression(C=10,max_iter=3000))]),
      'rf_500': lambda: RandomForestClassifier(n_estimators=500,max_depth=7,min_samples_leaf=8,max_features='sqrt',random_state=42,n_jobs=-1),
      'hgb': lambda: HistGradientBoostingClassifier(max_iter=250,max_leaf_nodes=15,learning_rate=.04,l2_regularization=1.0,random_state=42)
    }


def align(model,X):
    raw=model.predict_proba(X); out=np.full((len(X),3),1e-7)
    for j,c in enumerate(model.classes_): out[:,CLASSES.index(str(c))]=raw[:,j]
    return normalize(out)


def walk_forward(X,y,ts):
    results={}
    for name,factory in model_factories().items():
        probs=[]; ys=[]; stamps=[]
        for end in range(MIN_TRAIN,len(X),TEST_BLOCK):
            test_end=min(end+TEST_BLOCK,len(X))
            model=factory(); train_y=y[:end]
            if len(set(train_y.tolist()))<3: continue
            model.fit(X[:end],train_y)
            p=align(model,X[end:test_end])
            probs.extend(p.tolist()); ys.extend(y[end:test_end].tolist()); stamps.extend(ts[end:test_end].tolist())
        if len(ys)>=10000:
            results[name]={'metrics':metrics(np.array(ys),np.asarray(probs)),'y':ys,'p':probs,'ts':stamps}
    return results


def baseline(y):
    p=np.tile(np.array([1/3,1/3,1/3]),(len(y),1))
    return metrics(y,p)


def main():
    started=datetime.now(timezone.utc)
    rows=fetch_history()
    raw_path=OUT/'btc_usd_1m_raw.csv'
    with raw_path.open('w',newline='') as f:
        w=csv.writer(f); w.writerow(['timestamp','open','high','low','close','volume']); w.writerows(rows)
    report={'started_utc':started.isoformat(),'finished_utc':None,'source':'Coinbase Exchange BTC-USD 1m','days':DAYS,'candles':len(rows),'threshold':THRESHOLD,'min_train':MIN_TRAIN,'test_block':TEST_BLOCK,'horizons':{}}
    for h,steps in TARGETS.items():
        X,y,ts=build_dataset(rows,steps)
        r=walk_forward(X,y,ts)
        report['horizons'][h]={'samples':int(len(y)),'class_counts':{c:int((y==c).sum()) for c in CLASSES},'baseline':baseline(y),'models':{k:{'metrics':v['metrics']} for k,v in r.items()}}
        # Persist full OOS predictions for later statistical testing/replay.
        out=OUT/f'oos_{h}.csv'
        with out.open('w',newline='') as f:
            w=csv.writer(f); w.writerow(['timestamp','actual','p_down','p_flat','p_up'])
            if r:
                for yy,pp,stamp in zip(r[min(r,key=lambda k:-r[k]['metrics']['n'])]['y'],r[min(r,key=lambda k:-r[k]['metrics']['n'])]['p'],r[min(r,key=lambda k:-r[k]['metrics']['n'])]['ts']): w.writerow([int(stamp),yy,float(pp[0]),float(pp[1]),float(pp[2])])
    report['finished_utc']=datetime.now(timezone.utc).isoformat()
    (OUT/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))

if __name__=='__main__': main()
