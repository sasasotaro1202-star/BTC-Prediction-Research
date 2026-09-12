"""BTC historical research v3: Binance multi-source, leakage-safe OOS.

Uses Binance USD-M futures + spot 1m klines. Futures candles also provide
trade count and taker-buy volume; spot/futures basis, cross-asset context,
range/volatility, and flow features are derived causally. No live-model
adoption happens here; this script only produces auditable OOS evidence.
"""
from __future__ import annotations
import csv,json,math,time,urllib.parse,urllib.request,zipfile,io
from datetime import datetime,timezone,timedelta
from pathlib import Path
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier,HistGradientBoostingClassifier,RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss

OUT=Path("data/historical_research"); OUT.mkdir(parents=True,exist_ok=True)
CLASSES=["DOWN","FLAT","UP"]
DAYS=45; MIN_TRAIN=12000; TEST_BLOCK=5000; EMBARGO=10; NEUTRAL_BPS=1.0
TARGETS={"5m":5,"10m":10}
SYMS={"btc":"BTCUSDT","eth":"ETHUSDT","sol":"SOLUSDT"}
FEATURES=[
"ret1","ret3","ret5","ret10","accel","rv5","rv10","rangepos10",
"body","upper","lower","volratio","voltrend","tradesratio","takerimb",
"basis","basis_delta","mark_gap","premium","eth_ret5","sol_ret5","eth_ret10","sol_ret10",
"ret5_x_vol","ret10_x_vol","flow_x_vol","range_x_flow"
]

def req_json(url,timeout=30,retries=4):
    last=None
    for k in range(retries):
        try:
            req=urllib.request.Request(url,headers={"User-Agent":"BTC-Prediction-Research/3.0","Accept":"application/json"})
            with urllib.request.urlopen(req,timeout=timeout) as r:return json.loads(r.read())
        except Exception as e:
            last=e; time.sleep(1.5*(k+1))
    raise RuntimeError(f"request failed: {url}: {last}")

def fetch_binance(symbol,start_ms,end_ms,base="https://fapi.binance.com",limit=1500):
    out=[]; cur=start_ms; step=limit*60_000
    while cur<end_ms:
        e=min(end_ms,cur+step)
        q=urllib.parse.urlencode({"symbol":symbol,"interval":"1m","startTime":cur,"endTime":e,"limit":limit})
        rows=req_json(f"{base}/fapi/v1/klines?{q}")
        if not rows: break
        out.extend(rows); last=int(rows[-1][0]);
        if last<cur: break
        cur=last+60_000
        if len(rows)<limit: break
        time.sleep(.03)
    d={int(r[0]):r for r in out}; return [d[k] for k in sorted(d)]

def fetch_spot(symbol,start_ms,end_ms,limit=1000):
    out=[]; cur=start_ms; step=limit*60_000
    while cur<end_ms:
        e=min(end_ms,cur+step)
        q=urllib.parse.urlencode({"symbol":symbol,"interval":"1m","startTime":cur,"endTime":e,"limit":limit})
        rows=req_json(f"https://api.binance.com/api/v3/klines?{q}")
        if not rows: break
        out.extend(rows); last=int(rows[-1][0]); cur=last+60_000
        if len(rows)<limit: break
        time.sleep(.03)
    d={int(r[0]):r for r in out}; return [d[k] for k in sorted(d)]

def arr(rows,idx): return np.asarray([float(r[idx]) for r in rows],float)
def ret(a,n): return a[-1]/a[-1-n]-1.0 if len(a)>n and a[-1-n]!=0 else 0.0
def ema(a,n):
    x=float(a[0]); alpha=2/(n+1)
    for v in a[1:]: x=alpha*float(v)+(1-alpha)*x
    return x

def build_panel():
    end=datetime.now(timezone.utc).replace(second=0,microsecond=0); start=end-timedelta(days=DAYS)
    s=int(start.timestamp()*1000); e=int(end.timestamp()*1000)
    raw={k:fetch_binance(v,s,e) for k,v in SYMS.items()}
    spot=fetch_spot("BTCUSDT",s,e)
    maps={k:{int(r[0]):r for r in v} for k,v in raw.items()}; smap={int(r[0]):r for r in spot}
    common=sorted(set(maps["btc"]) & set(smap) & set(maps["eth"]) & set(maps["sol"]))
    if len(common)<MIN_TRAIN+15000: raise RuntimeError(f"insufficient aligned Binance history: {len(common)}")
    rows=[]
    for i,t in enumerate(common):
        if i<40: continue
        rb=[maps[k][x] for k in SYMS for x in []] # keeps symbol order explicit
        b=maps["btc"]; eth=maps["eth"]; sol=maps["sol"]
        window=common[max(0,i-40):i+1]
        bc=np.array([float(b[x][4]) for x in window]); bo=np.array([float(b[x][1]) for x in window]); bh=np.array([float(b[x][2]) for x in window]); bl=np.array([float(b[x][3]) for x in window]); bv=np.array([float(b[x][5]) for x in window]); bt=np.array([float(b[x][8]) for x in window]); tb=np.array([float(b[x][9]) for x in window])
        sc=np.array([float(smap[x][4]) for x in window]); ec=np.array([float(eth[x][4]) for x in window]); lc=np.array([float(sol[x][4]) for x in window])
        c=bc[-1]; r1,r3,r5,r10=[ret(bc,n) for n in (1,3,5,10)]; accel=r1-r3/3
        rr5=np.diff(bc[-6:])/bc[-6:-1]; rr10=np.diff(bc[-11:])/bc[-11:-1]; rv5=float(np.std(rr5)); rv10=float(np.std(rr10))
        hi=float(max(bh[-10:])); lo=float(min(bl[-10:])); rp=(c-lo)/(hi-lo) if hi>lo else .5
        body=(c-bo[-1])/c; upper=(bh[-1]-max(bo[-1],c))/c; lower=(min(bo[-1],c)-bl[-1])/c
        vr=float(np.mean(bv[-5:]))/max(1e-12,float(np.mean(bv[-15:-5]))); vt=float(np.mean(bv[-5:]))/max(1e-12,float(np.mean(bv[-10:])))
        tr=float(np.mean(bt[-5:]))/max(1e-12,float(np.mean(bt[-15:-5])))
        flow=(2*float(np.sum(tb[-5:]))/max(1e-12,float(np.sum(bv[-5:])))-1.0)
        basis=c/max(1e-12,float(sc[-1]))-1.0; basis_prev=bc[-2]/max(1e-12,float(sc[-2]))-1.0; bd=basis-basis_prev
        mark_gap=0.0; premium=0.0 # optional derivatives microprice proxies; kept explicit
        er5=ret(ec,5); sr5=ret(lc,5); er10=ret(ec,10); sr10=ret(lc,10)
        x=[r1,r3,r5,r10,accel,rv5,rv10,rp,body,upper,lower,vr,vt,tr,flow,basis,bd,mark_gap,premium,er5,sr5,er10,sr10,r5*rv10,r10*rv10,flow*rv5,rp*flow]
        if all(math.isfinite(v) for v in x): rows.append((t,x,c))
    return rows

def labels(rows,h):
    y=[]; X=[]; ts=[]; base=[]
    for i in range(len(rows)-h):
        t,x,p=rows[i]; future=rows[i+h][2]; r=(future/p-1)*10000
        y.append("UP" if r>NEUTRAL_BPS else "DOWN" if r<-NEUTRAL_BPS else "FLAT"); X.append(x); ts.append(t); base.append(p)
    return np.asarray(X,float),np.asarray(y),np.asarray(ts),np.asarray(base,float)

def norm(p):
    p=np.clip(np.asarray(p,float),1e-7,1); return p/p.sum(axis=1,keepdims=True)
def metrics(y,p):
    idx={c:i for i,c in enumerate(CLASSES)}; yi=np.array([idx[v] for v in y]); p=norm(p); pred=p.argmax(1); one=np.eye(3)[yi]; conf=p.max(1); hit=(pred==yi).astype(float); ece=0
    for b in range(10):
        lo=b/10; hi=(b+1)/10; m=(conf>=lo)&((conf<hi) if hi<1 else (conf<=hi))
        if m.any(): ece+=float(m.mean())*abs(float(hit[m].mean())-float(conf[m].mean()))
    return {"n":len(y),"accuracy":float(hit.mean()),"logloss":float(log_loss(yi,p,labels=[0,1,2])),"brier":float(np.mean(np.sum((p-one)**2,1))),"ece":float(ece)}

def factories():
    return {
      "logreg":lambda:Pipeline([("s",StandardScaler()),("m",LogisticRegression(C=.3,max_iter=1500))]),
      "extra":lambda:ExtraTreesClassifier(n_estimators=300,max_depth=12,min_samples_leaf=12,max_features="sqrt",random_state=42,n_jobs=-1),
      "rf":lambda:RandomForestClassifier(n_estimators=250,max_depth=9,min_samples_leaf=12,max_features="sqrt",random_state=42,n_jobs=-1),
      "hgb":lambda:HistGradientBoostingClassifier(max_iter=160,max_leaf_nodes=15,learning_rate=.04,l2_regularization=1.5,random_state=42)
    }

def align(m,X):
    raw=m.predict_proba(X); out=np.full((len(X),3),1e-7)
    for j,c in enumerate(m.classes_): out[:,CLASSES.index(str(c))]=raw[:,j]
    return norm(out)

def wf(X,y,ts):
    out={}
    for name,factory in factories().items():
        ps=[]; ys=[]; stamps=[]
        for end in range(MIN_TRAIN,len(X),TEST_BLOCK):
            a=end+EMBARGO; b=min(a+TEST_BLOCK,len(X))
            if a>=len(X): break
            m=factory(); m.fit(X[:end],y[:end]); p=align(m,X[a:b]); ps.extend(p.tolist()); ys.extend(y[a:b]); stamps.extend(ts[a:b].tolist())
        if len(ys)>=10000: out[name]={"metrics":metrics(np.asarray(ys),np.asarray(ps)),"y":ys,"p":ps,"ts":stamps}
    return out

def main():
    started=datetime.now(timezone.utc); rows=build_panel()
    with (OUT/"aligned_panel.csv").open("w",newline="") as f:
        w=csv.writer(f); w.writerow(["timestamp","price"]+FEATURES); w.writerows([[t,p]+x for t,x,p in rows])
    report={"protocol_version":"historical-v3-binance-multisource","source":"Binance USD-M futures + Binance spot; ETH/SOL cross-asset","days":DAYS,"rows":len(rows),"neutral_bps":NEUTRAL_BPS,"min_train":MIN_TRAIN,"test_block":TEST_BLOCK,"embargo":EMBARGO,"features":FEATURES,"horizons":{}}
    for h,steps in TARGETS.items():
        X,y,ts,base=labels(rows,steps); r=wf(X,y,ts)
        freq=np.array([(y==c).sum() for c in CLASSES],float); freq/=freq.sum()
        hz={"samples":len(y),"class_counts":{c:int((y==c).sum()) for c in CLASSES},"baseline":{"uniform":metrics(y,np.tile([1/3]*3,(len(y),1))),"frequency":metrics(y,np.tile(freq,(len(y),1)))},"models":{}}
        for name,o in r.items():
            hz["models"][name]=o["metrics"]
            for n in (2000,5000,10000):
                if len(o["y"])>=n: hz["models"][name][f"oos_{n}"]=metrics(np.asarray(o["y"][:n]),np.asarray(o["p"][:n]))
            with (OUT/f"oos_{h}_{name}.csv").open("w",newline="") as f:
                w=csv.writer(f); w.writerow(["timestamp","actual","p_down","p_flat","p_up"])
                for yy,pp,tt in zip(o["y"],o["p"],o["ts"]): w.writerow([int(tt),yy,*map(float,pp)])
        report["horizons"][h]=hz
    report["finished_utc"]=datetime.now(timezone.utc).isoformat(); (OUT/"report.json").write_text(json.dumps(report,indent=2),encoding="utf-8"); print(json.dumps(report,indent=2))
if __name__=="__main__": main()
