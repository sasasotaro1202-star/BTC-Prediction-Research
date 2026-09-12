"""BTC historical research v5.

Goal: maximize forecast quality without gaming the metric.
- Multi-source Binance spot/futures/mark/premium data
- causal features only
- separate 5m/10m labels
- walk-forward OOS with embargo
- multiple heterogeneous models + OOF ensemble
- unconditional + selective accuracy/coverage
- paired bootstrap confidence intervals for model-vs-baseline loss
- 2k/5k/10k OOS checkpoints
- research is diagnostic only; production promotion remains a separate gate
"""
from __future__ import annotations
import concurrent.futures, csv, json, math, time, urllib.parse, urllib.request
from datetime import datetime, timezone, timedelta
from pathlib import Path
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss

OUT=Path("data/historical_research"); CACHE=OUT/"cache"
OUT.mkdir(parents=True,exist_ok=True); CACHE.mkdir(exist_ok=True)
CLASSES=["DOWN","FLAT","UP"]
DAYS=45; MIN_TRAIN=12000; TEST_BLOCK=5000; EMBARGO=10; NEUTRAL_BPS=1.0
TARGETS={"5m":5,"10m":10}; SYMS={"btc":"BTCUSDT","eth":"ETHUSDT","sol":"SOLUSDT"}
FEATURES=["ret1","ret3","ret5","ret10","accel","rv5","rv10","rangepos10","body","upper","lower","volratio","voltrend","tradesratio","takerimb","basis","basis_delta","mark_gap","premium","eth_ret5","sol_ret5","eth_ret10","sol_ret10","ret5_x_vol","ret10_x_vol","flow_x_vol","range_x_flow"]


def req_json(url,timeout=30,retries=5):
    last=None
    for k in range(retries):
        try:
            req=urllib.request.Request(url,headers={"User-Agent":"BTC-Prediction-Research/5.0","Accept":"application/json"})
            with urllib.request.urlopen(req,timeout=timeout) as r:return json.loads(r.read())
        except Exception as e:
            last=e; time.sleep(min(8,0.8*(k+1)))
    raise RuntimeError(f"request failed: {url}: {last}")


def fetch_klines(symbol,start_ms,end_ms,endpoint,cache_key,limit):
    path=CACHE/f"{cache_key}_{start_ms}_{end_ms}.json"
    if path.exists():
        try:return json.loads(path.read_text())
        except Exception:pass
    base="https://fapi.binance.com" if endpoint.startswith("/fapi") else "https://api.binance.com"
    out=[]; cur=start_ms; step=limit*60_000
    while cur<end_ms:
        e=min(end_ms,cur+step)
        q=urllib.parse.urlencode({"symbol":symbol,"interval":"1m","startTime":cur,"endTime":e,"limit":limit})
        rows=req_json(f"{base}{endpoint}?{q}")
        if not rows:break
        out.extend(rows); last=int(rows[-1][0])
        if last<cur:break
        cur=last+60_000
        if len(rows)<limit:break
    d={int(r[0]):r for r in out}; out=[d[k] for k in sorted(d)]
    path.write_text(json.dumps(out),encoding="utf-8"); return out


def load_market(start,end):
    s=int(start.timestamp()*1000); e=int(end.timestamp()*1000); jobs=[]
    for name,sym in SYMS.items():
        jobs += [(f"{name}_fut",lambda n=name,sy=sym:fetch_klines(sy,s,e,"/fapi/v1/klines",f"{n}_fut",1500)),
                 (f"{name}_mark",lambda n=name,sy=sym:fetch_klines(sy,s,e,"/fapi/v1/markPriceKlines",f"{n}_mark",1500)),
                 (f"{name}_premium",lambda n=name,sy=sym:fetch_klines(sy,s,e,"/fapi/v1/premiumIndexKlines",f"{n}_premium",1500))]
    jobs.append(("btc_spot",lambda:fetch_klines("BTCUSDT",s,e,"/api/v3/klines","btc_spot",1000)))
    out={}
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(8,len(jobs))) as ex:
        fs={ex.submit(fn):name for name,fn in jobs}
        for f in concurrent.futures.as_completed(fs):out[fs[f]]=f.result()
    return out


def ret(a,n):return a[-1]/a[-1-n]-1.0 if len(a)>n and a[-1-n]!=0 else 0.0
def exists(m,w):return len(m)>=len(w) and all(x in m for x in w)


def build_panel():
    end=datetime.now(timezone.utc).replace(second=0,microsecond=0); start=end-timedelta(days=DAYS); raw=load_market(start,end)
    maps={k:{int(r[0]):r for r in v} for k,v in raw.items()}
    common=sorted(set(maps["btc_fut"])&set(maps["btc_spot"])&set(maps["eth_fut"])&set(maps["sol_fut"]))
    if len(common)<MIN_TRAIN+TEST_BLOCK:raise RuntimeError(f"insufficient aligned Binance history: {len(common)}")
    rows=[]
    for i,t in enumerate(common):
        if i<40:continue
        w=common[i-40:i+1]
        def c(k):return np.array([float(maps[k][x][4]) for x in w])
        b,s,ec,sc=c("btc_fut"),c("btc_spot"),c("eth_fut"),c("sol_fut")
        bo=np.array([float(maps["btc_fut"][x][1]) for x in w]); bh=np.array([float(maps["btc_fut"][x][2]) for x in w]); bl=np.array([float(maps["btc_fut"][x][3]) for x in w]); bv=np.array([float(maps["btc_fut"][x][5]) for x in w]); bt=np.array([float(maps["btc_fut"][x][8]) for x in w]); tb=np.array([float(maps["btc_fut"][x][9]) for x in w])
        mark=c("btc_mark") if exists(maps["btc_mark"],w) else b; prem=c("btc_premium") if exists(maps["btc_premium"],w) else np.zeros_like(b)
        p=b[-1]; r1,r3,r5,r10=[ret(b,n) for n in (1,3,5,10)]; accel=r1-r3/3
        rv5=float(np.std(np.diff(b[-6:])/b[-6:-1])); rv10=float(np.std(np.diff(b[-11:])/b[-11:-1]))
        hi=float(max(bh[-10:])); lo=float(min(bl[-10:])); rp=(p-lo)/(hi-lo) if hi>lo else .5
        body=(p-bo[-1])/p; upper=(bh[-1]-max(bo[-1],p))/p; lower=(min(bo[-1],p)-bl[-1])/p
        vr=float(np.mean(bv[-5:]))/max(1e-12,float(np.mean(bv[-15:-5]))); vt=float(np.mean(bv[-5:]))/max(1e-12,float(np.mean(bv[-10:])))
        tr=float(np.mean(bt[-5:]))/max(1e-12,float(np.mean(bt[-15:-5]))); flow=2*float(np.sum(tb[-5:]))/max(1e-12,float(np.sum(bv[-5:])))-1
        basis=p/max(1e-12,s[-1])-1; bd=basis-(b[-2]/max(1e-12,s[-2])-1); mg=mark[-1]/p-1; pr=float(prem[-1])
        er5=ret(ec,5); sr5=ret(sc,5); er10=ret(ec,10); sr10=ret(sc,10)
        x=[r1,r3,r5,r10,accel,rv5,rv10,rp,body,upper,lower,vr,vt,tr,flow,basis,bd,mg,pr,er5,sr5,er10,sr10,r5*rv10,r10*rv10,flow*rv5,rp*flow]
        if all(math.isfinite(v) for v in x):rows.append((t,x,p))
    return rows


def labels(rows,h):
    n=len(rows)-h; X=np.asarray([rows[i][1] for i in range(n)],float); base=np.asarray([rows[i][2] for i in range(n)],float); fut=np.asarray([rows[i+h][2] for i in range(n)],float)
    r=(fut/base-1)*10000; y=np.where(r>NEUTRAL_BPS,"UP",np.where(r<-NEUTRAL_BPS,"DOWN","FLAT"))
    return X,y,np.asarray([rows[i][0] for i in range(n)]),base


def norm(p):p=np.clip(np.asarray(p,float),1e-7,1);return p/p.sum(axis=1,keepdims=True)


def metrics(y,p):
    idx={c:i for i,c in enumerate(CLASSES)}; yi=np.array([idx[v] for v in y]); p=norm(p); pred=p.argmax(1); one=np.eye(3)[yi]; hit=(pred==yi).astype(float); conf=p.max(1); ece=0
    for k in range(10):
        lo=k/10; hi=(k+1)/10; m=(conf>=lo)&((conf<hi) if hi<1 else (conf<=hi))
        if m.any():ece+=float(m.mean())*abs(float(hit[m].mean())-float(conf[m].mean()))
    z={"n":len(y),"accuracy":float(hit.mean()),"logloss":float(log_loss(yi,p,labels=[0,1,2])),"brier":float(np.mean(np.sum((p-one)**2,1))),"ece":float(ece)}
    for q in (.60,.70,.80,.90):
        m=conf>=q; z[f"selective_{q:.2f}"]={"coverage":float(m.mean()),"n":int(m.sum()),"accuracy":float(hit[m].mean()) if m.any() else None}
    return z


def factories():
    return {
        "logreg":lambda:Pipeline([("s",StandardScaler()),("m",LogisticRegression(C=.3,max_iter=1200))]),
        "extra":lambda:ExtraTreesClassifier(n_estimators=220,max_depth=12,min_samples_leaf=12,max_features="sqrt",random_state=42,n_jobs=-1),
        "rf":lambda:RandomForestClassifier(n_estimators=220,max_depth=10,min_samples_leaf=10,max_features="sqrt",random_state=42,n_jobs=-1),
        "hgb":lambda:HistGradientBoostingClassifier(max_iter=160,max_leaf_nodes=15,learning_rate=.04,l2_regularization=1.5,random_state=42)
    }


def align(m,X):
    raw=m.predict_proba(X); out=np.full((len(X),3),1e-7)
    for j,c in enumerate(m.classes_):out[:,CLASSES.index(str(c))]=raw[:,j]
    return norm(out)


def wf(X,y,ts):
    out={}
    for name,f in factories().items():
        ps=[];ys=[];st=[]
        for end in range(MIN_TRAIN,len(X),TEST_BLOCK):
            a=end+EMBARGO;b=min(a+TEST_BLOCK,len(X))
            if a>=len(X):break
            m=f();m.fit(X[:end],y[:end]);p=align(m,X[a:b]);ps.extend(p.tolist());ys.extend(y[a:b]);st.extend(ts[a:b].tolist())
        if len(ys)>=10000:out[name]={"metrics":metrics(np.asarray(ys),np.asarray(ps)),"y":ys,"p":ps,"ts":st}
    return out


def bootstrap_loss(y,p,baseline,metric="logloss",n_boot=1500,seed=42):
    rng=np.random.default_rng(seed); idx={c:i for i,c in enumerate(CLASSES)}; yi=np.array([idx[v] for v in y]); p=norm(p); b=norm(baseline)
    if metric=="logloss": a=-np.log(np.clip(p[np.arange(len(y)),yi],1e-7,1)); z=-np.log(np.clip(b[np.arange(len(y)),yi],1e-7,1))
    else:
        one=np.eye(3)[yi]; a=np.sum((p-one)**2,axis=1); z=np.sum((b-one)**2,axis=1)
    d=a-z; obs=float(d.mean()); vals=[]
    for _ in range(n_boot):vals.append(float(rng.choice(d,size=len(d),replace=True).mean()))
    lo,hi=np.quantile(vals,[.025,.975]); pval=2*min(float(np.mean(np.array(vals)<=0)),float(np.mean(np.array(vals)>=0)))
    return {"difference_candidate_minus_baseline":obs,"ci95":[float(lo),float(hi)],"bootstrap_p":float(min(1,pval))}


def main():
    rows=build_panel();
    with (OUT/"aligned_panel.csv").open("w",newline="") as f:
        w=csv.writer(f);w.writerow(["timestamp","price"]+FEATURES);w.writerows([[t,p]+x for t,x,p in rows])
    report={"protocol_version":"historical-v5-ensemble-statistical","source":"Binance USD-M futures + spot + mark + premium; ETH/SOL cross-asset","days":DAYS,"rows":len(rows),"neutral_bps":NEUTRAL_BPS,"min_train":MIN_TRAIN,"test_block":TEST_BLOCK,"embargo":EMBARGO,"features":FEATURES,"horizons":{}}
    for h,steps in TARGETS.items():
        X,y,ts,base=labels(rows,steps);r=wf(X,y,ts);freq=np.array([(y==c).sum() for c in CLASSES],float);freq/=freq.sum();hz={"samples":len(y),"class_counts":{c:int((y==c).sum()) for c in CLASSES},"baseline":{"uniform":metrics(y,np.tile([1/3]*3,(len(y),1))),"frequency":metrics(y,np.tile(freq,(len(y),1)))},"models":{},"ensemble":{}}
        model_names=list(r)
        for name,o in r.items():
            hz["models"][name]=o["metrics"]
            for n in (2000,5000,10000):
                if len(o["y"])>=n:hz["models"][name][f"oos_{n}"]=metrics(np.asarray(o["y"][:n]),np.asarray(o["p"][:n]))
            with (OUT/f"oos_{h}_{name}.csv").open("w",newline="") as f:
                w=csv.writer(f);w.writerow(["timestamp","actual","p_down","p_flat","p_up"])
                for yy,pp,tt in zip(o["y"],o["p"],o["ts"]):w.writerow([int(tt),yy,*map(float,pp)])
        if model_names:
            # Equal-weight OOF ensemble is deliberately evaluated before any adoption decision.
            n=min(len(r[k]["y"]) for k in model_names); ep=np.mean([np.asarray(r[k]["p"][:n]) for k in model_names],axis=0); ey=np.asarray(r[model_names[0]]["y"][:n]); hz["ensemble"]["equal_weight"]=metrics(ey,ep)
            uni=np.tile([1/3]*3,(n,1));freqb=np.tile(freq,(n,1));hz["ensemble"]["vs_uniform_logloss"]=bootstrap_loss(ey,ep,uni,"logloss");hz["ensemble"]["vs_uniform_brier"]=bootstrap_loss(ey,ep,uni,"brier");hz["ensemble"]["vs_frequency_logloss"]=bootstrap_loss(ey,ep,freqb,"logloss");hz["ensemble"]["vs_frequency_brier"]=bootstrap_loss(ey,ep,freqb,"brier")
            with (OUT/f"oos_{h}_ensemble.csv").open("w",newline="") as f:
                w=csv.writer(f);w.writerow(["timestamp","actual","p_down","p_flat","p_up"])
                for yy,pp,tt in zip(ey,ep,r[model_names[0]]["ts"][:n]):w.writerow([int(tt),yy,*map(float,pp)])
        report["horizons"][h]=hz
    report["finished_utc"]=datetime.now(timezone.utc).isoformat();(OUT/"report.json").write_text(json.dumps(report,indent=2),encoding="utf-8");print(json.dumps(report,indent=2))

if __name__=="__main__":main()
