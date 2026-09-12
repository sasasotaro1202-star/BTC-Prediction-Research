"""BTC historical research v6.

Accuracy-first research engine:
- Binance USD-M futures + spot + mark + premium
- historical funding + open-interest when available
- causal multi-timeframe / cross-asset / microstructure-proxy features
- separate 5m / 10m targets
- walk-forward OOS + embargo
- heterogeneous models + OOF ensemble
- confidence/coverage + calibration metrics
- bootstrap uncertainty vs baselines
- 2k/5k/10k OOS checkpoints
- daily incremental cache: unchanged historical chunks are never downloaded twice
- diagnostic only: production promotion remains a separate 10k-OOS gate
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
FEATURES=["ret1","ret3","ret5","ret10","ret15","ret30","accel","rv5","rv10","rv30","rangepos10","rangepos30","body","upper","lower","volratio","voltrend","tradesratio","takerimb","basis","basis_delta","mark_gap","premium","eth_ret5","sol_ret5","eth_ret10","sol_ret10","eth_btc_rel5","sol_btc_rel5","ret5_x_vol","ret10_x_vol","flow_x_vol","range_x_flow","hour_sin","hour_cos","dow_sin","dow_cos","funding","funding_delta","oi_change","oi_z"]

def req_json(url,timeout=30,retries=5):
    last=None
    for k in range(retries):
        try:
            req=urllib.request.Request(url,headers={"User-Agent":"BTC-Prediction-Research/6.0","Accept":"application/json"})
            with urllib.request.urlopen(req,timeout=timeout) as r:return json.loads(r.read())
        except Exception as e:
            last=e; time.sleep(min(8,0.8*(k+1)))
    raise RuntimeError(f"request failed: {url}: {last}")

def _cache_path(kind,symbol,day): return CACHE/f"{kind}_{symbol}_{day:%Y%m%d}.json"

def fetch_klines_chunk(symbol,start_ms,end_ms,endpoint,kind,day,limit):
    path=_cache_path(kind,symbol,day)
    if path.exists():
        try:return json.loads(path.read_text())
        except Exception:pass
    base="https://fapi.binance.com" if endpoint.startswith("/fapi") else "https://api.binance.com"
    out=[]; cur=start_ms
    while cur<end_ms:
        e=min(end_ms,cur+limit*60_000)
        q=urllib.parse.urlencode({"symbol":symbol,"interval":"1m","startTime":cur,"endTime":e,"limit":limit})
        rows=req_json(f"{base}{endpoint}?{q}")
        if not rows:break
        out.extend(rows); last=int(rows[-1][0])
        if last<cur:break
        cur=last+60_000
        if len(rows)<limit:break
    d={int(r[0]):r for r in out}; out=[d[k] for k in sorted(d)]
    path.write_text(json.dumps(out),encoding="utf-8"); return out

def fetch_klines_range(symbol,start,end,endpoint,kind,limit):
    jobs=[]; day=start.replace(hour=0,minute=0,second=0,microsecond=0)
    while day<end:
        a=max(start,day); b=min(end,day+timedelta(days=1)); jobs.append((a,b,day)); day+=timedelta(days=1)
    out=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(8,len(jobs))) as ex:
        fs={ex.submit(fetch_klines_chunk,symbol,int(a.timestamp()*1000),int(b.timestamp()*1000),endpoint,kind,d,limit):d for a,b,d in jobs}
        for f in concurrent.futures.as_completed(fs):out.extend(f.result())
    d={int(r[0]):r for r in out}; return [d[k] for k in sorted(d)]

def fetch_funding(symbol,start,end):
    path=CACHE/f"funding_{symbol}_{start:%Y%m%d}_{end:%Y%m%d}.json"
    if path.exists():
        try:return json.loads(path.read_text())
        except Exception:pass
    q=urllib.parse.urlencode({"symbol":symbol,"startTime":int(start.timestamp()*1000),"endTime":int(end.timestamp()*1000),"limit":1000})
    rows=req_json(f"https://fapi.binance.com/fapi/v1/fundingRate?{q}")
    path.write_text(json.dumps(rows),encoding="utf-8"); return rows

def fetch_oi(symbol,start,end):
    start=max(start,end-timedelta(days=30)); path=CACHE/f"oi_{symbol}_{start:%Y%m%d}_{end:%Y%m%d}.json"
    if path.exists():
        try:return json.loads(path.read_text())
        except Exception:pass
    out=[]; cur=int(start.timestamp()*1000); finish=int(end.timestamp()*1000)
    while cur<finish:
        q=urllib.parse.urlencode({"symbol":symbol,"period":"15m","startTime":cur,"endTime":min(finish,cur+500*15*60_000),"limit":500})
        try:rows=req_json(f"https://futures.binance.com/futures/data/openInterestHist?{q}")
        except Exception:break
        if not rows:break
        out.extend(rows); last=int(rows[-1]["timestamp"])
        if last<cur:break
        cur=last+1
        if len(rows)<500:break
    d={int(r["timestamp"]):r for r in out}; out=[d[k] for k in sorted(d)]
    path.write_text(json.dumps(out),encoding="utf-8"); return out

def load_market(start,end):
    jobs=[]
    for name,sym in SYMS.items():
        jobs += [(f"{name}_fut",lambda n=name,sy=sym:fetch_klines_range(sy,start,end,"/fapi/v1/klines",f"{n}_fut",1500)),(f"{name}_mark",lambda n=name,sy=sym:fetch_klines_range(sy,start,end,"/fapi/v1/markPriceKlines",f"{n}_mark",1500)),(f"{name}_premium",lambda n=name,sy=sym:fetch_klines_range(sy,start,end,"/fapi/v1/premiumIndexKlines",f"{n}_premium",1500))]
    jobs += [("btc_spot",lambda:fetch_klines_range("BTCUSDT",start,end,"/api/v3/klines","btc_spot",1000)),("funding",lambda:fetch_funding("BTCUSDT",start,end)),("oi",lambda:fetch_oi("BTCUSDT",start,end))]
    out={}
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        fs={ex.submit(fn):name for name,fn in jobs}
        for f in concurrent.futures.as_completed(fs):out[fs[f]]=f.result()
    return out

def ret(a,n):return a[-1]/a[-1-n]-1.0 if len(a)>n and a[-1-n] else 0.0
def exists(m,w):return len(m)>=len(w) and all(x in m for x in w)

def build_panel():
    # Binance daily public archives are published with a delay. Keep a
    # conservative three-day completed-data boundary so CI never asks for
    # an unpublished UTC day. This preserves the full 45-day research
    # sample while shifting its window backward by a few days.
    end=(datetime.now(timezone.utc)-timedelta(days=3)).replace(hour=0,minute=0,second=0,microsecond=0)
    start=end-timedelta(days=DAYS)
    raw=load_market(start,end)
    maps={k:{int(r[0]):r for r in v} for k,v in raw.items() if k not in ("funding","oi")}
    funding={int(r["fundingTime"]):float(r["fundingRate"]) for r in raw.get("funding",[])}; oi={int(r["timestamp"]):float(r["sumOpenInterest"]) for r in raw.get("oi",[])}
    common=sorted(set(maps["btc_fut"])&set(maps["btc_spot"])&set(maps["eth_fut"])&set(maps["sol_fut"]))
    if len(common)<MIN_TRAIN+TEST_BLOCK:raise RuntimeError(f"insufficient aligned Binance history: {len(common)}")
    fk=sorted(funding); ok=sorted(oi); rows=[]
    for i,t in enumerate(common):
        if i<50:continue
        w=common[i-50:i+1]
        def c(k):return np.array([float(maps[k][x][4]) for x in w])
        b,s,ec,sc=c("btc_fut"),c("btc_spot"),c("eth_fut"),c("sol_fut")
        bo=np.array([float(maps["btc_fut"][x][1]) for x in w]); bh=np.array([float(maps["btc_fut"][x][2]) for x in w]); bl=np.array([float(maps["btc_fut"][x][3]) for x in w]); bv=np.array([float(maps["btc_fut"][x][5]) for x in w]); bt=np.array([float(maps["btc_fut"][x][8]) for x in w]); tb=np.array([float(maps["btc_fut"][x][9]) for x in w])
        mark=c("btc_mark") if exists(maps["btc_mark"],w) else b; prem=c("btc_premium") if exists(maps["btc_premium"],w) else np.zeros_like(b); p=b[-1]
        r1,r3,r5,r10,r15,r30=[ret(b,n) for n in (1,3,5,10,15,30)]; accel=r1-r3/3
        rv5=float(np.std(np.diff(b[-6:])/b[-6:-1])); rv10=float(np.std(np.diff(b[-11:])/b[-11:-1])); rv30=float(np.std(np.diff(b[-31:])/b[-31:-1]))
        hi10,lo10=max(bh[-10:]),min(bl[-10:]); hi30,lo30=max(bh[-30:]),min(bl[-30:]); rp10=(p-lo10)/(hi10-lo10) if hi10>lo10 else .5; rp30=(p-lo30)/(hi30-lo30) if hi30>lo30 else .5
        body=(p-bo[-1])/p; upper=(bh[-1]-max(bo[-1],p))/p; lower=(min(bo[-1],p)-bl[-1])/p
        vr=float(np.mean(bv[-5:]))/max(1e-12,float(np.mean(bv[-15:-5]))); vt=float(np.mean(bv[-5:]))/max(1e-12,float(np.mean(bv[-10:]))); tr=float(np.mean(bt[-5:]))/max(1e-12,float(np.mean(bt[-15:-5]))); flow=2*float(np.sum(tb[-5:]))/max(1e-12,float(np.sum(bv[-5:])))-1
        basis=p/max(1e-12,s[-1])-1; bd=basis-(b[-2]/max(1e-12,s[-2])-1); mg=mark[-1]/p-1; pr=float(prem[-1]); er5,sr5,er10,sr10=ret(ec,5),ret(sc,5),ret(ec,10),ret(sc,10); erbtc5=er5-r5; srb5=sr5-r5
        ft=max([k for k in fk if k<=t],default=None); ot=max([k for k in ok if k<=t],default=None); funding_v=funding.get(ft,0.0) if ft else 0.0; prev_f=max([k for k in fk if k<ft],default=None) if ft else None; funding_delta=funding_v-(funding.get(prev_f,funding_v) if prev_f else funding_v); oi_v=oi.get(ot,np.nan) if ot else np.nan; prev_oi=max([k for k in ok if k<ot],default=None) if ot else None; oi_change=(oi_v/oi.get(prev_oi,oi_v)-1) if prev_oi and oi.get(prev_oi,0) else 0.0; recent_oi=[oi[k] for k in ok if k<=t][-96:]; oi_z=(oi_v-np.mean(recent_oi))/max(1e-12,np.std(recent_oi)) if recent_oi and np.isfinite(oi_v) else 0.0
        dt=datetime.fromtimestamp(t/1000,timezone.utc); hour=dt.hour+dt.minute/60; hs,hc=math.sin(2*math.pi*hour/24),math.cos(2*math.pi*hour/24); dow=dt.weekday(); ds,dc=math.sin(2*math.pi*dow/7),math.cos(2*math.pi*dow/7)
        x=[r1,r3,r5,r10,r15,r30,accel,rv5,rv10,rv30,rp10,rp30,body,upper,lower,vr,vt,tr,flow,basis,bd,mg,pr,er5,sr5,er10,sr10,erbtc5,srb5,r5*rv10,r10*rv10,flow*rv5,rp10*flow,hs,hc,ds,dc,funding_v,funding_delta,oi_change,oi_z]
        if all(math.isfinite(v) for v in x):rows.append((t,x,p))
    return rows

def labels(rows,h):
    n=len(rows)-h; X=np.asarray([rows[i][1] for i in range(n)],float); base=np.asarray([rows[i][2] for i in range(n)],float); fut=np.asarray([rows[i+h][2] for i in range(n)],float); r=(fut/base-1)*10000; y=np.where(r>NEUTRAL_BPS,"UP",np.where(r<-NEUTRAL_BPS,"DOWN","FLAT")); return X,y,np.asarray([rows[i][0] for i in range(n)]),base

def norm(p):p=np.clip(np.asarray(p,float),1e-7,1);return p/p.sum(axis=1,keepdims=True)
def metrics(y,p):
    idx={c:i for i,c in enumerate(CLASSES)}; yi=np.array([idx[v] for v in y]); p=norm(p); pred=p.argmax(1); one=np.eye(3)[yi]; hit=(pred==yi).astype(float); conf=p.max(1); ece=0.0
    for k in range(10):
        lo,hi=k/10,(k+1)/10; m=(conf>=lo)&((conf<hi) if hi<1 else (conf<=hi))
        if m.any():ece+=float(m.mean())*abs(float(hit[m].mean())-float(conf[m].mean()))
    z={"n":len(y),"accuracy":float(hit.mean()),"logloss":float(log_loss(yi,p,labels=[0,1,2])),"brier":float(np.mean(np.sum((p-one)**2,1))),"ece":float(ece)}
    for q in (.60,.70,.80,.90):
        m=conf>=q; z[f"selective_{q:.2f}"]={"coverage":float(m.mean()),"n":int(m.sum()),"accuracy":float(hit[m].mean()) if m.any() else None}
    return z

def factories():
    return {"logreg":lambda:Pipeline([("s",StandardScaler()),("m",LogisticRegression(C=.3,max_iter=1200))]),"extra":lambda:ExtraTreesClassifier(n_estimators=220,max_depth=12,min_samples_leaf=12,max_features="sqrt",random_state=42,n_jobs=-1),"rf":lambda:RandomForestClassifier(n_estimators=220,max_depth=10,min_samples_leaf=10,max_features="sqrt",random_state=42,n_jobs=-1),"hgb":lambda:HistGradientBoostingClassifier(max_iter=160,max_leaf_nodes=15,learning_rate=.04,l2_regularization=1.5,random_state=42)}
def align(m,X):
    raw=m.predict_proba(X); out=np.full((len(X),3),1e-7)
    for j,c in enumerate(m.classes_):out[:,CLASSES.index(str(c))]=raw[:,j]
    return norm(out)
def wf(X,y,ts):
    out={}
    for name,f in factories().items():
        ps,ys,st=[],[],[]
        for end in range(MIN_TRAIN,len(X),TEST_BLOCK):
            a=end+EMBARGO; b=min(a+TEST_BLOCK,len(X))
            if a>=len(X):break
            m=f();m.fit(X[:end],y[:end]);p=align(m,X[a:b]);ps.extend(p.tolist());ys.extend(y[a:b]);st.extend(ts[a:b].tolist())
        if len(ys)>=10000:out[name]={"metrics":metrics(np.asarray(ys),np.asarray(ps)),"y":ys,"p":ps,"ts":st}
    return out
def bootstrap_loss(y,p,baseline,metric="logloss",n_boot=1500,seed=42):
    rng=np.random.default_rng(seed); idx={c:i for i,c in enumerate(CLASSES)}; yi=np.array([idx[v] for v in y]); p=norm(p); b=norm(baseline)
    if metric=="logloss":a=-np.log(np.clip(p[np.arange(len(y)),yi],1e-7,1)); z=-np.log(np.clip(b[np.arange(len(y)),yi],1e-7,1))
    else:one=np.eye(3)[yi]; a=np.sum((p-one)**2,1); z=np.sum((b-one)**2,1)
    d=a-z; obs=float(d.mean()); vals=np.array([float(rng.choice(d,size=len(d),replace=True).mean()) for _ in range(n_boot)]); lo,hi=np.quantile(vals,[.025,.975]); pval=2*min(float(np.mean(vals<=0)),float(np.mean(vals>=0))); return {"difference_candidate_minus_baseline":obs,"ci95":[float(lo),float(hi)],"bootstrap_p":float(min(1,pval))}
def main():
    rows=build_panel()
    with (OUT/"aligned_panel.csv").open("w",newline="") as f:
        w=csv.writer(f);w.writerow(["timestamp","price"]+FEATURES);w.writerows([[t,p]+x for t,x,p in rows])
    report={"protocol_version":"historical-v6-microstructure-incremental-cache","source":"Binance USD-M futures + spot + mark + premium + funding + OI; ETH/SOL cross-asset","days":DAYS,"rows":len(rows),"neutral_bps":NEUTRAL_BPS,"min_train":MIN_TRAIN,"test_block":TEST_BLOCK,"embargo":EMBARGO,"features":FEATURES,"horizons":{}}
    for h,steps in TARGETS.items():
        X,y,ts,base=labels(rows,steps); r=wf(X,y,ts); freq=np.array([(y==c).sum() for c in CLASSES],float); freq/=freq.sum(); hz={"samples":len(y),"class_counts":{c:int((y==c).sum()) for c in CLASSES},"baseline":{"uniform":metrics(y,np.tile([1/3]*3,(len(y),1))),"frequency":metrics(y,np.tile(freq,(len(y),1)))},"models":{},"ensemble":{}}
        names=list(r)
        for name,o in r.items():
            hz["models"][name]=o["metrics"]
            for n in (2000,5000,10000):
                if len(o["y"])>=n:hz["models"][name][f"oos_{n}"]=metrics(np.asarray(o["y"][:n]),np.asarray(o["p"][:n]))
            with (OUT/f"oos_{h}_{name}.csv").open("w",newline="") as f:
                w=csv.writer(f);w.writerow(["timestamp","actual","p_down","p_flat","p_up"])
                for yy,pp,tt in zip(o["y"],o["p"],o["ts"]):w.writerow([int(tt),yy,*map(float,pp)])
        if names:
            n=min(len(r[k]["y"]) for k in names); ep=np.mean([np.asarray(r[k]["p"][:n]) for k in names],axis=0); ey=np.asarray(r[names[0]]["y"][:n]); hz["ensemble"]["equal_weight"]=metrics(ey,ep); uni=np.tile([1/3]*3,(n,1));freqb=np.tile(freq,(n,1))
            for base_name,b in [("uniform",uni),("frequency",freqb)]:
                hz["ensemble"][f"vs_{base_name}_logloss"]=bootstrap_loss(ey,ep,b,"logloss"); hz["ensemble"][f"vs_{base_name}_brier"]=bootstrap_loss(ey,ep,b,"brier")
            with (OUT/f"oos_{h}_ensemble.csv").open("w",newline="") as f:
                w=csv.writer(f);w.writerow(["timestamp","actual","p_down","p_flat","p_up"])
                for yy,pp,tt in zip(ey,ep,r[names[0]]["ts"][:n]):w.writerow([int(tt),yy,*map(float,pp)])
        report["horizons"][h]=hz
    report["finished_utc"]=datetime.now(timezone.utc).isoformat(); (OUT/"report.json").write_text(json.dumps(report,indent=2),encoding="utf-8"); print(json.dumps(report,indent=2))
if __name__=="__main__":main()
