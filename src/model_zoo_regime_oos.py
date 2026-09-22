"""Research-only causal model-zoo + regime-aware soft routing OOS evaluator.

The router chooses model weights from completed prior OOS blocks only.
It never uses current test-block labels or the final holdout for selection.
Production artifacts are never modified.
"""
from __future__ import annotations
import json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
SRC_DIR=Path(__file__).resolve().parent
ROOT_DIR=SRC_DIR.parent
for _path in (ROOT_DIR,SRC_DIR):
    if str(_path) not in sys.path: sys.path.insert(0,str(_path))
from model_compare import HORIZONS, metrics, _temperature, apply_temperature
from binance_history import binance_archive_rows
from bootstrap_train import make_features, THRESHOLD
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier
OUT=ROOT_DIR/"data/historical_research/model_zoo_regime_oos.json"
CLASSES=("DOWN","FLAT","UP"); MIN_TRAIN=2000; TEST_BLOCK=300; MAX_ROWS=30000; MAX_OOS_BLOCKS=24
TREND_INDEX=2
VOL_INDEX=6
MIN_HISTORY_PER_REGIME=180; FLOOR=0.10; MAX_WEIGHT=0.55; SHRINKAGE=0.30
META_MIN_ROWS=900; META_MAX_ROWS=6000
META_REGIMES=("TREND_UP|LOW_VOL","TREND_UP|HIGH_VOL","TREND_DOWN|LOW_VOL","TREND_DOWN|HIGH_VOL","RANGE|LOW_VOL","RANGE|HIGH_VOL")

def factories():
    return {
        "logreg":lambda:Pipeline([("scale",StandardScaler()),("model",LogisticRegression(C=0.25,max_iter=3000))]),
        "extra_trees":lambda:ExtraTreesClassifier(n_estimators=320,max_depth=12,min_samples_leaf=10,max_features="sqrt",random_state=42,n_jobs=-1),
        "hgb":lambda:HistGradientBoostingClassifier(max_iter=220,max_leaf_nodes=15,learning_rate=0.04,l2_regularization=1.5,random_state=42),
        "lightgbm":lambda:LGBMClassifier(objective="multiclass",num_class=3,n_estimators=260,num_leaves=15,learning_rate=0.035,min_child_samples=24,reg_lambda=2.0,subsample=0.9,colsample_bytree=0.9,random_state=42,n_jobs=-1,verbosity=-1),
        **({"xgboost":lambda:XGBClassifier(objective="multi:softprob",num_class=3,n_estimators=220,max_depth=4,learning_rate=0.035,min_child_weight=10,subsample=0.9,colsample_bytree=0.9,reg_lambda=2.0,reg_alpha=0.05,eval_metric="mlogloss",random_state=42,n_jobs=-1,verbosity=0)} if XGBClassifier is not None else {}),
    }

def _regime_thresholds(train):
    x=np.asarray([r["x"] for r in train],dtype=float)
    return {"trend_q":float(np.quantile(np.abs(x[:,TREND_INDEX]),0.55)),"vol_q":float(np.quantile(x[:,VOL_INDEX],0.70))}

def regime_key(row,thresholds):
    x=np.asarray(row["x"],dtype=float); trend=float(x[TREND_INDEX]); vol=float(x[VOL_INDEX])
    trend_cut=max(abs(float(thresholds["trend_q"])),1e-8)
    direction="RANGE" if abs(trend)<=trend_cut else ("TREND_UP" if trend>0 else "TREND_DOWN")
    vol_cut=max(float(thresholds["vol_q"]),1e-10)
    return f"{direction}|{'HIGH_VOL' if vol>vol_cut else 'LOW_VOL'}"

def _aligned(model,rows):
    raw=np.asarray(model.predict_proba(np.asarray([r["x"] for r in rows],dtype=float)),dtype=float)
    out=np.full((len(rows),3),1e-7,dtype=float); idx={c:i for i,c in enumerate(CLASSES)}
    for j,cls in enumerate(model.classes_):
        if str(cls) in idx: out[:,idx[str(cls)]]=raw[:,j]
    out=np.clip(out,1e-7,1.0); return out/out.sum(axis=1,keepdims=True)

def _fit_calibrated(train,test,factory):
    split=int(len(train)*0.75)
    if split<300 or len(train)-split<100:return None
    cal=factory(); cal.fit(np.asarray([r["x"] for r in train[:split]],dtype=float),np.asarray([r["y"] for r in train[:split]]))
    temp=_temperature(_aligned(cal,train[split:]),[r["y"] for r in train[split:]])
    model=factory(); model.fit(np.asarray([r["x"] for r in train],dtype=float),np.asarray([r["y"] for r in train]))
    return apply_temperature(_aligned(model,test),temp)

def _bounded_weights(losses,briers,eces):
    ll=np.asarray(losses,float); br=np.asarray(briers,float); ec=np.asarray(eces,float); n=len(ll)
    if n==0 or not all(np.all(np.isfinite(v)) for v in (ll,br,ec)): return np.full(n,1.0/max(1,n))
    score=0.65*(ll/max(float(np.median(ll)),1e-12))+0.20*(br/max(float(np.median(br)),1e-12))+0.15*(ec/max(float(np.median(ec)),1e-12))
    raw=np.exp(-(score-score.min())/0.35); raw/=raw.sum()
    w=(1.0-SHRINKAGE)*raw+SHRINKAGE*(1.0/n); w=np.clip(w,FLOOR,MAX_WEIGHT)
    for _ in range(50):
        gap=1.0-float(w.sum())
        if abs(gap)<1e-12:break
        room=(MAX_WEIGHT-w) if gap>0 else (w-FLOOR); active=room>1e-12
        if not np.any(active):break
        w[active]+=gap*room[active]/room[active].sum()
    w=np.clip(w,FLOOR,MAX_WEIGHT); return w/w.sum()

def _history_weights(history,keys,model_names):
    if not history:
        equal=np.full(len(model_names),1.0/len(model_names)); return equal,{k:equal.copy() for k in keys}
    regime_rows=defaultdict(list)
    for rec in history: regime_rows[rec["regime"]].append(rec)
    def calc(records):
        if not records:return None
        arr={k:np.asarray([r[k] for r in records],dtype=float) for k in ("logloss","brier","ece")}
        return _bounded_weights([float(v) for v in arr["logloss"].mean(axis=0)],[float(v) for v in arr["brier"].mean(axis=0)],[float(v) for v in arr["ece"].mean(axis=0)])
    global_w=calc(history)
    routed={}
    for key in keys:
        local=regime_rows.get(key,[])
        if len(local)<MIN_HISTORY_PER_REGIME:routed[key]=global_w.copy()
        else:routed[key]=(0.55*calc(local)+0.45*global_w); routed[key]/=routed[key].sum()
    return global_w,routed

def _mix(parts,weights):
    out=np.zeros_like(parts[0],dtype=float)
    for p,w in zip(parts,weights): out+=float(w)*p
    out=np.clip(out,1e-7,1.0); return out/out.sum(axis=1,keepdims=True)

def _meta_features(parts, keys):
    """Represent only prior-OOS component predictions + current regime."""
    probs=np.hstack([np.asarray(p,dtype=float) for p in parts])
    grouped=probs.reshape(len(keys),-1,3)
    entropy=-np.sum(grouped*np.log(np.clip(grouped,1e-7,1.0)),axis=2)
    regime=np.zeros((len(keys),len(META_REGIMES)),dtype=float)
    for i,key in enumerate(keys):
        if key in META_REGIMES:
            regime[i,META_REGIMES.index(key)]=1.0
    return np.hstack([probs,entropy,regime])


def _fit_meta_router(meta_history):
    """Fit the meta-router only from completed prior OOS observations."""
    if len(meta_history)<META_MIN_ROWS:
        return None
    data=meta_history[-META_MAX_ROWS:]
    X=np.asarray([r["x"] for r in data],dtype=float)
    y=np.asarray([r["y"] for r in data])
    if len(set(y.tolist()))<3:
        return None
    split=max(int(len(y)*0.75),len(y)-600)
    if split<500 or len(y)-split<150:
        return None
    probe=Pipeline([("scale",StandardScaler()),("model",LogisticRegression(C=0.20,max_iter=3000))])
    probe.fit(X[:split],y[:split])
    temp=_temperature(_aligned(probe,[{"x":z} for z in X[split:]]),y[split:].tolist())
    model=Pipeline([("scale",StandardScaler()),("model",LogisticRegression(C=0.20,max_iter=3000))])
    model.fit(X,y)
    raw=model.predict_proba(X[:1]) if len(X) else None
    classes=getattr(model,"classes_",[])
    if len(classes)!=3:
        return None
    def predict(Xnew):
        pp=np.asarray(model.predict_proba(Xnew),dtype=float)
        out=np.full((len(Xnew),3),1e-7,dtype=float)
        for j,cls in enumerate(model.classes_):
            if str(cls) in CLASSES:
                out[:,CLASSES.index(str(cls))]=pp[:,j]
        return apply_temperature(out,temp)
    return predict


def _meta_predict(meta_history, parts, keys):
    if not meta_history:
        return None
    fn=_fit_meta_router(meta_history)
    if fn is None:
        return None
    return fn(_meta_features(parts,keys))

def _historical_rows(horizon):
    """Build runtime-compatible causal rows from closed 1-minute BTC history."""
    raw = binance_archive_rows(MAX_ROWS)
    source = "binance_vision_archive"
    steps=int(horizon[:-1])
    out=[]
    for i in range(30, len(raw)-steps):
        x=make_features(raw[:i+1])
        future_return=raw[i+steps][4]/raw[i][4]-1.0
        y="UP" if future_return>THRESHOLD else "DOWN" if future_return<-THRESHOLD else "FLAT"
        if all(np.isfinite(v) for v in x):
            out.append({"id":str(int(raw[i][0])),"created":str(int(raw[i][0])),"x":np.asarray(x,dtype=float),"y":y,"source":source})
    return out

def evaluate(horizon):
    try:
        rows=_historical_rows(horizon)
    except Exception as exc:
        return {"status":"DEFERRED","n":0,"reason":"historical_source_unavailable","error":f"{type(exc).__name__}: {exc}"}
    if len(rows)>MAX_ROWS: rows=rows[-MAX_ROWS:]
    if len(rows)<MIN_TRAIN+TEST_BLOCK+200:return {"status":"DEFERRED","n":len(rows),"reason":"insufficient_rows"}
    split=int(len(rows)*0.80); development=rows[:split]; holdout=rows[split:]
    if len(development)<MIN_TRAIN+TEST_BLOCK or len(holdout)<100:return {"status":"DEFERRED","n":len(rows),"reason":"insufficient_split"}
    mf=factories(); names=list(mf); history=[]; meta_history=[]; blocks=[]
    all_endpoints=list(range(MIN_TRAIN,len(development),TEST_BLOCK))
    if len(all_endpoints)>MAX_OOS_BLOCKS:
        endpoints=sorted(set(int(v) for v in np.linspace(all_endpoints[0],all_endpoints[-1],MAX_OOS_BLOCKS)))
    else:
        endpoints=all_endpoints
    for end in endpoints:
        train_end=max(0,end-int(horizon[:-1])-60); train=development[:train_end]; test=development[end:min(end+TEST_BLOCK,len(development))]
        if len(train)<MIN_TRAIN or len(test)<TEST_BLOCK//2:continue
        thresholds=_regime_thresholds(train); keys=[regime_key(r,thresholds) for r in test]
        parts=[]
        for name in names:
            pred=_fit_calibrated(train,test,mf[name])
            if pred is None: parts=[]; break
            parts.append(pred)
        if len(parts)!=len(names):continue
        global_w,regime_w=_history_weights(history,sorted(set(keys)),names)
        routed=np.asarray([_mix([p[i:i+1] for p in parts],regime_w.get(keys[i],global_w))[0] for i in range(len(test))])
        equal=_mix(parts,np.full(len(parts),1.0/len(parts)))
        meta_pred=_meta_predict(meta_history,parts,keys)
        y=[r["y"] for r in test]
        em=metrics(y,equal); rm=metrics(y,routed); mm=metrics(y,meta_pred) if meta_pred is not None else None; comp=[metrics(y,p) for p in parts]
        # Store one causal observation per row for local-regime learning. This is
        # intentionally prior-block data only for every future block.
        meta_x=_meta_features(parts,keys)
        if mm is not None:
            meta_history.extend({"x":meta_x[i].tolist(),"y":y[i]} for i in range(len(y)))
        else:
            # Even before the meta-router is eligible, store causal OOS rows so
            # later blocks can learn without ever seeing the current block twice.
            meta_history.extend({"x":meta_x[i].tolist(),"y":y[i]} for i in range(len(y)))
        for i,key in enumerate(keys):
            one=[metrics([y[i]],p[i:i+1]) for p in parts]
            history.append({"regime":key,"logloss":[s["logloss"] for s in one],"brier":[s["brier"] for s in one],"ece":[s["calibration_error"] for s in one]})
        blocks.append({
            "n": len(test),
            "regime_counts": {k: keys.count(k) for k in sorted(set(keys))},
            "equal": em,
            "routed": rm,
            "meta": mm if mm is not None else {"status":"not_yet_eligible"},
            "delta": {
                "accuracy": rm["accuracy"] - em["accuracy"],
                "logloss": rm["logloss"] - em["logloss"],
                "brier": rm["brier"] - em["brier"],
            },
            "global_weights_before": global_w.tolist(),
            "regime_weights_before": {
                k: regime_w[k].tolist()
                for k in sorted(regime_w)
            },
        })
    if len(blocks)<8:return {"status":"DEFERRED","n":len(rows),"reason":"insufficient_valid_oos_blocks"}
    ll=np.asarray([b["delta"]["logloss"] for b in blocks]); br=np.asarray([b["delta"]["brier"] for b in blocks]); ac=np.asarray([b["delta"]["accuracy"] for b in blocks])
    meta_blocks=[b for b in blocks if isinstance(b.get("meta"),dict) and "delta" in b["meta"]]
    mll=np.asarray([b["meta"]["delta"]["logloss"] for b in meta_blocks],dtype=float) if meta_blocks else np.asarray([])
    mbr=np.asarray([b["meta"]["delta"]["brier"] for b in meta_blocks],dtype=float) if meta_blocks else np.asarray([])
    mac=np.asarray([b["meta"]["delta"]["accuracy"] for b in meta_blocks],dtype=float) if meta_blocks else np.asarray([])
    summary={"blocks":len(blocks),"meta_blocks":len(meta_blocks),"samples":int(sum(b["n"] for b in blocks)),"max_oos_blocks":MAX_OOS_BLOCKS,"history_source":"Binance Vision closed archives","mean_accuracy_delta":float(ac.mean()),"mean_logloss_delta":float(ll.mean()),"mean_brier_delta":float(br.mean()),"improved_logloss_ratio":float(np.mean(ll<0)),"improved_brier_ratio":float(np.mean(br<0)),"non_worse_accuracy_ratio":float(np.mean(ac>=-0.005)),"meta_mean_accuracy_delta":float(mac.mean()) if len(mac) else None,"meta_mean_logloss_delta":float(mll.mean()) if len(mll) else None,"meta_mean_brier_delta":float(mbr.mean()) if len(mbr) else None,"meta_improved_logloss_ratio":float(np.mean(mll<0)) if len(mll) else None,"meta_improved_brier_ratio":float(np.mean(mbr<0)) if len(mbr) else None,"meta_non_worse_accuracy_ratio":float(np.mean(mac>=-0.005)) if len(mac) else None}
    thresholds=_regime_thresholds(development); hold_keys=[regime_key(r,thresholds) for r in holdout]
    hold_parts=[_fit_calibrated(development,holdout,mf[name]) for name in names]
    if any(p is None for p in hold_parts):return {"status":"DEFERRED","n":len(rows),"reason":"holdout_prediction_failed"}
    global_w,regime_w=_history_weights(history,sorted(set(hold_keys)),names)
    equal_hold=_mix(hold_parts,np.full(len(hold_parts),1.0/len(hold_parts)))
    routed_hold=np.asarray([_mix([p[i:i+1] for p in hold_parts],regime_w.get(hold_keys[i],global_w))[0] for i in range(len(holdout))])
    meta_hold=_meta_predict(meta_history,hold_parts,hold_keys)
    y_hold=[r["y"] for r in holdout]
    final_meta={"status":"DEFERRED","reason":"insufficient_prior_oos_meta_rows"} if meta_hold is None else metrics(y_hold,meta_hold)
    return {"status":"OK","schema_version":1,"research_only":True,"production_changed":False,"policy":"runtime_15_feature_causal_prior_oos_regime_loss_weighting_plus_causal_meta_router","final_holdout_protected":True,"final_holdout_used_for_selection":False,"feature_schema":"bootstrap_train.FEATURES_compatible_runtime_15","model_zoo":names,"summary":summary,"development_n":len(development),"final_holdout_n":len(holdout),"final_holdout":{"equal":metrics(y_hold,equal_hold),"routed":metrics(y_hold,routed_hold),"meta_router":final_meta,"weights":{"global":global_w.tolist(),"regime":{k:regime_w[k].tolist() for k in sorted(regime_w)}}},"blocks":blocks}

def main():
    payload={"schema_version":1,"research_only":True,"production_changed":False,"horizons":{h:evaluate(h) for h in HORIZONS}}
    OUT.parent.mkdir(parents=True,exist_ok=True); OUT.write_text(json.dumps(payload,indent=2,sort_keys=True),encoding="utf-8"); print(json.dumps(payload,indent=2,sort_keys=True))
if __name__=="__main__": main()
