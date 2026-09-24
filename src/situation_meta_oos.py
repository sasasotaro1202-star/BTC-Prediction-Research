"""PIT/OOS situation-aware meta-model research for BTC short-horizon prediction.

The meta-model consumes only prediction-time information already persisted with
historical production predictions: calibrated production probabilities,
microstructure snapshots, and descriptive situation state. It is strictly
research-only and never mutates production artifacts.
"""
from __future__ import annotations
import json, math, sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from model_compare import HORIZONS, CLASSES, DB, prediction_precedes_target, strict_pit_provenance_reason

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"data/historical_research/situation_meta_oos.json"
MIN_ROWS=3000; MIN_TRAIN=1000; TEST_BLOCK=25; MAX_BLOCKS=24; HOLDOUT_MIN=1000
PURGE_BARS={"5m":5,"10m":10}; EMBARGO_BARS={"5m":60,"10m":60}
MICRO_KEYS=("vwap_distance_5m","vwap_distance_15m","vwap_distance_30m","volume_burst_5m","volume_burst_15m","range_compression_5m","range_compression_15m","taker_imbalance_5m","taker_imbalance_15m","taker_imbalance_delta_5m_15m","book_imbalance","bybit_book_imbalance","cross_exchange_gap","spot_futures_gap","funding_binance","funding_bybit")
CAT_KEYS=(("trend_state",("RANGE","TREND_UP","TREND_DOWN")),("volatility_state",("STABLE","EXPANDING","COMPRESSING")),("orderflow_state",("BALANCED","BUY_PRESSURE","SELL_PRESSURE")),("signal_quality",("LOW","MEDIUM","HIGH")),("horizon_alignment",("AGREE","CONFLICT")),("direction_5m",CLASSES),("direction_10m",CLASSES),("data_state",("HEALTHY","PARTIAL","DEGRADED")))

def _safe_json(v:Any)->dict[str,Any]:
    try:
        x=json.loads(v) if isinstance(v,str) else v
        return x if isinstance(x,dict) else {}
    except (TypeError,ValueError,json.JSONDecodeError): return {}
def _finite(v:Any,default:float=0.0)->float:
    try:
        x=float(v); return x if math.isfinite(x) else float(default)
    except (TypeError,ValueError): return float(default)
def _parse_utc(v:Any)->datetime|None:
    try: d=datetime.fromisoformat(str(v).replace("Z","+00:00"))
    except (TypeError,ValueError): return None
    return d.astimezone(timezone.utc) if d.tzinfo else None
def primary_live_scenario(s:dict[str,Any])->bool:
    return isinstance(s,dict) and s.get("production_mode")=="binance_primary"
def _row_key(r:dict[str,Any])->tuple:
    return (r.get("created",""),r.get("target",""),r.get("model_version",""),tuple(round(float(v),12) for v in r.get("production",[])),r.get("horizon",""))

def _load_rows(horizon:str)->list[dict[str,Any]]:
    target_col="target_5m" if horizon=="5m" else "target_10m"; actual_col="actual_direction_5m" if horizon=="5m" else "actual_direction_10m"
    with sqlite3.connect(DB) as con:
        raw=con.execute(f"""SELECT prediction_id,created_at_utc,{target_col},feature_json,{actual_col},p_up_{horizon},p_down_{horizon},p_flat_{horizon},model_version,scenario_json FROM predictions WHERE {actual_col} IS NOT NULL ORDER BY created_at_utc,prediction_id""").fetchall()
    out=[]
    for r in raw:
        if not prediction_precedes_target(r[1],r[2]): continue
        s=_safe_json(r[9])
        if not primary_live_scenario(s): continue
        if strict_pit_provenance_reason(s,r[1]) is not None: continue
        f=_safe_json(r[3]); micro=s.get("microstructure"); situation=s.get("situation")
        production=[_finite(r[6]),_finite(r[7]),_finite(r[5])]
        if not isinstance(f,dict) or not isinstance(micro,dict) or not isinstance(situation,dict): continue
        if r[4] not in CLASSES or any(v<0 for v in production) or sum(production)<=0: continue
        total=sum(production); production=[v/total for v in production]
        out.append({"id":r[0],"created":r[1],"target":r[2],"x":f,"production":production,"y":r[4],"model_version":r[8],"microstructure":micro,"situation":situation,"production_mode":str(s.get("production_mode","")),"horizon":horizon})
    seen=set(); ded=[]
    for r in out:
        k=_row_key(r)
        if k in seen: continue
        seen.add(k); ded.append(r)
    return ded

def meta_feature_names()->list[str]:
    names=["prod_down","prod_flat","prod_up",*MICRO_KEYS]
    names.extend(f"missing={k}" for k in MICRO_KEYS)
    names.extend(["situation_entropy","situation_margin"])
    for k,values in CAT_KEYS: names.extend(f"{k}={v}" for v in values)
    return names

def build_meta_vector(row:dict[str,Any])->np.ndarray:
    p=row["production"]; micro=row.get("microstructure") or {}; situation=row.get("situation") or {}
    values=[float(p[0]),float(p[1]),float(p[2])]
    for k in MICRO_KEYS:
        raw=micro.get(k); values.append(_finite(raw,0.0))
    for k in MICRO_KEYS:
        raw=micro.get(k); values.append(1.0 if raw is None or not math.isfinite(_finite(raw,float("nan"))) else 0.0)
    values.extend([_finite(situation.get("normalized_entropy"),0.5),_finite(situation.get("probability_margin"),0.0)])
    for k,allowed in CAT_KEYS:
        actual=str(situation.get(k,"")); values.extend([1.0 if actual==v else 0.0 for v in allowed])
    arr=np.asarray(values,dtype=float)
    if not np.isfinite(arr).all(): raise ValueError("meta_feature_nonfinite")
    return arr

def causal_train_rows(rows:list[dict[str,Any]],test_start:str,horizon:str)->list[dict[str,Any]]:
    start=_parse_utc(test_start)
    if start is None: return []
    cutoff=start-timedelta(minutes=int(EMBARGO_BARS[horizon])); out=[]
    for r in rows:
        created=_parse_utc(r.get("created")); target=_parse_utc(r.get("target"))
        if created is None or target is None or created>=target: continue
        if target>=cutoff: continue
        out.append(r)
    return out

def factories()->dict[str,Callable[[],object]]:
    return {
      "logreg":lambda:Pipeline([("scale",StandardScaler()),("model",LogisticRegression(C=0.15,max_iter=3000))]),
      "extra_trees":lambda:ExtraTreesClassifier(n_estimators=300,max_depth=8,min_samples_leaf=10,max_features="sqrt",random_state=42,n_jobs=-1),
      "hgb":lambda:HistGradientBoostingClassifier(max_iter=220,max_leaf_nodes=15,learning_rate=0.04,l2_regularization=1.0,random_state=42)
    }

def aligned_probs(model:object,rows:list[dict[str,Any]])->np.ndarray:
    if not rows: return np.empty((0,3),dtype=float)
    raw=np.asarray(model.predict_proba(np.stack([build_meta_vector(r) for r in rows])),dtype=float)
    out=np.full((len(rows),3),1e-6,dtype=float); classes=[str(v) for v in getattr(model,"classes_",[])]
    for j,c in enumerate(classes):
        if c in CLASSES: out[:,CLASSES.index(c)]=raw[:,j]
    return out/out.sum(axis=1,keepdims=True)

def _metrics(y:list[str],p:np.ndarray)->dict[str,float]:
    yi=np.asarray([CLASSES.index(v) for v in y]); p=np.asarray(p,dtype=float)
    if len(yi)==0 or p.shape!=(len(yi),3): raise ValueError("metric_shape_invalid")
    return {"logloss":float(-np.mean(np.log(np.clip(p[np.arange(len(yi)),yi],1e-12,1.0)))),"brier":float(np.mean(np.sum((p-np.eye(3)[yi])**2,axis=1))),"accuracy":float(np.mean(np.argmax(p,axis=1)==yi)),"n":int(len(yi))}

def _select_model(train:list[dict[str,Any]],horizon:str)->tuple[str,object]|None:
    if len(train)<MIN_TRAIN: return None
    split=max(int(len(train)*0.70),len(train)-500)
    if split<500 or len(train)-split<100: return None
    valid_rows=train[split:]; fit_rows=causal_train_rows(train[:split],valid_rows[0].get("created",""),horizon)
    if len(fit_rows)<500: return None
    y=np.asarray([r["y"] for r in fit_rows])
    if len(set(y.tolist()))<3: return None
    scored=[]
    for name,factory in factories().items():
        try:
            m=factory(); m.fit(np.stack([build_meta_vector(r) for r in fit_rows]),y)
            score=_metrics([r["y"] for r in valid_rows],aligned_probs(m,valid_rows)); scored.append((0.70*score["logloss"]+0.30*score["brier"],name))
        except Exception: continue
    if not scored: return None
    scored.sort(); winner=scored[0][1]; model=factories()[winner]()
    model.fit(np.stack([build_meta_vector(r) for r in train]),np.asarray([r["y"] for r in train]))
    return winner,model

def _endpoints(n:int)->list[int]:
    x=list(range(MIN_TRAIN,n,TEST_BLOCK))
    return x if len(x)<=MAX_BLOCKS else sorted(set(int(v) for v in np.linspace(x[0],x[-1],MAX_BLOCKS)))

def _situation_summary(rows,baseline,candidate):
    groups={}
    for i,r in enumerate(rows): groups.setdefault(str((r.get("situation") or {}).get("market_state","UNKNOWN")),[]).append(i)
    out={}
    for state,idx in groups.items():
        y=[rows[i]["y"] for i in idx]; bp=np.stack([baseline[i] for i in idx]); cp=np.stack([candidate[i] for i in idx]); bm=_metrics(y,bp); cm=_metrics(y,cp)
        out[state]={"n":len(idx),"baseline":bm,"candidate":cm,"delta":{"logloss":cm["logloss"]-bm["logloss"],"brier":cm["brier"]-bm["brier"],"accuracy":cm["accuracy"]-bm["accuracy"]}}
    return out

def evaluate_horizon(horizon:str)->dict[str,Any]:
    rows=_load_rows(horizon)
    if len(rows)<MIN_ROWS: return {"status":"DEFERRED","research_only":True,"production_changed":False,"reason":"insufficient_strict_pit_primary_rows","n":len(rows),"data_source":"live_binance_primary"}
    holdout_n=max(HOLDOUT_MIN,int(len(rows)*0.10))
    if len(rows)<=MIN_TRAIN+TEST_BLOCK+holdout_n: return {"status":"DEFERRED","research_only":True,"production_changed":False,"reason":"insufficient_development_and_frozen_holdout_rows","n":len(rows),"data_source":"live_binance_primary"}
    development=rows[:-holdout_n]; holdout=rows[-holdout_n:]; blocks=[]
    for end in _endpoints(len(development)):
        test=development[end:min(end+TEST_BLOCK,len(development))]
        if len(test)<max(10,TEST_BLOCK//2): continue
        train=causal_train_rows(development[:end],test[0].get("created",""),horizon); selected=_select_model(train,horizon)
        if selected is None: continue
        name,model=selected; candidate=aligned_probs(model,test); baseline=np.stack([r["production"] for r in test]); bm=_metrics([r["y"] for r in test],baseline); cm=_metrics([r["y"] for r in test],candidate)
        blocks.append({"n":len(test),"model":name,"baseline":bm,"candidate":cm,"delta":{"logloss":cm["logloss"]-bm["logloss"],"brier":cm["brier"]-bm["brier"],"accuracy":cm["accuracy"]-bm["accuracy"]},"situations":_situation_summary(test,baseline,candidate)})
    if len(blocks)<12: return {"status":"DEFERRED","research_only":True,"production_changed":False,"reason":"insufficient_valid_oos_blocks","n":len(rows),"blocks":len(blocks),"data_source":"live_binance_primary"}
    ll=np.asarray([b["delta"]["logloss"] for b in blocks]); br=np.asarray([b["delta"]["brier"] for b in blocks]); ac=np.asarray([b["delta"]["accuracy"] for b in blocks])
    summary={"blocks":len(blocks),"samples":int(sum(b["n"] for b in blocks)),"mean_logloss_delta":float(ll.mean()),"mean_brier_delta":float(br.mean()),"mean_accuracy_delta":float(ac.mean()),"improved_logloss_ratio":float(np.mean(ll<0)),"improved_brier_ratio":float(np.mean(br<0)),"non_worse_accuracy_ratio":float(np.mean(ac>=-0.01))}
    selected=_select_model(development,horizon)
    if selected is None: return {"status":"DEFERRED","research_only":True,"production_changed":False,"reason":"final_model_fit_unavailable","n":len(rows),"blocks":len(blocks),"data_source":"live_binance_primary"}
    final_name,final_model=selected; hb=np.stack([r["production"] for r in holdout]); hc=aligned_probs(final_model,holdout); bm=_metrics([r["y"] for r in holdout],hb); cm=_metrics([r["y"] for r in holdout],hc)
    fd={"logloss":cm["logloss"]-bm["logloss"],"brier":cm["brier"]-bm["brier"],"accuracy":cm["accuracy"]-bm["accuracy"]}
    eligible=(summary["blocks"]>=12 and summary["improved_logloss_ratio"]>=0.60 and summary["improved_brier_ratio"]>=0.60 and summary["mean_logloss_delta"]<=-0.003 and summary["mean_brier_delta"]<=-0.0015 and summary["non_worse_accuracy_ratio"]>=0.80)
    return {"status":"OK","schema_version":1,"research_only":True,"production_changed":False,"final_holdout_protected":True,"final_holdout_used_for_selection":False,"data_source":"live_binance_primary","promotion_evidence_eligible":True,"selected_model_family":final_name,"summary":summary,"eligible_pending_frozen_holdout_confirmation":bool(eligible),"final_holdout":{"n":holdout_n,"baseline":bm,"candidate":cm,"delta":fd,"selected_model_family":final_name,"descriptive_only":True},"blocks":blocks}

def main():
    payload={"schema_version":1,"research_only":True,"production_changed":False,"policy":"strict_pit_primary_only; chronological_oos; purge_target_overlap; embargo_60m; frozen_holdout_descriptive_only; no_runtime_selection","feature_names":meta_feature_names(),"horizons":{h:evaluate_horizon(h) for h in HORIZONS}}
    OUT.parent.mkdir(parents=True,exist_ok=True); OUT.write_text(json.dumps(payload,indent=2,sort_keys=True),encoding="utf-8"); print(json.dumps(payload,indent=2,sort_keys=True))
if __name__=="__main__": main()
