"""Bounded multi-pattern chronological OOS laboratory for BTC research.

Research-only: no production model, registry, calibration artifact, frozen holdout,
or live prediction state is mutated. The lab explores model x feature family x
training-window patterns, then applies chronological calibration and a selective
decision policy only to a small pre-declared survivor set.
"""
from __future__ import annotations
import hashlib, json, math, os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from rolling_challenger_oos import load_current_rows
    from model_compare import CLASSES, aligned, metrics, _temperature, apply_temperature
except ModuleNotFoundError:
    from src.rolling_challenger_oos import load_current_rows
    from src.model_compare import CLASSES, aligned, metrics, _temperature, apply_temperature

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/historical_research/pattern_lab_oos.json"
CHECKPOINT = ROOT / "data/historical_research/pattern_lab_checkpoint.json"
FEATURES = ("ret_1m","ret_3m","ret_5m","ret_10m","acceleration","volatility_5m","volatility_10m","range_position_10m","body_1m","upper_wick_1m","lower_wick_1m","volume_ratio","volume_trend","ema_gap_5m","ema_gap_10m")
FEATURE_GROUPS = {
    "core_momentum": FEATURES[:5],
    "momentum_volatility": FEATURES[:7],
    "momentum_structure": FEATURES[:5] + FEATURES[7:11],
    "momentum_flow": FEATURES[:5] + FEATURES[11:13],
    "momentum_trend": FEATURES[:5] + FEATURES[13:15],
    "full_15": FEATURES,
}
MODEL_NAMES = ("logreg_c01","logreg_c1","extra_trees","hist_gradient_boosting")
WINDOWS = ("expanding","trailing_750","trailing_1500")
CALIBRATIONS = ("none","temperature")
SELECTIVE_THRESHOLDS = (0.34,0.45,0.55,0.65)
MAX_ROWS = int(os.getenv("PATTERN_LAB_MAX_ROWS","3000"))
TEST_BLOCK = int(os.getenv("PATTERN_LAB_TEST_BLOCK","100"))
MIN_TRAIN = int(os.getenv("PATTERN_LAB_MIN_TRAIN","800"))
FINAL_HOLDOUT_FRAC = 0.20
PURGE_MINUTES = 60
TOP_K_DEEP = 6

def _hash_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()[:16]

def _factory(name: str):
    if name == "logreg_c01":
        return Pipeline([("scale",StandardScaler()),("model",LogisticRegression(C=0.1,max_iter=2500))])
    if name == "logreg_c1":
        return Pipeline([("scale",StandardScaler()),("model",LogisticRegression(C=1.0,max_iter=2500))])
    if name == "extra_trees":
        return ExtraTreesClassifier(n_estimators=80,max_depth=7,min_samples_leaf=8,max_features="sqrt",random_state=42,n_jobs=-1)
    if name == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(max_iter=120,max_leaf_nodes=15,learning_rate=0.04,l2_regularization=1.0,random_state=42)
    raise ValueError(name)

def _candidate_names():
    return [f"{m}|{g}|{w}" for m in MODEL_NAMES for g in FEATURE_GROUPS for w in WINDOWS]

def _parse_candidate(name):
    return tuple(name.split("|"))

def _window_train(rows, end, window, horizon):
    purge_rows=max(1,int(math.ceil(PURGE_MINUTES/float(horizon.rstrip("m")))))
    safe_end=end-purge_rows
    if safe_end<=0: return []
    if window=="expanding": start=0
    elif window=="trailing_750": start=max(0,safe_end-750)
    elif window=="trailing_1500": start=max(0,safe_end-1500)
    else: raise ValueError(window)
    return rows[start:safe_end]

def _x(rows, feature_group):
    idx=[FEATURES.index(f) for f in FEATURE_GROUPS[feature_group]]
    return np.asarray([[r["x"][i] for i in idx] for r in rows],dtype=float)

def _fit_raw(train,test,model_name,feature_group):
    if len(train)<MIN_TRAIN or len(set(r["y"] for r in train))<3: return None
    try:
        model=_factory(model_name)
        model.fit(_x(train,feature_group),np.asarray([r["y"] for r in train]))
        return aligned(model,_x(test,feature_group))
    except (ValueError,RuntimeError):
        return None

def _fit_calibrated(train,test,model_name,feature_group):
    if len(train)<MIN_TRAIN or len(set(r["y"] for r in train))<3: return None
    split=max(int(len(train)*0.75),MIN_TRAIN-100)
    if split>=len(train)-30: return None
    cal_train,cal_rows=train[:split],train[split:]
    if len(set(r["y"] for r in cal_train))<3: return None
    try:
        cal_model=_factory(model_name)
        cal_model.fit(_x(cal_train,feature_group),np.asarray([r["y"] for r in cal_train]))
        temp=_temperature(aligned(cal_model,_x(cal_rows,feature_group)),[r["y"] for r in cal_rows])
        model=_factory(model_name)
        model.fit(_x(train,feature_group),np.asarray([r["y"] for r in train]))
        return apply_temperature(aligned(model,_x(test,feature_group)),temp)
    except (ValueError,RuntimeError):
        return None

def _ece(y,probs,bins=10):
    yi=np.asarray([CLASSES.index(v) for v in y],dtype=int)
    conf=np.max(probs,axis=1); correct=(np.argmax(probs,axis=1)==yi).astype(float)
    if not len(y): return float("nan")
    total=0.0
    edges=np.linspace(0,1,bins+1)
    for lo,hi in zip(edges[:-1],edges[1:]):
        mask=(conf>=lo)&((conf<hi) if hi<1 else (conf<=hi))
        if mask.any(): total+=float(mask.mean())*abs(float(correct[mask].mean())-float(conf[mask].mean()))
    return total

def _selective(y,probs,threshold):
    conf=np.max(probs,axis=1); keep=conf>=threshold
    cov=float(np.mean(keep)) if len(y) else 0.0
    acc=float("nan")
    if keep.any():
        yi=np.asarray([CLASSES.index(v) for v in y],dtype=int)
        acc=float(np.mean(np.argmax(probs[keep],axis=1)==yi[keep]))
    return {"threshold":threshold,"coverage":cov,"selective_accuracy":acc,"n_predicted":int(keep.sum())}

def _enrich_metrics(y, probs, base):
    """Add approximate dependent-series accuracy CI/ESS diagnostics."""
    yi=np.asarray([CLASSES.index(v) for v in y],dtype=int)
    correct=(np.argmax(probs,axis=1)==yi).astype(float)
    n=len(correct)
    if n<2:
        return dict(base, accuracy_ci95_approx=None, effective_sample_size_accuracy=None)
    mean=float(correct.mean())
    centered=correct-mean
    denom=float(np.sum(centered*centered))
    tau=1.0
    if denom>0:
        for lag in range(1,min(100,n-1)+1):
            rho=float(np.dot(centered[:-lag],centered[lag:])/denom)
            if rho<=0:
                break
            tau += 2.0*rho
    ess=float(max(1.0,min(float(n),n/tau)))
    z=1.959963984540054
    denom_w=1.0+z*z/ess
    center=(mean+z*z/(2*ess))/denom_w
    half=z*math.sqrt(max(0.0,mean*(1-mean)/ess+z*z/(4*ess*ess)))/denom_w
    return dict(base, accuracy_ci95_approx=[max(0.0,center-half),min(1.0,center+half)], effective_sample_size_accuracy=ess)

def _fingerprint(model,group,window):
    return _hash_text(f"model={model}|features={group}|window={window}")

def _checkpoint(status,horizon,**extra):
    CHECKPOINT.parent.mkdir(parents=True,exist_ok=True)
    CHECKPOINT.write_text(json.dumps({"schema_version":1,"experiment":"btc_pattern_lab_oos","status":status,"horizon":horizon,"analysis_git_sha":os.getenv("GITHUB_SHA","LOCAL_UNPINNED"),"updated_at_utc":datetime.now(timezone.utc).isoformat(),**extra},indent=2,sort_keys=True)+"\n")

def _evaluate_horizon(horizon):
    rows,prod_version,data_source=load_current_rows(horizon)
    rows=list(rows[-MAX_ROWS:])
    pit_status="STRICT_PRIMARY" if data_source=="live_binance_primary" else "ARCHIVE_PIT_UNPROVEN"
    names=_candidate_names()
    if len(rows)<MIN_TRAIN+TEST_BLOCK+100:
        return {"status":"DEFERRED","reason":"insufficient_rows","n":len(rows),"data_source":data_source,"pit_status":pit_status,"promotion_eligible":False,"candidate_count":len(names)}
    holdout_start=int(len(rows)*(1-FINAL_HOLDOUT_FRAC))
    dev,holdout=rows[:holdout_start],rows[holdout_start:]
    block_starts=[b for b in range(MIN_TRAIN,max(MIN_TRAIN+1,len(dev)-TEST_BLOCK+1),TEST_BLOCK) if b+max(10,TEST_BLOCK//2)<=len(dev)]
    direct={}
    for name in names:
        mname,group,window=_parse_candidate(name); ys=[]; cps=[]; pps=[]; blocks=[]
        for end in block_starts:
            test=dev[end:min(end+TEST_BLOCK,len(dev))]; train=_window_train(dev,end,window,horizon)
            cp=_fit_raw(train,test,mname,group)
            if cp is None: continue
            y=[r["y"] for r in test]; pp=np.asarray([r["production"] for r in test],float)
            cm,pm=metrics(y,cp),metrics(y,pp)
            blocks.append({"test_start":test[0]["created"],"test_end":test[-1]["created"],"test_n":len(test),"candidate_logloss":cm["logloss"],"production_logloss":pm["logloss"],"logloss_delta":cm["logloss"]-pm["logloss"],"brier_delta":cm["brier"]-pm["brier"],"accuracy_delta":cm["accuracy"]-pm["accuracy"]})
            ys.extend(y); cps.extend(cp.tolist()); pps.extend(pp.tolist())
        if ys:
            cm,pm=metrics(ys,cps),metrics(ys,pps)
            cm=_enrich_metrics(ys,np.asarray(cps,float),cm); pm=_enrich_metrics(ys,np.asarray(pps,float),pm)
            ll=np.asarray([b["logloss_delta"] for b in blocks]); br=np.asarray([b["brier_delta"] for b in blocks]); ac=np.asarray([b["accuracy_delta"] for b in blocks])
            direct[name]={"fingerprint":_fingerprint(mname,group,window),"model":mname,"feature_group":group,"feature_count":len(FEATURE_GROUPS[group]),"window":window,"blocks":len(blocks),"candidate":cm,"production":pm,"delta":{"accuracy":cm["accuracy"]-pm["accuracy"],"logloss":cm["logloss"]-pm["logloss"],"brier":cm["brier"]-pm["brier"]},"stability":{"improved_logloss_fraction":float(np.mean(ll<0)) if len(ll) else 0.0,"improved_brier_fraction":float(np.mean(br<0)) if len(br) else 0.0,"non_worse_accuracy_fraction":float(np.mean(ac>=-0.005)) if len(ac) else 0.0,"worst_logloss_delta":float(np.max(ll)) if len(ll) else None,"newest_logloss_delta":float(ll[-1]) if len(ll) else None}}
    if len(direct)<3:
        return {"status":"DEFERRED","reason":"too_few_successful_patterns","n":len(rows),"data_source":data_source,"pit_status":pit_status,"promotion_eligible":False,"candidate_count":len(names),"successful_candidates":len(direct)}
    history={name:[] for name in names}; preq=[]
    for end in block_starts:
        test=dev[end:min(end+TEST_BLOCK,len(dev))]; current={}
        for name in names:
            if name not in direct: continue
            mname,group,window=_parse_candidate(name); train=_window_train(dev,end,window,horizon); cp=_fit_raw(train,test,mname,group)
            if cp is None: continue
            y=[r["y"] for r in test]; current[name]=(metrics(y,cp)["logloss"],metrics(y,cp)["brier"],metrics(y,cp)["accuracy"])
        eligible=[n for n,s in history.items() if len(s)>=2 and n in current]
        if eligible:
            selected=min(eligible,key=lambda n:(float(np.mean(history[n])),n))
            ll,br,acc=current[selected]; pm=metrics([r["y"] for r in test],np.asarray([r["production"] for r in test],float))
            preq.append({"test_start":test[0]["created"],"test_end":test[-1]["created"],"test_n":len(test),"selected_pattern":selected,"selected_fingerprint":direct[selected]["fingerprint"],"candidate_logloss":ll,"production_logloss":pm["logloss"],"logloss_delta":ll-pm["logloss"],"accuracy_delta":acc-pm["accuracy"]})
        for n,v in current.items(): history[n].append(v[0])
    ranked=sorted(direct,key=lambda n:(direct[n]["candidate"]["logloss"],direct[n]["candidate"]["brier"],-direct[n]["candidate"]["accuracy"]))
    top_names=ranked[:TOP_K_DEEP]; deep={}
    for name in top_names:
        mname,group,window=_parse_candidate(name)
        for cal in CALIBRATIONS:
            ys=[]; ps=[]; blocks=[]
            for end in block_starts:
                test=dev[end:min(end+TEST_BLOCK,len(dev))]; train=_window_train(dev,end,window,horizon)
                cp=_fit_raw(train,test,mname,group) if cal=="none" else _fit_calibrated(train,test,mname,group)
                if cp is None: continue
                y=[r["y"] for r in test]; met=metrics(y,cp); blocks.append({"test_start":test[0]["created"],"test_end":test[-1]["created"],"test_n":len(test),"logloss":met["logloss"],"brier":met["brier"],"accuracy":met["accuracy"],"ece":_ece(y,cp)}); ys.extend(y); ps.extend(cp.tolist())
            if ys:
                met=metrics(ys,ps); met=_enrich_metrics(ys,np.asarray(ps,float),met); deep[f"{name}|cal={cal}"]={"pattern":name,"calibration":cal,"candidate":met,"ece":_ece(ys,np.asarray(ps,float)),"blocks":blocks,"block_count":len(blocks)}
    if not deep: return {"status":"DEFERRED","reason":"deep_pattern_replay_failed","n":len(rows),"data_source":data_source,"pit_status":pit_status,"promotion_eligible":False,"candidate_count":len(names),"successful_candidates":len(direct),"prequential":{"blocks":preq}}
    best_key=min(deep,key=lambda k:(deep[k]["candidate"]["logloss"],deep[k]["candidate"]["brier"],-deep[k]["candidate"]["accuracy"]))
    best=deep[best_key]; best_pattern=best["pattern"]; mname,group,window=_parse_candidate(best_pattern)
    train_hold=_window_train(dev,len(dev),window,horizon)
    holdout_probs=_fit_raw(train_hold,holdout,mname,group) if best["calibration"]=="none" else _fit_calibrated(train_hold,holdout,mname,group)
    y_hold=[r["y"] for r in holdout]; final=None; hold_sel=None; dev_sel=[]
    prod_hold=metrics(y_hold,np.asarray([r["production"] for r in holdout],float))
    chosen_threshold=0.34
    if holdout_probs is not None:
        fm=metrics(y_hold,holdout_probs); hold_sel=[_selective(y_hold,holdout_probs,t) for t in SELECTIVE_THRESHOLDS]
        gate_start=max(MIN_TRAIN,len(dev)-TEST_BLOCK); gate_test=dev[gate_start:]; gate_train=_window_train(dev,gate_start,window,horizon)
        gp=_fit_raw(gate_train,gate_test,mname,group) if best["calibration"]=="none" else _fit_calibrated(gate_train,gate_test,mname,group)
        if gp is not None:
            dev_sel=[_selective([r["y"] for r in gate_test],gp,t) for t in SELECTIVE_THRESHOLDS]
            valid=[x for x in dev_sel if x["coverage"]>=0.30 and np.isfinite(x["selective_accuracy"])]
            if valid: chosen_threshold=sorted(valid,key=lambda x:(-x["selective_accuracy"],-x["coverage"],x["threshold"]))[0]["threshold"]
        final={"selected_pattern":best_pattern,"calibration":best["calibration"],"candidate":fm,"production":prod_hold,"delta":{"accuracy":fm["accuracy"]-prod_hold["accuracy"],"logloss":fm["logloss"]-prod_hold["logloss"],"brier":fm["brier"]-prod_hold["brier"]},"selective":hold_sel,"protected":True}
    return {"status":"OK","schema_version":1,"research_only":True,"production_changed":False,"production_version":prod_version,"data_source":data_source,"pit_status":pit_status,"promotion_eligible":False,"candidate_budget":{"candidate_count":len(names),"model_count":len(MODEL_NAMES),"feature_group_count":len(FEATURE_GROUPS),"window_count":len(WINDOWS),"calibration_count":len(CALIBRATIONS),"decision_threshold_count":len(SELECTIVE_THRESHOLDS),"top_k_deep":TOP_K_DEEP,"selection_iterations":len(preq)},"evaluation":{"max_rows":MAX_ROWS,"test_block":TEST_BLOCK,"min_train":MIN_TRAIN,"purge_minutes":PURGE_MINUTES,"final_holdout_fraction":FINAL_HOLDOUT_FRAC,"chronological":True,"random_split":False,"nested_prequential_selection":True,"holdout_protected":True},"development":{"n":len(dev),"block_count":len(block_starts),"direct_patterns":direct,"prequential":{"blocks":preq,"selected_from_prior_blocks_only":True},"deep":deep,"best_development_key":best_key},"final_holdout":final,"selected_decision_policy":{"threshold":chosen_threshold,"development_gate_records":dev_sel,"holdout":hold_sel},"negative_knowledge_note":"All weak patterns remain descriptive evidence; no automatic production effect."}

def main():
    if os.getenv("TESTS_PASSED","").lower()!="true": raise SystemExit("BLOCKED: TESTS_PASSED must be true")
    if os.getenv("AUDIT_PASSED","").lower()!="true": raise SystemExit("BLOCKED: AUDIT_PASSED must be true")
    payload={"schema_version":1,"experiment":"btc_pattern_lab_oos","research_only":True,"production_changed":False,"analysis_git_sha":os.getenv("GITHUB_SHA","LOCAL_UNPINNED"),"status":"RUNNING","generated_at_utc":datetime.now(timezone.utc).isoformat(),"horizons":{},"cross_project_mechanism_only":True,"promotion_eligible":False}
    OUT.parent.mkdir(parents=True,exist_ok=True); OUT.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
    for h in ("5m","10m"):
        _checkpoint("RUNNING",h)
        payload["horizons"][h]=_evaluate_horizon(h)
        payload["updated_at_utc"]=datetime.now(timezone.utc).isoformat()
        OUT.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
    payload["status"]="COMPLETED_RESEARCH_ONLY" if all(payload["horizons"][h].get("status")=="OK" for h in ("5m","10m")) else "DEFERRED"
    payload["finished_at_utc"]=datetime.now(timezone.utc).isoformat()
    OUT.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
    return 0

if __name__=="__main__":
    raise SystemExit(main())
