"""Broad research-only BTC pattern matrix; never mutates Production."""
from __future__ import annotations
import hashlib, json, math, os, sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/"src"
if str(SRC) not in sys.path:
    sys.path.insert(0,str(SRC))
from ensemble_model import SoftVotingEnsemble
from model_compare import CLASSES, load_archive_research_rows

OUT=ROOT/"data/historical_research/pattern_matrix_research.json"
HORIZONS=("5m","10m")
MAX_ROWS=16000
HOLDOUT=2000
MIN_TRAIN=2500
# Broad but bounded hosted-run sweep.
SCREEN_BLOCKS=4
SCREEN_TEST=400
FINAL_BLOCKS=10
FINAL_TEST=350
FINALISTS=12
BOOTSTRAPS=400
BOOT_BLOCK=20
PURGE={"5m":5,"10m":10}
EMBARGO={"5m":60,"10m":60}

FEATURE_SETS={
 "all_15":("ret_1m","ret_3m","ret_5m","ret_10m","acceleration","volatility_5m","volatility_10m","range_position_10m","body_1m","upper_wick_1m","lower_wick_1m","volume_ratio","volume_trend","ema_gap_5m","ema_gap_10m"),
 "returns_momentum":("ret_1m","ret_3m","ret_5m","ret_10m","acceleration"),
 "volatility_regime":("volatility_5m","volatility_10m","range_position_10m"),
 "candle_shape":("range_position_10m","body_1m","upper_wick_1m","lower_wick_1m"),
 "volume_flow":("volume_ratio","volume_trend"),
 "trend":("ret_5m","ret_10m","ema_gap_5m","ema_gap_10m"),
 "compact_cross":("ret_1m","ret_3m","ret_5m","ret_10m","volatility_5m","range_position_10m","volume_ratio","ema_gap_5m"),
 "mean_reversion":("ret_1m","ret_3m","ret_5m","volatility_5m","range_position_10m","body_1m","ema_gap_5m","ema_gap_10m"),
 "price_structure":("ret_1m","ret_5m","ret_10m","acceleration","range_position_10m","body_1m","upper_wick_1m","lower_wick_1m"),
 "flow_trend":("ret_5m","ret_10m","volatility_5m","volatility_10m","volume_ratio","volume_trend","ema_gap_5m","ema_gap_10m"),
}
FEATURE_ORDER=FEATURE_SETS["all_15"]
FEATURE_INDEX={x:i for i,x in enumerate(FEATURE_ORDER)}
WINDOWS={"expanding":None,"recent_1500":1500,"recent_3000":3000}
MODELS=("logreg_c0.03","logreg_c0.1","logreg_c1.0","logreg_c3.0","extra_trees","rf","hgb","soft_ensemble")

def factory(name):
    if name=="logreg_c0.03":
        return Pipeline([("scale",StandardScaler()),("model",LogisticRegression(C=0.03,max_iter=3000))])
    if name=="logreg_c0.1":
        return Pipeline([("scale",StandardScaler()),("model",LogisticRegression(C=0.1,max_iter=3000))])
    if name=="logreg_c1.0":
        return Pipeline([("scale",StandardScaler()),("model",LogisticRegression(C=1.0,max_iter=3000))])
    if name=="logreg_c3.0":
        return Pipeline([("scale",StandardScaler()),("model",LogisticRegression(C=3.0,max_iter=3000))])
    if name=="extra_trees":
        return ExtraTreesClassifier(n_estimators=120,max_depth=9,min_samples_leaf=10,max_features="sqrt",random_state=42,n_jobs=-1)
    if name=="rf":
        return RandomForestClassifier(n_estimators=120,max_depth=9,min_samples_leaf=10,max_features="sqrt",random_state=42,n_jobs=-1)
    if name=="hgb":
        return HistGradientBoostingClassifier(max_iter=160,max_leaf_nodes=15,learning_rate=0.04,l2_regularization=1.5,random_state=42)
    if name=="soft_ensemble":
        return SoftVotingEnsemble(learn_weights=True)
    raise KeyError(name)

def norm(p):
    p=np.clip(np.asarray(p,float),1e-8,1.0)
    return p/p.sum(axis=1,keepdims=True)

def aligned(model,x):
    raw=norm(model.predict_proba(x))
    out=np.full((len(x),3),1e-8)
    for j,c in enumerate(getattr(model,"classes_",[])):
        if str(c) in CLASSES:
            out[:,CLASSES.index(str(c))]=raw[:,j]
    return norm(out)

def metrics(y,p):
    yi=np.asarray([CLASSES.index(str(v)) for v in y],int)
    p=norm(p); hit=p.argmax(1)==yi; one=np.eye(3)[yi]; conf=p.max(1)
    ece=0.0
    for i in range(10):
        lo,hi=i/10.0,(i+1)/10.0
        m=(conf>=lo)&((conf<=hi) if hi>=1 else (conf<hi))
        if m.any(): ece+=float(m.mean())*abs(float(hit[m].mean())-float(conf[m].mean()))
    return {"n":len(yi),"accuracy":float(hit.mean()),"logloss":float(-np.mean(np.log(np.clip(p[np.arange(len(yi)),yi],1e-8,1.0)))),"brier":float(np.mean(np.sum((p-one)**2,axis=1))),"ece":float(ece)}

def ess(v,max_lag=40):
    x=np.asarray(v,float); n=len(x)
    if n<5:return float(n)
    x=x-x.mean(); var=float(np.dot(x,x)/n)
    if var<=1e-12:return float(n)
    s=0.0
    for lag in range(1,min(max_lag,n-1)+1):
        rho=float(np.dot(x[:-lag],x[lag:])/(n-lag))/var
        if rho<=0: break
        s+=rho
    return float(min(n,max(1.0,n/(1+2*s))))

def fold_ends(n,minimum,test_size,count):
    latest=n-test_size
    if latest<=minimum:return []
    return sorted({int(v) for v in np.linspace(minimum,latest,num=count,dtype=int) if minimum<=int(v)<=latest})

def train_rows(rows,end,window):
    w=WINDOWS[window]
    return rows[:end] if w is None else rows[max(0,end-w):end]

def feature_matrix(rows,feature_set):
    idx=[FEATURE_INDEX[x] for x in FEATURE_SETS[feature_set]]
    return np.asarray([[r["x"][i] for i in idx] for r in rows],float)

def freq(train,n):
    c=np.asarray([sum(str(r["y"])==x for r in train) for x in CLASSES],float)
    if c.sum()<=0:c[:]=1.0
    return np.tile(c/c.sum(),(n,1))

def fit_raw(cfg,train,test):
    X=feature_matrix(train,cfg["feature_set"]); Xt=feature_matrix(test,cfg["feature_set"]); y=np.asarray([str(r["y"]) for r in train])
    if len(set(y.tolist()))<3: raise ValueError("training_class_coverage_below_three")
    m=factory(cfg["model"]); m.fit(X,y); return aligned(m,Xt)

def fit_temp(cfg,train,test):
    if len(train)<400:return fit_raw(cfg,train,test)
    split=min(max(int(len(train)*0.75),MIN_TRAIN-250),len(train)-100)
    inner,cal=train[:split],train[split:]
    X=feature_matrix(inner,cfg["feature_set"]); Xt=feature_matrix(cal,cfg["feature_set"]); yi=np.asarray([CLASSES.index(str(r["y"])) for r in cal],int)
    im=factory(cfg["model"]); im.fit(X,np.asarray([str(r["y"]) for r in inner]))
    cp=aligned(im,Xt)
    logits=np.log(np.clip(cp,1e-8,1.0)); best_t,best=float(1.0),float("inf")
    for t in np.linspace(0.7,2.5,37):
        z=logits/float(t); z-=z.max(1,keepdims=True); q=np.exp(z); q/=q.sum(1,keepdims=True)
        ll=float(-np.mean(np.log(np.clip(q[np.arange(len(yi)),yi],1e-8,1.0))))
        if ll<best:best,best_t=ll,float(t)
    Xall=feature_matrix(train,cfg["feature_set"]); yall=np.asarray([str(r["y"]) for r in train]); Xtest=feature_matrix(test,cfg["feature_set"])
    m=factory(cfg["model"]); m.fit(Xall,yall); raw=aligned(m,Xtest)
    z=np.log(np.clip(raw,1e-8,1.0))/best_t; z-=z.max(1,keepdims=True); q=np.exp(z); q/=q.sum(1,keepdims=True)
    return q

def bootstrap(y,ref,cand,seed=42):
    yi=np.asarray([CLASSES.index(str(v)) for v in y],int); ref=norm(ref); cand=norm(cand); n=len(y)
    ll=-np.log(np.clip(cand[np.arange(n),yi],1e-8,1.0))+np.log(np.clip(ref[np.arange(n),yi],1e-8,1.0))
    one=np.eye(3)[yi]; br=np.sum((cand-one)**2,1)-np.sum((ref-one)**2,1)
    ac=(cand.argmax(1)==yi).astype(float)-(ref.argmax(1)==yi).astype(float)
    if n<max(30,BOOT_BLOCK): return {"n":n,"ess":ess(ll),"bootstraps":0}
    rng=np.random.default_rng(seed); starts=np.arange(0,n-BOOT_BLOCK+1); k=math.ceil(n/BOOT_BLOCK); out=[[],[],[]]
    for _ in range(BOOTSTRAPS):
        ix=[]
        for _ in range(k):
            s=int(rng.choice(starts)); ix.extend(range(s,min(s+BOOT_BLOCK,n)))
        ix=np.asarray(ix[:n],int)
        out[0].append(float(ll[ix].mean())); out[1].append(float(br[ix].mean())); out[2].append(float(ac[ix].mean()))
    return {"n":n,"ess":ess(ll),"bootstraps":BOOTSTRAPS,
            "logloss_diff_mean_candidate_minus_reference":float(ll.mean()),
            "logloss_diff_ci95":[float(np.quantile(out[0],.025)),float(np.quantile(out[0],.975))],
            "brier_diff_mean_candidate_minus_reference":float(br.mean()),
            "brier_diff_ci95":[float(np.quantile(out[1],.025)),float(np.quantile(out[1],.975))],
            "accuracy_diff_mean_candidate_minus_reference":float(ac.mean()),
            "accuracy_diff_ci95":[float(np.quantile(out[2],.025)),float(np.quantile(out[2],.975))]}

def cfg(f,m,w):
    base={"feature_set":f,"model":m,"window":w}
    base["fingerprint"]=hashlib.sha256(json.dumps(base,sort_keys=True).encode()).hexdigest()
    return base

def candidates():
    return [cfg(f,m,w) for f in FEATURE_SETS for m in MODELS for w in WINDOWS]

def screen(h,dev,c):
    starts=fold_ends(len(dev),MIN_TRAIN,SCREEN_TEST,SCREEN_BLOCKS); details=[]; ys=[]; refs=[]; cps=[]
    for s in starts:
        tr=train_rows(dev,max(0,s-PURGE[h]-EMBARGO[h]),c["window"]); te=dev[s:s+SCREEN_TEST]
        if len(tr)<MIN_TRAIN or len(te)<250: continue
        try: cp=fit_raw(c,tr,te)
        except (ValueError,RuntimeError): continue
        y=[str(r["y"]) for r in te]; rp=freq(tr,len(te)); cm,rm=metrics(y,cp),metrics(y,rp)
        details.append({"start_index":s,"train_n":len(tr),"n":len(te),"candidate":cm,"frequency_baseline":rm,"logloss_delta":cm["logloss"]-rm["logloss"],"brier_delta":cm["brier"]-rm["brier"],"accuracy_delta":cm["accuracy"]-rm["accuracy"]})
        ys+=y; refs.append(rp); cps.append(cp)
    if len(ys)<SCREEN_TEST*3:return None
    cm,rm=metrics(ys,np.vstack(cps)),metrics(ys,np.vstack(refs))
    ll=np.asarray([d["logloss_delta"] for d in details]); br=np.asarray([d["brier_delta"] for d in details]); ac=np.asarray([d["accuracy_delta"] for d in details])
    return {**c,"horizon":h,"stage":"SCREEN","folds":len(details),"aggregate":{"candidate":cm,"frequency_baseline":rm,"relative_logloss_improvement":(rm["logloss"]-cm["logloss"])/abs(rm["logloss"]),"relative_brier_improvement":(rm["brier"]-cm["brier"])/abs(rm["brier"])},
            "stability":{"accuracy_non_worse_ratio":float(np.mean(ac>=-.005)),"logloss_improved_ratio":float(np.mean(ll<0)),"brier_improved_ratio":float(np.mean(br<0)),"worst_logloss_delta":float(ll.max()),"worst_accuracy_delta":float(ac.min())},"folds_detail":details}


def _rank01(values, higher_is_better=True):
    vals=np.asarray(values,dtype=float)
    if len(vals)==0: return np.asarray([],dtype=float)
    order=np.argsort(vals if higher_is_better else -vals,kind="mergesort")
    ranks=np.empty(len(vals),dtype=float); ranks[order]=np.arange(len(vals),dtype=float)
    if len(vals)==1: return np.ones(1,dtype=float)
    return ranks/(len(vals)-1)

def screen_selection_score(rows):
    """Multi-objective screen score for future-generalization-oriented finalist selection."""
    if not rows: return np.asarray([],dtype=float)
    ll=_rank01([r["aggregate"]["relative_logloss_improvement"] for r in rows],True)
    br=_rank01([r["aggregate"]["relative_brier_improvement"] for r in rows],True)
    acc=_rank01([r["stability"]["accuracy_non_worse_ratio"] for r in rows],True)
    lls=_rank01([r["stability"]["logloss_improved_ratio"] for r in rows],True)
    brs=_rank01([r["stability"]["brier_improved_ratio"] for r in rows],True)
    wll=_rank01([r["stability"]["worst_logloss_delta"] for r in rows],False)
    wacc=_rank01([r["stability"]["worst_accuracy_delta"] for r in rows],True)
    return 0.24*ll + 0.18*br + 0.18*acc + 0.12*lls + 0.10*brs + 0.10*wll + 0.08*wacc

def select_finalists(screened, limit=FINALISTS):
    """Select screen finalists while preserving model/feature/window diversity."""
    if not screened: return []
    scores=screen_selection_score(screened)
    ranked=[]
    for row,score in zip(screened,scores):
        copy=dict(row)
        copy["screen_selection_score"]=float(score)
        ranked.append(copy)
    ranked.sort(key=lambda x:(
        -x["screen_selection_score"],
        -x["aggregate"]["relative_logloss_improvement"],
        -x["aggregate"]["relative_brier_improvement"],
        x["stability"]["worst_logloss_delta"],
    ))
    pool=ranked[:max(limit*4,limit)]
    chosen=[]
    seen_feature=set()
    seen_model=set()
    seen_window=set()
    while pool and len(chosen)<limit:
        best=None
        best_key=None
        for row in pool:
            bonus=0.0
            if row["feature_set"] not in seen_feature: bonus+=0.035
            if row["model"] not in seen_model: bonus+=0.025
            if row["window"] not in seen_window: bonus+=0.020
            key=(
                row["screen_selection_score"]+bonus,
                row["stability"]["accuracy_non_worse_ratio"],
                -row["stability"]["worst_logloss_delta"],
            )
            if best is None or key>best_key:
                best=row
                best_key=key
        chosen.append(best)
        pool=[r for r in pool if r["fingerprint"]!=best["fingerprint"]]
        seen_feature.add(best["feature_set"])
        seen_model.add(best["model"])
        seen_window.add(best["window"])
    chosen.sort(key=lambda x:-x["screen_selection_score"])
    return chosen

def final(h,dev,c):
    starts=fold_ends(len(dev),MIN_TRAIN,FINAL_TEST,FINAL_BLOCKS); details=[]; ys=[]; refs=[]; cps=[]; parts=[]
    for s in starts:
        tr=train_rows(dev,max(0,s-PURGE[h]-EMBARGO[h]),c["window"]); te=dev[s:s+FINAL_TEST]
        if len(tr)<MIN_TRAIN or len(te)<200:continue
        try: cp=fit_temp(c,tr,te)
        except (ValueError,RuntimeError):continue
        y=[str(r["y"]) for r in te]; rp=freq(tr,len(te)); cm,rm=metrics(y,cp),metrics(y,rp)
        details.append({"start_index":s,"train_n":len(tr),"n":len(te),"candidate":cm,"frequency_baseline":rm,"logloss_delta":cm["logloss"]-rm["logloss"],"brier_delta":cm["brier"]-rm["brier"],"accuracy_delta":cm["accuracy"]-rm["accuracy"]})
        ys+=y; refs.append(rp); cps.append(cp); parts.append((s,cp))
    if len(ys)<FINAL_TEST*5:return None
    cp,rp=np.vstack(cps),np.vstack(refs); ll=np.asarray([d["logloss_delta"] for d in details]); br=np.asarray([d["brier_delta"] for d in details]); ac=np.asarray([d["accuracy_delta"] for d in details])
    champ=None; champ_version=None; hpfile=ROOT/"models"/f"{h}.joblib"; metafile=ROOT/"models"/f"{h}.json"
    if hpfile.is_file() and metafile.is_file():
        champion=joblib.load(hpfile); x=np.asarray([r["x"] for r in dev],float); ids=sorted(i for s,p in parts for i in range(s,s+len(p)))
        ids=sorted(set(ids)); yp=[str(dev[i]["y"]) for i in ids]; cand_by={}
        for s,p in parts:
            for j,rowp in enumerate(p): cand_by[s+j]=rowp
        ids=[i for i in ids if i in cand_by]
        if ids:
            cp2=np.vstack([cand_by[i] for i in ids]); hp=aligned(champion,x[ids]); champ={"candidate":metrics([str(dev[i]["y"]) for i in ids],cp2),"champion":metrics([str(dev[i]["y"]) for i in ids],hp),"bootstrap":bootstrap([str(dev[i]["y"]) for i in ids],hp,cp2,137),"n":len(ids)}; champ_version=json.loads(metafile.read_text())["model_version"]
    yidx=np.asarray([CLASSES.index(v) for v in ys]); paired=-np.log(np.clip(cp[np.arange(len(ys)),yidx],1e-8,1))+np.log(np.clip(rp[np.arange(len(ys)),yidx],1e-8,1))
    return {**c,"horizon":h,"stage":"FINAL_DEVELOPMENT","folds":len(details),"aggregate":{"candidate":metrics(ys,cp),"frequency_baseline":metrics(ys,rp),"relative_logloss_improvement":(metrics(ys,rp)["logloss"]-metrics(ys,cp)["logloss"])/abs(metrics(ys,rp)["logloss"]),"relative_brier_improvement":(metrics(ys,rp)["brier"]-metrics(ys,cp)["brier"])/abs(metrics(ys,rp)["brier"]),"effective_sample_size":ess(paired)},"stability":{"accuracy_non_worse_ratio":float(np.mean(ac>=-.005)),"logloss_improved_ratio":float(np.mean(ll<0)),"brier_improved_ratio":float(np.mean(br<0)),"worst_logloss_delta":float(ll.max()),"worst_accuracy_delta":float(ac.min())},"paired_block_bootstrap_vs_frequency":bootstrap(ys,rp,cp),"incumbent_comparison":champ,"incumbent_model_version":champ_version,"folds_detail":details}

def holdout(h,rows,c):
    if len(rows)<=HOLDOUT+MIN_TRAIN:return {"status":"DEFERRED","reason":"insufficient_holdout_geometry"}
    dev,ho=rows[:-HOLDOUT],rows[-HOLDOUT:]; y=[str(r["y"]) for r in ho]
    try:cp=fit_temp(c,dev,ho)
    except (ValueError,RuntimeError) as e:return {"status":"DEFERRED","reason":type(e).__name__}
    result={"status":"DESCRIPTIVE_ONLY","n":len(ho),"candidate":metrics(y,cp),"frequency_baseline":metrics(y,freq(dev,len(ho))),"selection_allowed":False,"used_for_selection":False,"used_for_gate":False}
    hpfile=ROOT/"models"/f"{h}.joblib"
    if hpfile.is_file():
        hp=aligned(joblib.load(hpfile),np.asarray([r["x"] for r in ho],float)); result["champion"]=metrics(y,hp); result["candidate_vs_champion"]=bootstrap(y,hp,cp,271)
    return result

def run_horizon(h,rows):
    if len(rows)<=HOLDOUT+MIN_TRAIN+FINAL_TEST*5:return {"status":"DEFERRED","reason":"insufficient_archive_rows","n":len(rows)}
    dev=rows[:-HOLDOUT]; cs=candidates(); screened=[]; screen_failures=[]
    for c in cs:
        try:
            r=screen(h,dev,c)
            if r is not None:
                screened.append(r)
            else:
                screen_failures.append({"config":c,"error_type":"NO_VALID_SCREEN_FOLDS","error":"candidate produced fewer than the minimum valid chronological screen blocks"})
        except Exception as e:
            screen_failures.append({"config":c,"error_type":type(e).__name__,"error":str(e)[:300]})
    screened.sort(key=lambda x:(-x["aggregate"]["relative_logloss_improvement"],-x["aggregate"]["relative_brier_improvement"],-x["stability"]["accuracy_non_worse_ratio"],x["stability"]["worst_logloss_delta"]))
    fs=[]; failures=[]
    selected_finalists=select_finalists(screened,FINALISTS)
    for c in selected_finalists:
        clean={k:c[k] for k in ("feature_set","model","window","fingerprint")}
        try:
            r=final(h,dev,clean)
            if r is not None:fs.append(r)
        except Exception as e:failures.append({"config":clean,"error_type":type(e).__name__,"error":str(e)[:300]})
    fs.sort(key=lambda x:(-x["aggregate"]["relative_logloss_improvement"],-x["aggregate"]["relative_brier_improvement"],-x["stability"]["accuracy_non_worse_ratio"],x["stability"]["worst_logloss_delta"]))
    best={k:fs[0][k] for k in ("feature_set","model","window","fingerprint")} if fs else None
    return {"status":"OK","horizon":h,"archive_rows":len(rows),"data_sources":sorted({str(r.get("data_source",r.get("production_mode","unknown"))) for r in rows}),"development_rows":len(dev),"frozen_holdout_rows":HOLDOUT,"candidate_count":len(cs),"screened_count":len(screened),"finalist_count":len(fs),"final_holdout_protected":True,"holdout_used_for_selection":False,"holdout_used_for_gate":False,"candidate_budget":{"feature_sets":len(FEATURE_SETS),"models":len(MODELS),"windows":len(WINDOWS),"cartesian_candidates":len(cs),"screen_blocks":SCREEN_BLOCKS,"final_blocks":FINAL_BLOCKS,"finalists_evaluated":FINALISTS},"screen_top":screened[:20],"finalists":fs,"screen_selected_finalists":selected_finalists[:FINALISTS],"finalist_selection_policy":"rank_ensemble_plus_diversity_bonus;no_holdout_access","best_development_pattern":best,"descriptive_frozen_holdout":holdout(h,rows,best) if best else {"status":"DEFERRED","reason":"no_finalist"},"failures":screen_failures+failures,"research_only":True,"production_changed":False,"promotion_allowed":False}

def main():
    sha=os.getenv("GITHUB_SHA") or "LOCAL_UNPINNED"
    out={"schema_version":1,"generated_at_utc":datetime.now(timezone.utc).isoformat(),"analysis_git_sha":sha,"analysis_git_sha_status":"PINNED" if sha!="LOCAL_UNPINNED" else "LOCAL_UNPINNED","research_only":True,"production_changed":False,"promotion_allowed":False,"pit_evidence_status":"NON_STRICT_ARCHIVE_TIMING","promotion_evidence_eligible":False,"policy":"broad_pattern_matrix; chronological_WFO; prequential_temperature; protected_holdout; dependence_aware_bootstrap; incumbent_comparison","feature_sets":FEATURE_SETS,"models":list(MODELS),"windows":list(WINDOWS),"horizons":{}}
    for h in HORIZONS: out["horizons"][h]=run_horizon(h,load_archive_research_rows(h,MAX_ROWS))
    OUT.parent.mkdir(parents=True,exist_ok=True); OUT.write_text(json.dumps(out,indent=2,sort_keys=True)+"\n",encoding="utf-8"); print(json.dumps(out,indent=2))

if __name__=="__main__":main()
