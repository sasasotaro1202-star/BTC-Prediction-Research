"""Research-only high-risk conditional expert rescue routing for BTC.

The production/ensemble prediction remains the default. Only cases classified
as high-risk from prediction-time uncertainty are eligible for routing to a
specialist expert. Specialist rescue models are trained chronologically from
strictly prior labels: expert correct AND ensemble wrong. The current test
block never contributes to routing decisions.

Archive replay is research-only because publication timing is not live-PIT.
No production artifacts are modified.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import sys

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from model_compare import CLASSES, load_archive_research_rows, metrics

OUT = ROOT / "data" / "historical_research" / "risk_conditional_rescue_oos.json"
HORIZONS = ("5m", "10m")
MIN_TRAIN = 3000
META_BLOCK = 600
TEST_BLOCK = 600
MIN_ROUTER_ROWS = 300
MAX_ROWS = 12000
HIGH_RISK_QUANTILE = 0.75
RESCUE_THRESHOLD = 0.60
EPS = 1e-7

def _align(model, rows):
    X = np.asarray([r["x"] for r in rows], dtype=float)
    raw = np.asarray(model.predict_proba(X), dtype=float)
    out = np.full((len(rows), 3), EPS, dtype=float)
    for j, cls in enumerate(model.classes_):
        if str(cls) in CLASSES:
            out[:, CLASSES.index(str(cls))] = raw[:, j]
    out = np.clip(out, EPS, 1.0)
    return out / out.sum(axis=1, keepdims=True)

def _factories():
    return {
        "logreg": lambda: Pipeline([
            ("scale", StandardScaler()),
            ("model", LogisticRegression(C=0.3, max_iter=2500)),
        ]),
        "extra_trees": lambda: ExtraTreesClassifier(
            n_estimators=180, max_depth=11, min_samples_leaf=14,
            max_features="sqrt", random_state=42, n_jobs=-1,
        ),
        "hgb": lambda: HistGradientBoostingClassifier(
            max_iter=180, max_leaf_nodes=15, learning_rate=0.04,
            l2_regularization=1.5, random_state=42,
        ),
    }

def _fit_experts(train):
    X=np.asarray([r["x"] for r in train],dtype=float)
    y=np.asarray([r["y"] for r in train],dtype=str)
    if len(train)<MIN_TRAIN or len(set(y.tolist()))<3:
        raise ValueError("insufficient_expert_training")
    out={}
    for name,factory in _factories().items():
        m=factory(); m.fit(X,y); out[name]=m
    return out

def _probs(models, rows):
    return {name:_align(model,rows) for name,model in models.items()}

def _entropy(p):
    q=np.clip(np.asarray(p,dtype=float),EPS,1.0)
    return -np.sum(q*np.log(q),axis=1)/math.log(3.0)

def _risk_features(probs, rows):
    names=tuple(probs)
    stack=np.stack([probs[n] for n in names],axis=0)
    center=stack.mean(axis=0)
    disagreement=np.mean(np.sum((stack-center[None,:,:])**2,axis=2),axis=0)
    ens=center
    ordered=np.sort(ens,axis=1)
    argmax=ens.argmax(axis=1)
    agreement=np.mean(stack.argmax(axis=2)==argmax[None,:],axis=0)
    x=np.asarray([r["x"] for r in rows],dtype=float)
    context=x[:,[2,3,5,6,7,11,12]]
    feat=np.column_stack([
        ens[:,0],ens[:,1],ens[:,2],ens.max(axis=1),
        ordered[:,-1]-ordered[:,-2],
        _entropy(ens),
        disagreement/0.10,
        agreement,
        context,
    ])
    feat=np.clip(feat,-10.0,10.0)
    if not np.isfinite(feat).all():
        raise ValueError("risk_features_nonfinite")
    return feat

def _fit_rescue(meta_rows, meta_probs, meta_risk):
    ensemble=np.mean(np.stack(list(meta_probs.values()),axis=0),axis=0)
    ens_pred=ensemble.argmax(axis=1)
    y=np.asarray([CLASSES.index(r["y"]) for r in meta_rows],dtype=int)
    models={}
    for expert,probs in meta_probs.items():
        pred=probs.argmax(axis=1)
        rescue=((pred==y)&(ens_pred!=y)).astype(int)
        if len(rescue)<MIN_ROUTER_ROWS or len(np.unique(rescue))<2:
            continue
        m=Pipeline([
            ("scale",StandardScaler()),
            ("model",LogisticRegression(C=0.25,max_iter=2200,class_weight="balanced")),
        ])
        m.fit(meta_risk,rescue)
        models[expert]=m
    return models

def _route(test_rows, test_probs, rescue_models, threshold, risk_threshold=1.0):
    names=tuple(test_probs)
    ensemble=np.mean(np.stack([test_probs[n] for n in names],axis=0),axis=0)
    x=_risk_features(test_probs,test_rows)
    risk=-np.sum(ensemble*np.log(np.clip(ensemble,EPS,1.0)),axis=1)/math.log(3.0)
    risk_q=float(risk_threshold)
    high=risk>=risk_q
    scores=np.full((len(test_rows),len(names)),-np.inf,dtype=float)
    for j,name in enumerate(names):
        m=rescue_models.get(name)
        if m is not None:
            scores[:,j]=m.predict_proba(x)[:,1]
    best=scores.argmax(axis=1)
    best_score=scores[np.arange(len(test_rows)),best]
    use=high & np.isfinite(best_score) & (best_score>=threshold)
    out=ensemble.copy()
    for i in np.where(use)[0]:
        out[i]=test_probs[names[best[i]]][i]
    out=np.clip(out,EPS,1.0)
    out/=out.sum(axis=1,keepdims=True)
    chosen=np.full(len(test_rows),"ensemble",dtype=object)
    for i in np.where(use)[0]:
        chosen[i]=names[best[i]]
    return out, use, chosen, high, risk_q, best_score

def _evaluate(horizon):
    rows=load_archive_research_rows(horizon,MAX_ROWS)
    if len(rows)<MIN_TRAIN+2*TEST_BLOCK:
        return {"status":"DEFERRED","reason":"insufficient_archive_rows","n":len(rows)}
    split=int(len(rows)*0.80)
    development=rows[:split]
    holdout=rows[split:]
    blocks=[]
    prior_meta=[]
    for test_start in range(MIN_TRAIN+META_BLOCK, len(development), TEST_BLOCK):
        test_end=min(test_start+TEST_BLOCK,len(development))
        test=development[test_start:test_end]
        meta_end=test_start
        meta_start=meta_end-META_BLOCK
        train=development[:meta_start]
        meta=development[meta_start:meta_end]
        if len(test)<TEST_BLOCK or len(meta)<MIN_ROUTER_ROWS or len(train)<MIN_TRAIN:
            continue
        base_models=_fit_experts(train)
        meta_probs=_probs(base_models,meta)
        meta_features=_risk_features(meta_probs,meta)
        meta_ensemble=np.mean(np.stack(list(meta_probs.values()),axis=0),axis=0)
        meta_risk=-np.sum(meta_ensemble*np.log(np.clip(meta_ensemble,EPS,1.0)),axis=1)/math.log(3.0)
        risk_threshold=float(np.quantile(meta_risk,HIGH_RISK_QUANTILE))
        rescue_models=_fit_rescue(meta,meta_probs,meta_features)
        test_models=_fit_experts(development[:test_start])
        test_probs=_probs(test_models,test)
        routed,use,chosen,high,risk_q,scores=_route(test,test_probs,rescue_models,RESCUE_THRESHOLD,risk_threshold)
        ensemble=np.mean(np.stack(list(test_probs.values()),axis=0),axis=0)
        y=[r["y"] for r in test]
        base_m=metrics(y,ensemble); cand_m=metrics(y,routed)
        idx=np.asarray([CLASSES.index(v) for v in y],dtype=int)
        high_n=int(high.sum()); route_n=int(use.sum())
        base_hit=(ensemble.argmax(axis=1)==idx)
        cand_hit=(routed.argmax(axis=1)==idx)
        blocks.append({
            "n":len(test),
            "baseline":base_m,
            "candidate":cand_m,
            "delta":{k:float(cand_m[k]-base_m[k]) for k in ("accuracy","logloss","brier")},
            "high_risk_n":high_n,
            "risk_coverage":float(high.mean()),
            "route_n":route_n,
            "route_coverage":float(use.mean()),
            "high_risk_accuracy_baseline":float(base_hit[high].mean()) if high_n else None,
            "high_risk_accuracy_candidate":float(cand_hit[high].mean()) if high_n else None,
            "chosen":{"ensemble":float(np.mean(chosen=="ensemble")),
                      **{n:float(np.mean(chosen==n)) for n in names(test_probs)}},
            "risk_quantile":risk_q,
        })
        prior_meta.append((meta,meta_probs))
    if not blocks:
        return {"status":"DEFERRED","reason":"no_valid_blocks","n":len(rows)}
    def agg(side,key):
        metric_key = "calibration_error" if key == "ece" else key
        return float(sum(b["n"]*b[side][metric_key] for b in blocks)/sum(b["n"] for b in blocks))
    base={k:agg("baseline",k) for k in ("accuracy","logloss","brier","ece")}
    cand={k:agg("candidate",k) for k in ("accuracy","logloss","brier","ece")}
    deltas={k:cand[k]-base[k] for k in base}
    stable={
        "improved_logloss_ratio":float(np.mean([b["delta"]["logloss"]<0 for b in blocks])),
        "improved_brier_ratio":float(np.mean([b["delta"]["brier"]<0 for b in blocks])),
        "non_worse_accuracy_ratio":float(np.mean([b["delta"]["accuracy"]>=-0.005 for b in blocks])),
    }
    # Freeze the last trained rescue layer from development only for the blind holdout.
    hold_models=_fit_experts(development)
    hold_probs=_probs(hold_models,holdout)
    if len(prior_meta)>=1:
        # Use only the immediately prior meta block, not holdout labels.
        meta_rows,meta_probs=prior_meta[-1]
        meta_features=_risk_features(meta_probs,meta_rows)
        meta_ensemble=np.mean(np.stack(list(meta_probs.values()),axis=0),axis=0)
        meta_risk=-np.sum(meta_ensemble*np.log(np.clip(meta_ensemble,EPS,1.0)),axis=1)/math.log(3.0)
        hold_risk_threshold=float(np.quantile(meta_risk,HIGH_RISK_QUANTILE))
        hold_rescue=_fit_rescue(meta_rows,meta_probs,meta_features)
    else:
        hold_rescue={}
        hold_risk_threshold=1.0
    routed,use,chosen,high,risk_q,scores=_route(holdout,hold_probs,hold_rescue,RESCUE_THRESHOLD,hold_risk_threshold)
    yh=[r["y"] for r in holdout]
    hb=metrics(yh,np.mean(np.stack(list(hold_probs.values()),axis=0),axis=0))
    hc=metrics(yh,routed)
    eligible=bool(
        deltas["logloss"]<=-0.003
        and deltas["brier"]<=-0.0015
        and deltas["accuracy"]>=-0.005
        and stable["improved_logloss_ratio"]>=0.70
        and stable["improved_brier_ratio"]>=0.70
    )
    hold_delta={k:float(hc[k]-hb[k]) for k in ("accuracy","logloss","brier")}
    hold_delta["ece"]=float(hc["calibration_error"]-hb["calibration_error"])
    return {
        "status":"OK","schema_version":1,"research_only":True,"production_changed":False,
        "strict_point_in_time_archive_replay":False,"promotion_evidence_eligible":False,
        "horizon":horizon,"n":len(rows),"development_n":len(development),"final_holdout_n":len(holdout),
        "policy":"high_risk_q75_from_prior_meta_prediction_distribution; rescue_labels_from_prior_meta_block; fixed_threshold_0.60",
        "development":{"blocks":len(blocks),"baseline":base,"candidate":cand,"delta":deltas,"stability":stable},
        "routing":{"high_risk_quantile":HIGH_RISK_QUANTILE,"rescue_threshold":RESCUE_THRESHOLD,
                   "mean_route_coverage":float(np.mean([b["route_coverage"] for b in blocks])),
                   "mean_high_risk_coverage":float(np.mean([b["risk_coverage"] for b in blocks]))},
        "final_holdout":{"protected":True,"used_for_selection":False,"baseline":hb,"candidate":hc,
                         "delta":hold_delta,
                         "route_coverage":float(use.mean()),"high_risk_coverage":float(high.mean()),
                         "high_risk_n":int(high.sum()),"risk_quantile":risk_q},
        "blocks":blocks,"eligibility":eligible,
    }

def names(probs):
    return tuple(probs.keys())

def main():
    payload={"schema_version":1,"research_only":True,"production_changed":False,
             "promotion_evidence_eligible":False,
             "horizons":{h:_evaluate(h) for h in HORIZONS}}
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps(payload,indent=2,sort_keys=True))

if __name__=="__main__":
    main()
