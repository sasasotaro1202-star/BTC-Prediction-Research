"""Bridge OOS: train on verified historical archive, blind-test on later strict-primary predictions.

Candidate selection uses only the historical archive. The later strict Binance-primary
prediction cohort is never used for model selection or threshold tuning and acts as an
external future-like test. This lane is research-only and never changes production.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
import sys

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/"src"
for p in (ROOT,SRC):
    if str(p) not in sys.path:
        sys.path.insert(0,str(p))

from binance_history import binance_archive_rows
from bootstrap_train import make_features
from label_policy import direction_from_return, CLASSES
from model_compare import (
    HORIZONS,
    MIN_OOS,
    aligned,
    metrics,
    apply_temperature,
    _temperature,
    load_primary_production_strict_rows,
)

OUT=ROOT/"data/historical_research/archive_primary_bridge_oos.json"
ARCHIVE_ROWS=60000
ARCHIVE_MIN=12000
FINAL_ARCHIVE_HOLDOUT_FRAC=0.15
GATE_FRAC=0.10
MIN_PRIMARY_ROWS=500

def _archive_dataset(horizon, cutoff=None):
    steps=int(horizon[:-1])
    raw=binance_archive_rows(ARCHIVE_ROWS)
    rows=[]
    for i in range(30,len(raw)-steps):
        cur=raw[i]; target=raw[i+steps]
        if int(target[0])-int(cur[0]) != steps*60_000:
            continue
        created=datetime.fromtimestamp(int(cur[0])/1000.0,timezone.utc)
        if cutoff is not None and created >= cutoff:
            continue
        x=np.asarray(make_features(raw[:i+1]),dtype=float)
        if x.shape!=(15,) or not np.isfinite(x).all():
            continue
        y=direction_from_return(float(target[4])/float(cur[4])-1.0)
        rows.append({"created":created.isoformat(),"x":x.tolist(),"y":y})
    return rows[-ARCHIVE_ROWS:]

def _factories():
    return {
        "logreg": lambda: Pipeline([
            ("scale",StandardScaler()),
            ("model",LogisticRegression(C=0.25,max_iter=3000)),
        ]),
        "random_forest": lambda: RandomForestClassifier(
            n_estimators=500,max_depth=10,min_samples_leaf=10,
            max_features="sqrt",random_state=42,n_jobs=-1
        ),
        "extra_trees": lambda: ExtraTreesClassifier(
            n_estimators=500,max_depth=12,min_samples_leaf=10,
            max_features="sqrt",random_state=42,n_jobs=-1
        ),
        "hgb": lambda: HistGradientBoostingClassifier(
            max_iter=240,max_leaf_nodes=15,learning_rate=0.04,
            l2_regularization=1.5,random_state=42
        ),
    }

def _fit_temperature(train, y_train):
    split=int(len(train)*0.80)
    if split<500 or len(train)-split<200:
        return 1.0
    model=_factories()["logreg"]()
    model.fit(np.asarray([r["x"] for r in train[:split]],float),np.asarray(y_train[:split]))
    p=aligned(model,np.asarray([r["x"] for r in train[split:]],float))
    return _temperature(p,y_train[split:])

def _fit_and_predict(factory, train, predict_rows, temperature=1.0):
    model=factory()
    model.fit(np.asarray([r["x"] for r in train],float),np.asarray([r["y"] for r in train]))
    p=aligned(model,np.asarray([r["x"] for r in predict_rows],float))
    return apply_temperature(p,temperature),model

def _choose_archive_candidate(dataset,horizon):
    n=len(dataset)
    hold_start=int(n*(1.0-FINAL_ARCHIVE_HOLDOUT_FRAC))
    gate_start=int(n*(1.0-FINAL_ARCHIVE_HOLDOUT_FRAC-GATE_FRAC))
    train=dataset[:int(n*(1.0-FINAL_ARCHIVE_HOLDOUT_FRAC-GATE_FRAC))]
    gate=dataset[gate_start:hold_start]
    holdout=dataset[hold_start:]
    candidates=[]
    alpha=0.05/max(1,len(_factories()))
    for name,factory in _factories().items():
        temp=_fit_temperature(train,[r["y"] for r in train])
        gp,model=_fit_and_predict(factory,train,gate,temp)
        gm=metrics([r["y"] for r in gate],gp)
        candidates.append({"name":name,"temperature":temp,"gate":gm,"model":model})
    selected=min(candidates,key=lambda c:(c["gate"]["logloss"],c["gate"]["brier"],-c["gate"]["accuracy"]))
    # Archive holdout is frozen/descriptive only. It is never used to choose the candidate.
    hp,_=_fit_and_predict(_factories()[selected["name"]],train+gate,holdout,selected["temperature"])
    hm=metrics([r["y"] for r in holdout],hp)
    return selected, candidates, train, gate, holdout, hm, alpha

def evaluate(horizon):
    primary=load_primary_production_strict_rows(horizon)
    if len(primary)<MIN_PRIMARY_ROWS:
        return {"status":"DEFERRED","reason":"insufficient_strict_primary_rows","n_primary":len(primary),"promotion_evidence_eligible":False}
    primary=sorted(primary,key=lambda r:(str(r["created"]),int(r["id"])))
    earliest=datetime.fromisoformat(str(primary[0]["created"]).replace("Z","+00:00"))
    archive=_archive_dataset(horizon,cutoff=earliest)
    if len(archive)<ARCHIVE_MIN:
        return {"status":"DEFERRED","reason":"insufficient_pre_primary_archive_rows","n_primary":len(primary),"n_archive":len(archive),"promotion_evidence_eligible":False}
    selected,candidates,train,gate,archive_hold,archive_hold_metrics,alpha=_choose_archive_candidate(archive,horizon)
    factory=_factories()[selected["name"]]
    bridge_train=train+gate
    live_rows=[{
        "id":r["id"],"created":r["created"],"x":r["x"],"y":r["y"],"production":r["production"]
    } for r in primary]
    live_p,_=_fit_and_predict(factory,bridge_train,live_rows,selected["temperature"])
    y_live=[r["y"] for r in live_rows]
    prod_p=[r["production"] for r in live_rows]
    cm=metrics(y_live,live_p)
    pm=metrics(y_live,prod_p)
    block=[]
    for start in range(0,len(y_live),250):
        ys=y_live[start:start+250]
        if len(ys)<125: continue
        ca=metrics(ys,live_p[start:start+250])
        ba=metrics(ys,prod_p[start:start+250])
        block.append({
            "n":len(ys),
            "accuracy_delta":ca["accuracy"]-ba["accuracy"],
            "logloss_delta":ca["logloss"]-ba["logloss"],
            "brier_delta":ca["brier"]-ba["brier"],
        })
    ll=np.asarray([b["logloss_delta"] for b in block],float)
    br=np.asarray([b["brier_delta"] for b in block],float)
    ac=np.asarray([b["accuracy_delta"] for b in block],float)
    eligible=bool(
        len(block)>=8
        and float(np.mean(ll<0))>=0.60
        and float(np.mean(br<0))>=0.60
        and cm["logloss"]<=pm["logloss"]-0.003
        and cm["brier"]<=pm["brier"]-0.0015
        and cm["accuracy"]>=pm["accuracy"]-0.005
    )
    return {
        "status":"OK",
        "schema_version":1,
        "research_only":True,
        "production_changed":False,
        "promotion_evidence_eligible":True,
        "final_holdout_protected":True,
        "external_strict_primary_test":True,
        "final_live_test_used_for_selection":False,
        "horizon":horizon,
        "archive_n":len(archive),
        "primary_n":len(live_rows),
        "primary_earliest_created":primary[0]["created"],
        "archive_latest_created":archive[-1]["created"],
        "archive_selection":{
            "selected_candidate":selected["name"],
            "selected_temperature":float(selected["temperature"]),
            "gate":selected["gate"],
            "archive_holdout_metrics":archive_hold_metrics,
        },
        "candidates":{
            c["name"]:{
                "temperature":float(c["temperature"]),
                "gate":c["gate"],
            } for c in candidates
        },
        "live_primary":{
            "candidate":cm,
            "production":pm,
            "delta":{
                "accuracy":cm["accuracy"]-pm["accuracy"],
                "logloss":cm["logloss"]-pm["logloss"],
                "brier":cm["brier"]-pm["brier"],
            },
            "block_stability":{
                "blocks":len(block),
                "improved_logloss_ratio":float(np.mean(ll<0)) if len(ll) else 0.0,
                "improved_brier_ratio":float(np.mean(br<0)) if len(br) else 0.0,
                "non_worse_accuracy_ratio":float(np.mean(ac>=-0.005)) if len(ac) else 0.0,
            },
            "eligible_pending_frozen_holdout_confirmation":eligible,
        },
    }

def main():
    payload={
        "schema_version":1,
        "research_only":True,
        "production_changed":False,
        "policy":"archive_candidate_selection_then_external_future_strict_primary_blind_test",
        "horizons":{h:evaluate(h) for h in HORIZONS},
        "generated_at_utc":datetime.now(timezone.utc).isoformat(),
    }
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(payload,indent=2,sort_keys=True),encoding="utf-8")
    print(json.dumps(payload,indent=2,sort_keys=True))

if __name__=="__main__":
    main()
