"""Research-only train-window grid for rolling two-memory BTC 10m prediction.

The candidate is a fixed 50/50 blend of:
- frozen production RF (long-memory);
- retrained RF fit on a recent causal window.

Only the recent training-window length is selected, from a small preregistered
grid, using development chronological OOS. Adaptive holdout is untouched by
selection, and the last 10% is a true frozen blind evaluation: one selected
window/model and one fixed blend weight are frozen before the blind period.

Binance Vision archive publication timing is not equivalent to live PIT, so this
is research-only and promotion-ineligible.
"""
from __future__ import annotations

import json, math
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss

from binance_history import binance_archive_rows
from bootstrap_train import make_features
from feature_schema import FEATURES
from label_policy import CLASSES, NEUTRAL_RETURN

ROOT=Path(__file__).resolve().parents[1]
MODEL_DIR=ROOT/"models"
OUT=ROOT/"data"/"historical_research"/"rolling_window_grid_10m_oos.json"

HORIZON="10m"
STEPS=10
TARGET_ROWS=50_000
WINDOW_GRID=(4_000,6_000,8_000)
BLEND_WEIGHT_GRID=(0.25,0.40,0.50,0.60,0.75)
TEST_BLOCK=500
GAP_BARS=10
DEV_FRAC=0.75
ADAPT_FRAC=0.15
BLIND_FRAC=0.10
MIN_DEV_ROWS=10_000
MIN_BLOCKS=8
RF_TREES=200
EPS=1e-8


def _dt(value:str)->datetime:
    return datetime.fromisoformat(str(value).replace("Z","+00:00"))


def _norm(p:np.ndarray)->np.ndarray:
    p=np.asarray(p,dtype=float)
    if p.ndim==1:p=p[None,:]
    p=np.clip(p,EPS,1.0)
    s=p.sum(axis=1,keepdims=True)
    if not np.isfinite(p).all() or np.any(s<=0):raise ValueError("invalid_probability_matrix")
    return p/s


def _align(model,x:np.ndarray)->np.ndarray:
    raw=np.asarray(model.predict_proba(x),dtype=float)
    out=np.full((len(x),3),EPS,dtype=float)
    for i,cls in enumerate(model.classes_):
        name=str(cls)
        if name in CLASSES:out[:,CLASSES.index(name)]=raw[:,i]
    return _norm(out)


def _metrics(y:list[str],p:np.ndarray)->dict[str,float|int]:
    yi=np.asarray([CLASSES.index(v) for v in y],dtype=int)
    p=_norm(p); pred=p.argmax(axis=1); hit=pred==yi; conf=p.max(axis=1)
    ece=0.0
    for k in range(10):
        lo,hi=k/10.0,(k+1)/10.0
        mask=(conf>=lo)&((conf<hi) if hi<1.0 else (conf<=hi))
        if np.any(mask):
            ece+=float(mask.mean())*abs(float(hit[mask].mean())-float(conf[mask].mean()))
    return {
        "n":int(len(y)),
        "accuracy":float(hit.mean()),
        "logloss":float(log_loss(yi,p,labels=[0,1,2])),
        "brier":float(np.mean(np.sum((p-np.eye(3)[yi])**2,axis=1))),
        "ece":float(ece),
    }


def _rf()->RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=RF_TREES,max_depth=10,min_samples_leaf=10,
        max_features="sqrt",random_state=42,n_jobs=-1
    )


def _dataset(rows:list[list[float]],trained_at:datetime):
    x,y,ts=[],[],[]
    for i in range(30,len(rows)-STEPS):
        created=datetime.fromtimestamp(int(rows[i][0])/1000.0,timezone.utc)
        if created<=trained_at:continue
        try:feat=make_features(rows[i-29:i+1])
        except Exception:continue
        if len(feat)!=len(FEATURES) or not all(math.isfinite(float(v)) for v in feat):continue
        future_return=float(rows[i+STEPS][4])/float(rows[i][4])-1.0
        y.append("UP" if future_return>NEUTRAL_RETURN else "DOWN" if future_return<-NEUTRAL_RETURN else "FLAT")
        x.append(feat); ts.append(created)
    return (np.asarray(x,dtype=float),y,ts) if x else (np.empty((0,len(FEATURES))),[],[])


def _blocks(x,y,ts,champion,start,end,window):
    out=[]; first=max(start,window+GAP_BARS)
    for test_start in range(first,end,TEST_BLOCK):
        test_end=min(test_start+TEST_BLOCK,end)
        if test_end-test_start<TEST_BLOCK:continue
        train_end=test_start-GAP_BARS
        train_start=max(0,train_end-window)
        if train_end-train_start<window:continue
        train_y=np.asarray(y[train_start:train_end])
        if len(set(train_y))<3:continue
        model=_rf(); model.fit(x[train_start:train_end],train_y)
        recent=_align(model,x[test_start:test_end])
        frozen=_align(champion,x[test_start:test_end])
        block_y=y[test_start:test_end]
        out.append({
            "start":timestamps(ts[test_start]),
            "end":timestamps(ts[test_end-1]),
            "n":test_end-test_start,
            "y":block_y,
            "frozen":frozen,
            "recent":recent,
        })
    return out


def timestamps(value):return value.isoformat() if hasattr(value,"isoformat") else str(value)


def _blend(frozen:np.ndarray,recent:np.ndarray,weight:float)->np.ndarray:
    if not 0.0 <= float(weight) <= 1.0:
        raise ValueError("invalid_blend_weight")
    return _norm((1.0-float(weight))*np.asarray(frozen,dtype=float)+float(weight)*np.asarray(recent,dtype=float))


def _evaluate_blocks(blocks,weight=0.5):
    ys=[]; bp=[]; rp=[]; cp=[]; block_deltas=[]
    for block in blocks:
        y=block["y"]; frozen=np.asarray(block["frozen"],dtype=float); recent=np.asarray(block["recent"],dtype=float)
        candidate=_blend(frozen,recent,weight)
        bm=_metrics(y,frozen); cm=_metrics(y,candidate)
        ys.extend(y); bp.extend(frozen.tolist()); rp.extend(recent.tolist()); cp.extend(candidate.tolist())
        block_deltas.append({k:float(cm[k]-bm[k]) for k in ("accuracy","logloss","brier","ece")})
    baseline=_metrics(ys,np.asarray(bp)); recent=_metrics(ys,np.asarray(rp)); candidate=_metrics(ys,np.asarray(cp))
    return {
        "baseline":baseline,
        "recent":recent,
        "candidate":candidate,
        "blocks":len(blocks),
        "block_deltas":block_deltas,
        "delta":{k:float(candidate[k]-baseline[k]) for k in ("accuracy","logloss","brier","ece")}
    }


def _aggregate_blocks(blocks,weight=0.5):
    return _evaluate_blocks(blocks,weight)


def evaluate():
    meta=json.loads((MODEL_DIR/"10m.json").read_text(encoding="utf-8"))
    trained_at=_dt(meta["trained_at_utc"])
    rows=binance_archive_rows(TARGET_ROWS)
    x,y,ts=_dataset(rows,trained_at)
    if len(y)<MIN_DEV_ROWS+2000:
        return {"status":"DEFERRED","reason":"insufficient_future_rows","n":len(y),"research_only":True,"production_changed":False,"strict_pit":False,"promotion_evidence_eligible":False}
    n=len(y)
    dev_end=int(round(n*DEV_FRAC))
    adapt_end=int(round(n*(DEV_FRAC+ADAPT_FRAC)))
    blind_start=adapt_end
    if dev_end<MIN_DEV_ROWS or blind_start>=n:
        raise ValueError("invalid_split")
    champion=joblib.load(MODEL_DIR/"10m.joblib")

    trial_results={}
    dev_blocks_by_window={}
    selection_candidates=[]
    for window in WINDOW_GRID:
        blocks=_blocks(x,y,ts,champion,0,dev_end,window)
        dev_blocks_by_window[window]=blocks
        if len(blocks)<MIN_BLOCKS:
            trial_results[str(window)]={"status":"DEFERRED","blocks":len(blocks)}
            continue
        weight_trials={}
        for weight in BLEND_WEIGHT_GRID:
            agg=_aggregate_blocks(blocks,weight)
            key=f"{weight:.2f}"
            weight_trials[key]={
                "weight":weight,
                "accuracy":agg["candidate"]["accuracy"],
                "logloss":agg["candidate"]["logloss"],
                "brier":agg["candidate"]["brier"],
                "ece":agg["candidate"]["ece"],
                "delta_accuracy":agg["delta"]["accuracy"],
                "delta_logloss":agg["delta"]["logloss"],
                "delta_brier":agg["delta"]["brier"],
                "delta_ece":agg["delta"]["ece"],
            }
            selection_candidates.append((
                float(agg["candidate"]["logloss"]),
                float(agg["candidate"]["brier"]),
                -float(agg["candidate"]["accuracy"]),
                window,
                weight,
            ))
        trial_results[str(window)]={"status":"OK","window":window,"blocks":len(blocks),"weights":weight_trials}

    if not selection_candidates:
        return {"status":"DEFERRED","reason":"no_window_has_enough_oos_blocks","n":n,"trials":trial_results,"research_only":True,"production_changed":False,"strict_pit":False,"promotion_evidence_eligible":False}
    _,_,_,selected_window,selected_weight=min(selection_candidates)
    selected_dev=_aggregate_blocks(dev_blocks_by_window[selected_window],selected_weight)

    # Adaptive holdout: selected window and blend weight are frozen, no tuning.
    adapt_blocks=_blocks(x,y,ts,champion,dev_end,adapt_end,selected_window)
    adapt_result=_aggregate_blocks(adapt_blocks,selected_weight) if adapt_blocks else None

    # True final blind: train exactly one recent model using only observations
    # before blind start. Freeze both model and selected blend weight through blind.
    train_end=blind_start-GAP_BARS
    train_start=max(0,train_end-selected_window)
    if train_end-train_start<selected_window or len(set(y[train_start:train_end]))<3:
        raise ValueError("blind_training_window_invalid")
    blind_model=_rf()
    blind_model.fit(x[train_start:train_end],np.asarray(y[train_start:train_end]))
    blind_recent=_align(blind_model,x[blind_start:])
    blind_frozen=_align(champion,x[blind_start:])
    blind_candidate=_blend(blind_frozen,blind_recent,selected_weight)
    blind_y=y[blind_start:]
    blind_b=_metrics(blind_y,blind_frozen)
    blind_c=_metrics(blind_y,blind_candidate)

    selected_block_deltas=selected_dev["block_deltas"]
    ll_d=[float(b["logloss"]) for b in selected_block_deltas]
    br_d=[float(b["brier"]) for b in selected_block_deltas]
    ac_d=[float(b["accuracy"]) for b in selected_block_deltas]
    ece_d=[float(b["ece"]) for b in selected_block_deltas]
    base=selected_dev["baseline"]; cand=selected_dev["candidate"]
    ll_gain=(base["logloss"]-cand["logloss"])/max(abs(base["logloss"]),EPS)
    br_gain=(base["brier"]-cand["brier"])/max(abs(base["brier"]),EPS)
    eligible=bool(
        ll_gain>=0.03 and br_gain>=0.01
        and np.mean(np.asarray(ll_d)<0)>=0.70
        and np.mean(np.asarray(br_d)<0)>=0.70
        and np.mean(np.asarray(ac_d)>=-0.005)>=0.70
        and np.mean(np.asarray(ece_d)<=0)>=0.70
        and blind_c["accuracy"]>=blind_b["accuracy"]-0.005
        and blind_c["logloss"]<=blind_b["logloss"]
        and blind_c["brier"]<=blind_b["brier"]
    )

    return {
        "status":"OK","schema_version":1,"research_only":True,"production_changed":False,
        "strict_pit":False,"promotion_evidence_eligible":False,
        "archive_publication_time_unknown":True,
        "horizon":HORIZON,"production_model_version":meta.get("model_version"),
        "model_version_under_test":"rolling_window_grid_weighted_blend",
        "trained_at_utc":meta.get("trained_at_utc"),
        "features":list(FEATURES),
        "config":{
            "window_grid":list(WINDOW_GRID),"blend_weight_grid":list(BLEND_WEIGHT_GRID),
            "selected_window":selected_window,"selected_blend_weight_recent":selected_weight,
            "test_block":TEST_BLOCK,"gap_bars":GAP_BARS,
            "rf_trees":RF_TREES,
            "dev_frac":DEV_FRAC,"adaptive_holdout_frac":ADAPT_FRAC,"blind_frac":BLIND_FRAC
        },
        "n":n,"development_n":dev_end,
        "adaptive_holdout_n":adapt_end-dev_end,"final_blind_n":n-blind_start,
        "window_selection":{"selected_window":selected_window,"selected_blend_weight_recent":selected_weight,"selected_from":"development_oos_only","trials":trial_results},
        "development":{
            "blocks":len(dev_blocks_by_window[selected_window]),"baseline":base,"candidate":cand,"delta":selected_dev["delta"],
            "block_stability":{
                "improved_logloss_ratio":float(np.mean(np.asarray(ll_d)<0)),
                "improved_brier_ratio":float(np.mean(np.asarray(br_d)<0)),
                "non_worse_accuracy_ratio":float(np.mean(np.asarray(ac_d)>=-0.005)),
                "non_worse_ece_ratio":float(np.mean(np.asarray(ece_d)<=0)),
            }
        },
        "adaptive_holdout":{
            "blocks":adapt_result["blocks"] if adapt_result else 0,
            "baseline":adapt_result["baseline"] if adapt_result else None,
            "candidate":adapt_result["candidate"] if adapt_result else None,
            "delta":adapt_result["delta"] if adapt_result else None,
            "used_for_selection":False
        },
        "final_blind":{
            "protected":True,"used_for_selection":False,
            "selected_window":selected_window,"blend_weight_recent":selected_weight,
            "baseline":blind_b,"candidate":blind_c,
            "delta":{k:float(blind_c[k]-blind_b[k]) for k in ("accuracy","logloss","brier","ece")}
        },
        "eligibility":eligible
    }


def main():
    payload={"schema_version":1,"research_only":True,"production_changed":False,"strict_pit":False,"promotion_evidence_eligible":False,"horizons":{"10m":evaluate()}}
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(payload,indent=2,sort_keys=True),encoding="utf-8")
    print(json.dumps(payload,indent=2,sort_keys=True))


if __name__=="__main__":main()
