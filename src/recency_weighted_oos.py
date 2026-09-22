"""Research-only recency-weighted BTC OOS evaluation.

Hyperparameters (half-life) are chosen only on an inner chronological
validation slice of each training window, then frozen for the unseen OOS block.
The final holdout is never used for selection.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import joblib
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/"src"
for p in (ROOT,SRC):
    if str(p) not in sys.path: sys.path.insert(0,str(p))

from model_compare import HORIZONS, load_primary_production_strict_rows, metrics, _temperature, aligned, apply_temperature, CLASSES
try:
    from lightgbm import LGBMClassifier
except ImportError:
    LGBMClassifier=None
from bootstrap_train import make_features
from binance_history import binance_archive_rows
from label_policy import direction_from_return

OUT=ROOT/"data/historical_research/recency_weighted_oos.json"
MIN_TRAIN=2000
TEST_BLOCK=150
MAX_ROWS=7000
MAX_BLOCKS=12
HALF_LIVES=(150,300,600,1200)
PURGE={"5m":5,"10m":10}
EMBARGO={"5m":60,"10m":60}

def load_research_archive_rows(horizon: str, max_rows: int = MAX_ROWS):
    """Build a research-only recent-history cohort from contiguous Binance Vision candles."""
    steps = int(str(horizon).rstrip("m"))
    try:
        raw = binance_archive_rows(max(max_rows + 40, 12000))
    except Exception:
        return []
    out = []
    for i in range(30, len(raw) - steps):
        try:
            x = make_features(raw[: i + 1])
            if not np.isfinite(np.asarray(x, dtype=float)).all():
                continue
            future_return = float(raw[i + steps][4]) / float(raw[i][4]) - 1.0
            created = datetime.fromtimestamp(int(raw[i][0]) / 1000.0, timezone.utc)
            out.append({
                "id": f"archive:{raw[i][0]}:{horizon}",
                "created": created.isoformat(),
                "target": datetime.fromtimestamp(int(raw[i + steps][0]) / 1000.0, timezone.utc).isoformat(),
                "x": [float(v) for v in x],
                "y": direction_from_return(future_return),
            })
        except (IndexError, ValueError, FloatingPointError):
            continue
    return out[-max_rows:]

def factories():
    d={
      "logreg":lambda:Pipeline([("scale",StandardScaler()),("model",LogisticRegression(C=.25,max_iter=3000))]),
      "extra_trees":lambda:ExtraTreesClassifier(n_estimators=260,max_depth=12,min_samples_leaf=10,max_features="sqrt",random_state=42,n_jobs=-1),
      "hgb":lambda:HistGradientBoostingClassifier(max_iter=200,max_leaf_nodes=15,learning_rate=.04,l2_regularization=1.5,random_state=42),
    }
    if LGBMClassifier is not None:
        d["lightgbm"]=lambda:LGBMClassifier(objective="multiclass",num_class=3,n_estimators=240,num_leaves=15,learning_rate=.035,min_child_samples=24,reg_lambda=2.0,subsample=.9,colsample_bytree=.9,random_state=42,n_jobs=-1,verbosity=-1)
    return d

def decay_weights(n, half_life):
    idx=np.arange(n,dtype=float)
    age=(n-1)-idx
    w=np.exp(np.log(0.5)*age/max(float(half_life),1e-9))
    w/=max(float(w.mean()),1e-12)
    return w

def _fit(model,X,y,weights=None):
    train_y=np.asarray(y)
    if XGBClassifier is not None and isinstance(model,XGBClassifier):
        train_y=np.asarray([CLASSES.index(str(v)) for v in y],dtype=int)
    if weights is None:
        model.fit(X,train_y)
    elif isinstance(model,Pipeline):
        model.fit(X,train_y,model__sample_weight=weights)
    else:
        model.fit(X,train_y,sample_weight=weights)
    return model

def _predict(model,rows):
    return aligned(model,np.asarray([r["x"] for r in rows],dtype=float))

def choose_half_life(factory,train):
    if len(train)<600:return None
    split=int(len(train)*.80)
    if split<400 or len(train)-split<100:return None
    Xtr=np.asarray([r["x"] for r in train[:split]],float)
    ytr=np.asarray([r["y"] for r in train[:split]])
    Xv=np.asarray([r["x"] for r in train[split:]],float)
    yv=[r["y"] for r in train[split:]]
    best=None
    for hl in HALF_LIVES:
        model=_fit(factory(),Xtr,ytr,decay_weights(len(ytr),hl))
        p=model.predict_proba(Xv)
        # Align manually because labels are strings.
        aligned_p=np.full((len(Xv),3),1e-7)
        for j,c in enumerate(model.classes_):
            if str(c) in CLASSES: aligned_p[:,CLASSES.index(str(c))]=p[:,j]
        aligned_p=np.clip(aligned_p,1e-7,1.0); aligned_p/=aligned_p.sum(axis=1,keepdims=True)
        m=metrics(yv,aligned_p)
        key=(m["logloss"],m["brier"],hl)
        if best is None or key<best[0]:best=(key,hl)
    return int(best[1]) if best else None

def _endpoints(n):
    raw=list(range(MIN_TRAIN,n,TEST_BLOCK))
    if len(raw)<=MAX_BLOCKS:return raw
    return sorted(set(int(x) for x in np.linspace(raw[0],raw[-1],MAX_BLOCKS)))

def evaluate(horizon):
    rows=load_primary_production_strict_rows(horizon)
    data_source="live_binance_primary"
    if len(rows)<MIN_TRAIN+TEST_BLOCK+100:
        archive_rows=load_research_archive_rows(horizon)
        if len(archive_rows)>len(rows):
            rows=archive_rows
            data_source="binance_vision_archive"
    if len(rows)>MAX_ROWS: rows=rows[-MAX_ROWS:]
    if len(rows)<MIN_TRAIN+TEST_BLOCK+100:return {"status":"DEFERRED","n":len(rows),"reason":"insufficient_research_rows","data_source":data_source}
    split=int(len(rows)*.80); development=rows[:split]; holdout=rows[split:]
    names=list(factories())
    blocks=[]
    for end in _endpoints(len(development)):
        train_end=max(0,end-PURGE[horizon]-EMBARGO[horizon])
        train=development[:train_end]; test=development[end:min(end+TEST_BLOCK,len(development))]
        if len(train)<MIN_TRAIN or len(test)<TEST_BLOCK//2:continue
        block={}
        y=[r["y"] for r in test]
        # Baseline: unweighted model zoo equal-probability mix.
        base_parts=[]
        decay_parts=[]
        selected={}
        for name in names:
            f=factories()[name]
            hl=choose_half_life(f,train)
            selected[name]=hl
            base=_fit(f(),np.asarray([r["x"] for r in train],float),np.asarray([r["y"] for r in train]))
            base_parts.append(_predict(base,test))
            weighted=_fit(f(),np.asarray([r["x"] for r in train],float),np.asarray([r["y"] for r in train]),decay_weights(len(train),hl) if hl else None)
            decay_parts.append(_predict(weighted,test))
        base_mix=np.mean(np.stack(base_parts,axis=0),axis=0)
        decay_mix=np.mean(np.stack(decay_parts,axis=0),axis=0)
        bm=metrics(y,base_mix); dm=metrics(y,decay_mix)
        block={"n":len(test),"selected_half_lives":selected,"baseline":bm,"recency":dm,"delta":{"accuracy":dm["accuracy"]-bm["accuracy"],"logloss":dm["logloss"]-bm["logloss"],"brier":dm["brier"]-bm["brier"]}}
        blocks.append(block)
    if len(blocks)<8:return {"status":"DEFERRED","n":len(rows),"reason":"insufficient_valid_oos_blocks"}
    ll=np.asarray([b["delta"]["logloss"] for b in blocks]); br=np.asarray([b["delta"]["brier"] for b in blocks]); ac=np.asarray([b["delta"]["accuracy"] for b in blocks])
    # Frozen development protocol applied exactly once to final holdout.
    y_hold=[r["y"] for r in holdout]
    base_parts=[]; decay_parts=[]; selected={}
    fs=factories()
    for name,f in fs.items():
        hl=choose_half_life(f,development); selected[name]=hl
        base=_fit(f(),np.asarray([r["x"] for r in development],float),np.asarray([r["y"] for r in development]))
        base_parts.append(_predict(base,holdout))
        weighted=_fit(f(),np.asarray([r["x"] for r in development],float),np.asarray([r["y"] for r in development]),decay_weights(len(development),hl) if hl else None)
        decay_parts.append(_predict(weighted,holdout))
    bh=np.mean(np.stack(base_parts,axis=0),axis=0); dh=np.mean(np.stack(decay_parts,axis=0),axis=0)
    return {
      "status":"OK","schema_version":1,"research_only":True,"production_changed":False,
      "final_holdout_protected":True,"final_holdout_used_for_selection":False,
      "policy":"inner_chronological_half_life_selection_only; final_holdout_descriptive_only",
      "models":list(fs),
      "summary":{
        "blocks":len(blocks),"samples":int(sum(b["n"] for b in blocks)),
        "mean_accuracy_delta":float(ac.mean()),"mean_logloss_delta":float(ll.mean()),"mean_brier_delta":float(br.mean()),
        "improved_logloss_ratio":float(np.mean(ll<0)),"improved_brier_ratio":float(np.mean(br<0)),
        "non_worse_accuracy_ratio":float(np.mean(ac>=-.005))
      },
      "final_holdout_n":len(holdout),
      "data_source":data_source,
      "promotion_evidence_eligible": data_source == "live_binance_primary",
      "final_holdout":{
        "baseline":metrics(y_hold,bh),"recency":metrics(y_hold,dh),
        "delta":{
          "accuracy":metrics(y_hold,dh)["accuracy"]-metrics(y_hold,bh)["accuracy"],
          "logloss":metrics(y_hold,dh)["logloss"]-metrics(y_hold,bh)["logloss"],
          "brier":metrics(y_hold,dh)["brier"]-metrics(y_hold,bh)["brier"]
        },
        "selected_half_lives":selected
      },
      "blocks":blocks
    }

def main():
    payload={"schema_version":1,"research_only":True,"production_changed":False,"horizons":{h:evaluate(h) for h in HORIZONS}}
    OUT.parent.mkdir(parents=True,exist_ok=True); OUT.write_text(json.dumps(payload,indent=2,sort_keys=True),encoding="utf-8"); print(json.dumps(payload,indent=2,sort_keys=True))
if __name__=="__main__":main()
