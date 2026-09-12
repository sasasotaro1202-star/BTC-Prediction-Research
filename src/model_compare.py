import json
import math
import sqlite3
from datetime import datetime, timezone

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss

from db import DB, init_db

HORIZONS = {
    "5m": ("actual_direction_5m", "p_up_5m", "p_down_5m", "p_flat_5m"),
    "10m": ("actual_direction_10m", "p_up_10m", "p_down_10m", "p_flat_10m"),
}
FEATURES = ["ret_1m", "ret_3m", "ret_5m", "ret_10m", "volatility_10m", "volume_ratio"]
CLASSES = ["DOWN", "FLAT", "UP"]
MIN_SAMPLES = 50
MIN_TRAIN = 30
TEST_BLOCK = 20
MODEL_DIR = DB.parent / "models"


def now(): return datetime.now(timezone.utc).isoformat()


def safe_json(text):
    try: return json.loads(text)
    except Exception: return {}


def load_rows(horizon):
    actual_col, *_ = HORIZONS[horizon]
    with sqlite3.connect(DB) as con:
        rows = con.execute(
            f"SELECT prediction_id, created_at_utc, feature_json, {actual_col}, p_up_{horizon}, p_down_{horizon}, p_flat_{horizon}, model_version "
            "FROM predictions WHERE " + actual_col + " IS NOT NULL ORDER BY created_at_utc"
        ).fetchall()
    out=[]
    for r in rows:
        f=safe_json(r[2])
        if not all(k in f for k in FEATURES) or r[3] not in CLASSES: continue
        x=[float(f[k]) for k in FEATURES]
        if not all(math.isfinite(v) for v in x): continue
        out.append({"id":r[0],"created":r[1],"x":x,"y":r[3],"production":[float(r[4]),float(r[5]),float(r[6])],"model_version":r[7]})
    return out


def metrics(y_true, probs):
    idx={c:i for i,c in enumerate(CLASSES)}; y=np.array([idx[v] for v in y_true])
    p=np.clip(np.asarray(probs,dtype=float),1e-6,1-1e-6); p/=p.sum(axis=1,keepdims=True)
    pred=p.argmax(axis=1); accuracy=float((pred==y).mean())
    ll=float(log_loss(y,p,labels=list(range(3))))
    onehot=np.eye(3)[y]; brier=float(np.mean(np.sum((p-onehot)**2,axis=1)))
    conf=p.max(axis=1); hit=(pred==y).astype(float); ece=0.0
    for i in range(10):
        lo=i/10; hi=(i+1)/10
        mask=(conf>=lo)&((conf<hi) if hi<1 else (conf<=hi))
        if mask.any(): ece+=float(mask.mean())*abs(float(hit[mask].mean())-float(conf[mask].mean()))
    return {"accuracy":accuracy,"logloss":ll,"brier":brier,"calibration_error":ece}


def production_oos(rows): return metrics([r["y"] for r in rows],[r["production"] for r in rows])


def aligned_probs(model,X):
    p=model.predict_proba(X); classes=list(model.classes_); aligned=np.full((len(X),3),1e-6)
    for j,c in enumerate(classes): aligned[:,CLASSES.index(c)]=p[:,j]
    return aligned/aligned.sum(axis=1,keepdims=True)


def walk_forward(rows,factory):
    if len(rows)<MIN_SAMPLES: return None
    preds=[]; ys=[]
    for end in range(MIN_TRAIN,len(rows),TEST_BLOCK):
        train=rows[:end]; test=rows[end:min(end+TEST_BLOCK,len(rows))]
        if not test: break
        model=factory(); X=np.array([r["x"] for r in train]); y=np.array([r["y"] for r in train])
        if len(set(y))<2: continue
        model.fit(X,y); preds.extend(aligned_probs(model,np.array([r["x"] for r in test])).tolist()); ys.extend(r["y"] for r in test)
    if len(ys)<MIN_SAMPLES: return None
    return metrics(ys,preds)


def train_and_save(rows,horizon,name,factory):
    X=np.array([r["x"] for r in rows]); y=np.array([r["y"] for r in rows]); model=factory()
    if len(set(y))<2: return None
    model.fit(X,y); MODEL_DIR.mkdir(parents=True,exist_ok=True)
    joblib.dump(model,MODEL_DIR/f'{horizon}.joblib')
    meta={"model_version":name,"horizon":horizon,"classes":list(model.classes_),"features":FEATURES,"artifact":f'{horizon}.joblib'}
    (MODEL_DIR/f'{horizon}.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    return meta


def materially_better(candidate,production):
    return (candidate["accuracy"]>=production["accuracy"]-0.01 and
            candidate["logloss"]<=production["logloss"]-0.005 and
            candidate["brier"]<=production["brier"]-0.002 and
            candidate["calibration_error"]<=production["calibration_error"]+0.01)


def save_metric(horizon,version,n,m):
    with sqlite3.connect(DB) as con:
        con.execute("INSERT INTO model_metrics(evaluated_at_utc,horizon,model_version,n,accuracy,logloss,brier,calibration_error) VALUES(?,?,?,?,?,?,?,?)",(now(),horizon,version,n,m["accuracy"],m["logloss"],m["brier"],m["calibration_error"]))


def get_production_version(horizon):
    with sqlite3.connect(DB) as con:
        row=con.execute("SELECT production_version FROM model_registry WHERE horizon=?",(horizon,)).fetchone()
    return row[0] if row else "v1.0"


def set_production_version(horizon,version):
    with sqlite3.connect(DB) as con:
        con.execute("INSERT INTO model_registry(horizon,production_version,updated_at_utc) VALUES(?,?,?) ON CONFLICT(horizon) DO UPDATE SET production_version=excluded.production_version,updated_at_utc=excluded.updated_at_utc",(horizon,version,now()))


def compare_horizon(horizon):
    rows=load_rows(horizon)
    if len(rows)<MIN_SAMPLES:
        print(f'{horizon}: insufficient OOS data ({len(rows)}/{MIN_SAMPLES})'); return {"status":"insufficient_data","n":len(rows)}
    production=production_oos(rows); save_metric(horizon,get_production_version(horizon),len(rows),production)
    candidates={
        'logreg_c0.1':lambda:LogisticRegression(C=0.1,max_iter=2000),
        'logreg_c1':lambda:LogisticRegression(C=1.0,max_iter=2000),
        'logreg_c10':lambda:LogisticRegression(C=10.0,max_iter=2000),
        'rf_300':lambda:RandomForestClassifier(n_estimators=300,max_depth=6,min_samples_leaf=5,random_state=42,n_jobs=-1),
    }
    results={}
    for name,factory in candidates.items():
        oos=walk_forward(rows,factory)
        if oos is not None: results[name]=oos; save_metric(horizon,name,len(rows),oos)
    eligible=[(name,m) for name,m in results.items() if materially_better(m,production)]
    if not eligible:
        print(f'{horizon}: REJECT — no candidate passed adoption gate'); return {"status":"rejected","production":production,"candidates":results}
    winner,winner_metrics=min(eligible,key=lambda kv:(kv[1]["logloss"],kv[1]["brier"]))
    payload=train_and_save(rows,horizon,winner,candidates[winner])
    if payload is None:
        print(f'{horizon}: REJECT — training/deployment failed'); return {"status":"rejected_training","winner":winner}
    new_version=f'v2.{datetime.now(timezone.utc).strftime("%Y%m%d%H%M")}'; payload["model_version"]=new_version
    (MODEL_DIR/f'{horizon}.json').write_text(json.dumps(payload,indent=2),encoding='utf-8')
    set_production_version(horizon,new_version)
    print(f'{horizon}: ADOPT {new_version} from {winner}'); print('old',production); print('new',winner_metrics)
    return {"status":"adopted","version":new_version,"source":winner,"old":production,"new":winner_metrics}


def compare():
    init_db(); print(json.dumps({h:compare_horizon(h) for h in HORIZONS},indent=2,default=str))

if __name__=='__main__': compare()
