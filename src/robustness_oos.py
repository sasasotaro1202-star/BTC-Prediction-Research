"""Research-only chronological robustness diagnostics for BTC production probabilities.

No model artifacts or production state are changed. Regimes use only cutoff-available
features already stored with each prediction.
"""
from __future__ import annotations
import json, math, sqlite3
from pathlib import Path
import numpy as np
from sklearn.metrics import log_loss

ROOT=Path(__file__).resolve().parents[1]
DB=ROOT/"data"/"predictions.db"
OUT=ROOT/"data"/"historical_research"/"robustness_oos_report.json"
CLASSES=("DOWN","FLAT","UP")
HORIZONS={"5m":"actual_direction_5m","10m":"actual_direction_10m"}
FEATURES=("ret_10m","volatility_10m")

def _metrics(y, p):
    if len(y)<1: return None
    idx={c:i for i,c in enumerate(CLASSES)}
    yy=np.array([idx[v] for v in y],dtype=int)
    pp=np.clip(np.asarray(p,float),1e-6,1.0)
    pp=pp/pp.sum(axis=1,keepdims=True)
    pred=pp.argmax(1)
    one=np.eye(3)[yy]
    return {
        "n":int(len(y)),
        "accuracy":float((pred==yy).mean()),
        "logloss":float(log_loss(yy,pp,labels=[0,1,2])),
        "brier":float(np.mean(np.sum((pp-one)**2,axis=1))),
    }

def _regimes(rows):
    # Expanding median volatility: no future observations are used to define the
    # high/low volatility boundary at each prediction cutoff.
    vols=[]; out=[]
    for r in rows:
        v=float(r["vol"])
        vols.append(v)
        med=float(np.median(vols))
        trend="UP_MOMENTUM" if r["ret"]>0 else "DOWN_MOMENTUM" if r["ret"]<0 else "FLAT_MOMENTUM"
        vol="HIGH_VOL" if v>=med else "LOW_VOL"
        out.append(f"{trend}|{vol}")
    return out

def load(h):
    actual=HORIZONS[h]
    with sqlite3.connect(DB) as con:
        rows=con.execute(f"""
          SELECT prediction_id,created_at_utc,feature_json,{actual},
                 p_up_{h},p_down_{h},p_flat_{h}
          FROM predictions
          WHERE {actual} IS NOT NULL
          ORDER BY created_at_utc,prediction_id
        """).fetchall()
    result=[]
    for r in rows:
        try:
            f=json.loads(r[2] or "{}")
            vals=[float(f[k]) for k in FEATURES]
            probs=[float(r[4]),float(r[5]),float(r[6])]
            if not all(math.isfinite(x) for x in vals+probs): continue
            if r[3] not in CLASSES or min(probs)<0 or sum(probs)<=0: continue
            result.append({"id":r[0],"created":r[1],"ret":vals[0],"vol":vals[1],"y":r[3],"p":probs})
        except (TypeError,ValueError,KeyError,json.JSONDecodeError): continue
    return result

def evaluate(h, rows):
    n=len(rows)
    if n<1000:
        return {"status":"insufficient_data","n":n,"minimum":1000}
    split=int(n*0.8)
    development=rows[:split]
    holdout=rows[split:]
    regimes=_regimes(rows)
    # Regime thresholds are derived chronologically over the complete feature
    # stream, but each threshold only uses observations through that row.
    result={"status":"ok","n":n,"development_n":len(development),"final_holdout_n":len(holdout),
            "final_holdout_protected":True}
    result["development"]=_metrics([r["y"] for r in development],[r["p"] for r in development])
    result["final_holdout"]=_metrics([r["y"] for r in holdout],[r["p"] for r in holdout])
    result["regimes"]={}
    for regime in sorted(set(regimes)):
        idx=[i for i,x in enumerate(regimes) if x==regime]
        dev=[i for i in idx if i<split]; hold=[i for i in idx if i>=split]
        entry={"all_n":len(idx)}
        if len(dev)>=100:
            entry["development"]=_metrics([rows[i]["y"] for i in dev],[rows[i]["p"] for i in dev])
        else: entry["development"]={"status":"insufficient_data","n":len(dev),"minimum":100}
        if len(hold)>=100:
            entry["final_holdout"]=_metrics([rows[i]["y"] for i in hold],[rows[i]["p"] for i in hold])
        else: entry["final_holdout"]={"status":"insufficient_data","n":len(hold),"minimum":100}
        result["regimes"][regime]=entry
    return result

def main():
    if not DB.exists(): raise SystemExit("prediction database missing")
    payload={"schema_version":1,"research_only":True,"policy":"diagnostic_only_no_model_input_no_promotion_effect","horizons":{}}
    for h in HORIZONS:
        payload["horizons"][h]=evaluate(h,load(h))
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(payload,indent=2,sort_keys=True),encoding="utf-8")
    print(json.dumps(payload,indent=2))

if __name__=="__main__": main()
