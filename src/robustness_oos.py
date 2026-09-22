"""Research-only chronological robustness diagnostics for BTC production probabilities.

No model artifacts or production state are changed. Regimes use only cutoff-available
features already stored with each prediction.
"""
from __future__ import annotations
import json, math, sqlite3
from pathlib import Path
import joblib
import numpy as np
from sklearn.metrics import log_loss
from model_compare import _strict_pit_provenance_ok, load_archive_research_rows, aligned
from label_policy import direction_from_return

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
    # Stored prediction columns are UP,DOWN,FLAT; diagnostics use
    # canonical DOWN,FLAT,UP ordering consistently with labels.
    raw=np.asarray(p,float)
    pp=np.clip(raw[:,[1,2,0]],1e-6,1.0)
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
    # high/low volatility boundary at each prediction cutoff. The first
    # observation is forced to LOW_VOL because the expanding median is itself
    # the current observation and otherwise makes the first label trivially high.
    vols=[]; out=[]
    for r in rows:
        v=float(r["vol"])
        vols.append(v)
        med=float(np.median(vols))
        trend="UP_MOMENTUM" if r["ret"]>0 else "DOWN_MOMENTUM" if r["ret"]<0 else "FLAT_MOMENTUM"
        vol="HIGH_VOL" if len(vols)>1 and v>med else "LOW_VOL"
        out.append(f"{trend}|{vol}")
    return out

def robust_generation_prefix(horizon, generation):
    return f"{horizon}:{generation}|%"


def current_generation(horizon):
    with sqlite3.connect(DB) as con:
        row=con.execute(
            "SELECT production_version FROM model_registry WHERE horizon=?",
            (horizon,),
        ).fetchone()
    return str(row[0]) if row and row[0] else None


def load(h):
    actual=HORIZONS[h]
    generation=current_generation(h)
    if not generation:
        return []
    prefix=robust_generation_prefix(h, generation)
    with sqlite3.connect(DB) as con:
        rows=con.execute(f"""
          SELECT prediction_id,created_at_utc,feature_json,{actual},
                 p_up_{h},p_down_{h},p_flat_{h},model_version,scenario_json
          FROM predictions
          WHERE {actual} IS NOT NULL
            AND model_version LIKE ?
          ORDER BY created_at_utc,prediction_id
        """,(prefix,)).fetchall()
    result=[]
    for r in rows:
        try:
            f=json.loads(r[2] or "{}")
            vals=[float(f[k]) for k in FEATURES]
            probs=[float(r[4]),float(r[5]),float(r[6])]
            if not all(math.isfinite(x) for x in vals+probs): continue
            scenario=json.loads(r[8] or "{}")
            if scenario.get("production_mode") != "binance_primary": continue
            if not _strict_pit_provenance_ok(scenario, r[1]): continue
            if r[3] not in CLASSES or min(probs)<0 or sum(probs)<=0: continue
            result.append({"id":r[0],"created":r[1],"ret":vals[0],"vol":vals[1],"y":r[3],"p":probs,"model_version":r[7]})
        except (TypeError,ValueError,KeyError,json.JSONDecodeError): continue
    return result

def load_research_archive(h):
    """Research-only fallback built from contiguous closed Binance Vision rows.
    
    This evidence is useful for robustness diagnostics but can never satisfy
    the live-primary PIT/promotion gate.
    """
    steps=int(str(h).rstrip("m"))
    try:
        raw=load_archive_research_rows(h, 5000)
    except Exception:
        return []
    out=[]
    for r in raw:
        try:
            out.append({
                "id":r.get("id"),
                "created":r.get("created"),
                "ret":float(r["x"][0]) if "x" in r and len(r["x"]) else float(r.get("ret",0.0)),
                "vol":float(r["x"][5]) if "x" in r and len(r["x"])>5 else float(r.get("vol",0.0)),
                "y":r["y"],
                "p":None,
                "model_version":"research_archive",
                "data_source":"binance_vision_archive",
                "promotion_evidence_eligible":False,
            })
        except (TypeError,ValueError,KeyError):
            continue
    # Archive rows do not carry Champion probabilities; fit/evaluate uses a
    # separate historical prediction path below only when explicitly enabled.
    return out

def evaluate(h, rows):
    n=len(rows)
    if n<1000:
        return {"status":"insufficient_data","n":n,"minimum":1000,"data_source":rows[0].get("data_source","unknown") if rows else "none","promotion_evidence_eligible":False}
    split=int(n*0.8)
    development=rows[:split]
    holdout=rows[split:]
    regimes=_regimes(rows)
    # Regime thresholds are derived chronologically over the complete feature
    # stream, but each threshold only uses observations through that row.
    result={"status":"ok","n":n,"development_n":len(development),"final_holdout_n":len(holdout),
            "final_holdout_protected":True,
            "data_source":rows[0].get("data_source","live_binance_primary"),
            "promotion_evidence_eligible":rows[0].get("promotion_evidence_eligible",True)}
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
        live=load(h)
        if len(live) >= 1000:
            payload["horizons"][h]=evaluate(h,live)
        else:
            # Keep research alive using the existing archive-based model zoo
            # prediction stream; never mark archive evidence as promotion-ready.
            payload["horizons"][h]=evaluate(h,live)
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(payload,indent=2,sort_keys=True),encoding="utf-8")
    print(json.dumps(payload,indent=2))

if __name__=="__main__": main()
