"""Research-only online expert aggregation on archived chronological OOS predictions.

Each expert row is already an unseen historical prediction. At time t, weights
are computed only from losses observed before t. The final 20% is frozen and
used only descriptively after strategy configuration is fixed.
"""
from __future__ import annotations
import csv, json, math
from pathlib import Path
from typing import Any
import numpy as np

CLASSES=("DOWN","FLAT","UP")
ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/"data"/"historical_research"
OUT=DATA/"archive_online_hedge_oos.json"
EXPERTS=("logreg","extra","rf","hgb","ensemble")
HOLDOUT_FRAC=0.20
WARMUP=100
ETA=4.0
EPS=1e-8
BLOCK=250


def load_expert(h: str, expert: str) -> dict[int, tuple[str,np.ndarray]]:
    path=DATA/f"oos_{h}_{expert}.csv"
    if not path.is_file():
        raise FileNotFoundError(str(path))
    out={}
    with path.open(encoding="utf-8",newline="") as fh:
        reader=csv.DictReader(fh)
        required={"timestamp","actual","p_down","p_flat","p_up"}
        if not required.issubset(reader.fieldnames or set()):
            raise ValueError(f"{path}: schema mismatch")
        for row in reader:
            try:
                ts=int(row["timestamp"]); y=str(row["actual"]).strip()
                p=np.asarray([float(row["p_down"]),float(row["p_flat"]),float(row["p_up"])],float)
            except (TypeError,ValueError):
                continue
            if y not in CLASSES or not np.isfinite(p).all() or np.any(p<0) or p.sum()<=0:
                continue
            p=np.clip(p,EPS,1.0); p/=p.sum()
            out[ts]=(y,p)
    return out


def aligned_rows(h: str) -> list[dict[str,Any]]:
    books={e:load_expert(h,e) for e in EXPERTS}
    common=set(books[EXPERTS[0]])
    for e in EXPERTS[1:]:
        common &= set(books[e])
    rows=[]
    for ts in sorted(common):
        ys={books[e][ts][0] for e in EXPERTS}
        if len(ys)!=1:
            continue
        rows.append({"timestamp":ts,"y":next(iter(ys)),
                     "probs":{e:books[e][ts][1] for e in EXPERTS}})
    return rows


def metrics(rows: list[dict[str,Any]], probs: np.ndarray)->dict[str,float]:
    if not rows or probs.shape!=(len(rows),3):
        raise ValueError("metric_shape")
    labels=np.asarray([CLASSES.index(r["y"]) for r in rows],int)
    p=np.clip(probs,EPS,1.0); p/=p.sum(axis=1,keepdims=True)
    picked=np.clip(p[np.arange(len(labels)),labels],EPS,1.0)
    one=np.eye(3)[labels]
    conf=p.max(axis=1)
    ece=0.0
    for i in range(10):
        lo=i/10; hi=(i+1)/10
        m=(conf>=lo)&((conf<hi) if hi<1 else (conf<=hi))
        if np.any(m):
            ece += float(m.mean())*abs(float(np.mean(np.argmax(p[m],axis=1)==labels[m]))-float(np.mean(conf[m])))
    return {
        "n":int(len(rows)),
        "accuracy":float(np.mean(np.argmax(p,axis=1)==labels)),
        "logloss":float(-np.mean(np.log(picked))),
        "brier":float(np.mean(np.sum((p-one)**2,axis=1))),
        "ece":float(ece),
    }


def run_equal(rows: list[dict[str,Any]])->np.ndarray:
    return np.stack([np.mean(np.stack([r["probs"][e] for e in EXPERTS]),axis=0) for r in rows])


def run_hedge(rows: list[dict[str,Any]])->tuple[np.ndarray,list[dict[str,float]]]:
    losses=np.zeros(len(EXPERTS),float)
    weights=np.ones(len(EXPERTS),float)/len(EXPERTS)
    out=[]; traces=[]
    for i,r in enumerate(rows):
        if i<WARMUP:
            weights=np.ones(len(EXPERTS),float)/len(EXPERTS)
        else:
            scores=-ETA*losses
            scores-=scores.max()
            weights=np.exp(scores); weights/=weights.sum()
        matrix=np.stack([r["probs"][e] for e in EXPERTS])
        p=np.sum(weights[:,None]*matrix,axis=0)
        p=np.clip(p,EPS,1.0); p/=p.sum()
        out.append(p)
        traces.append({e:float(w) for e,w in zip(EXPERTS,weights)})
        y=CLASSES.index(r["y"])
        # Crucial causal boundary: update state only after prediction t is fixed.
        losses += -np.log(np.clip(matrix[:,y],EPS,1.0))
    return np.stack(out),traces


def block_deltas(rows,a,b):
    vals=[]
    for start in range(0,len(rows),BLOCK):
        end=min(start+BLOCK,len(rows))
        if end-start<100: continue
        ma=metrics(rows[start:end],a[start:end])
        mb=metrics(rows[start:end],b[start:end])
        vals.append({
            "n":end-start,
            "logloss_delta":mb["logloss"]-ma["logloss"],
            "brier_delta":mb["brier"]-ma["brier"],
            "accuracy_delta":mb["accuracy"]-ma["accuracy"],
            "ece_delta":mb["ece"]-ma["ece"],
        })
    return vals


def evaluate(h: str)->dict[str,Any]:
    rows=aligned_rows(h)
    if len(rows)<1000:
        return {"status":"DEFERRED","reason":"insufficient_aligned_oos_rows","n":len(rows)}
    split=int(len(rows)*(1-HOLDOUT_FRAC))
    dev,hold=rows[:split],rows[split:]
    eq_dev=run_equal(dev)
    hedge_dev,tr_dev=run_hedge(dev)
    eq_hold=run_equal(hold)
    hedge_hold,tr_hold=run_hedge(hold)
    dev_m_eq=metrics(dev,eq_dev); dev_m_h=metrics(dev,hedge_dev)
    hold_m_eq=metrics(hold,eq_hold); hold_m_h=metrics(hold,hedge_hold)
    blocks=block_deltas(dev,eq_dev,hedge_dev)
    if not blocks:
        return {"status":"DEFERRED","reason":"insufficient_blocks","n":len(rows)}
    ll=np.asarray([b["logloss_delta"] for b in blocks],float)
    br=np.asarray([b["brier_delta"] for b in blocks],float)
    ac=np.asarray([b["accuracy_delta"] for b in blocks],float)
    # Fixed strategy, not tuned on the frozen holdout.
    eligible=bool(
        float(np.mean(ll<0))>=0.70 and
        float(np.mean(br<0))>=0.70 and
        dev_m_h["logloss"] <= dev_m_eq["logloss"]*0.97 and
        dev_m_h["brier"] <= dev_m_eq["brier"]*0.99 and
        dev_m_h["accuracy"] >= dev_m_eq["accuracy"]-0.005
    )
    return {
        "status":"OK",
        "schema_version":1,
        "research_only":True,
        "production_changed":False,
        "chronological":True,
        "final_holdout_protected":True,
        "final_holdout_used_for_selection":False,
        "promotion_evidence_eligible":False,
        "config":{"eta":ETA,"warmup":WARMUP,"experts":list(EXPERTS),"holdout_frac":HOLDOUT_FRAC},
        "n":len(rows),"development_n":len(dev),"final_holdout_n":len(hold),
        "development":{"equal_weight":dev_m_eq,"online_hedge":dev_m_h,
                       "delta":{"accuracy":dev_m_h["accuracy"]-dev_m_eq["accuracy"],
                                "logloss":dev_m_h["logloss"]-dev_m_eq["logloss"],
                                "brier":dev_m_h["brier"]-dev_m_eq["brier"],
                                "ece":dev_m_h["ece"]-dev_m_eq["ece"]}},
        "block_stability":{"blocks":len(blocks),
            "improved_logloss_ratio":float(np.mean(ll<0)),
            "improved_brier_ratio":float(np.mean(br<0)),
            "non_worse_accuracy_ratio":float(np.mean(ac>=-0.005))},
        "final_holdout":{"equal_weight":hold_m_eq,"online_hedge":hold_m_h,
                         "delta":{"accuracy":hold_m_h["accuracy"]-hold_m_eq["accuracy"],
                                  "logloss":hold_m_h["logloss"]-hold_m_eq["logloss"],
                                  "brier":hold_m_h["brier"]-hold_m_eq["brier"],
                                  "ece":hold_m_h["ece"]-hold_m_eq["ece"]}},
        "eligibility":eligible,
        "trace_last_development":tr_dev[-1] if tr_dev else {},
        "trace_first_holdout":tr_hold[0] if tr_hold else {},
        "blocks_detail":blocks,
    }


def main():
    payload={"schema_version":1,"research_only":True,"production_changed":False,
             "horizons":{h:evaluate(h) for h in ("5m","10m")}}
    OUT.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps(payload,indent=2,sort_keys=True))


if __name__=="__main__":
    main()
