"""Exhaustive tractable BTC feature-pattern screen.

Exact individual subset enumeration for 92 features is 2^92, so this research-only
screen exhaustively evaluates every non-empty combination of seven disjoint semantic
feature families (127 patterns) for both 5m and 10m using identical expanding
chronological WFO folds. It never mutates production state.
"""
from __future__ import annotations
import csv
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from label_policy import NEUTRAL_BPS, direction_from_return

ROOT=Path(__file__).resolve().parents[1]
PANEL=ROOT/"data/historical_research/aligned_panel.csv"
OUT=ROOT/"data/historical_research/feature_pattern_exhaustive.json"
CLASSES=("DOWN","FLAT","UP")
HORIZONS={"5m":5,"10m":10}
MIN_TRAIN=12000
EMBARGO=10

BASE_FEATURES=[
"ret1","ret3","ret5","ret10","ret15","ret30","accel","rv5","rv10","rv30",
"rangepos10","rangepos30","body","upper","lower","volratio","voltrend","tradesratio",
"takerimb","basis","basis_delta","mark_gap","premium","eth_ret5","sol_ret5","eth_ret10",
"sol_ret10","eth_btc_rel5","sol_btc_rel5","ret5_x_vol","ret10_x_vol","flow_x_vol",
"range_x_flow","hour_sin","hour_cos","dow_sin","dow_cos","funding","funding_delta",
"oi_change","oi_z"]
FRONTIER_FEATURES=[
"rsi5","rsi14","rsi30","bb_z20","bb_z60","ema_slope5","ema_slope15","ema_slope30",
"atr_ratio14","range_asymmetry10","wick_imbalance10","volume_z20","trades_z20",
"dollar_volume_z20","flow_accel5","flow_z20","return_skew20","return_kurtosis20",
"autocorr5","drawdown30","runup30","price_to_ema20","amihud10","volume_price_corr20",
"body_pressure20","oi_x_return5","funding_x_oi","basis_x_vol","vol_term_ratio","range_z20",
"flow_return_corr20","ret20","ret60","ret120","rv60","rv120","vol_of_vol20",
"trend_efficiency20","trend_efficiency60","range_compression20","range_compression60",
"close_location10","close_location30","breakout_high20","breakout_low20","up_volume_ratio20",
"signed_volume_pressure20","trade_size_z20","signed_flow_accel20","parkinson_vol20","garman_klass_vol20"]
FEATURES=BASE_FEATURES+FRONTIER_FEATURES

FAMILY_GROUPS={
"base":tuple(BASE_FEATURES),
"momentum":("rsi5","rsi14","rsi30","bb_z20","bb_z60","ema_slope5","ema_slope15","ema_slope30","ret20","ret60","ret120","trend_efficiency20","trend_efficiency60","price_to_ema20"),
"volatility":("rv60","rv120","vol_of_vol20","atr_ratio14","return_skew20","return_kurtosis20","vol_term_ratio","range_z20","range_compression20","range_compression60","parkinson_vol20","garman_klass_vol20"),
"price_action":("range_asymmetry10","wick_imbalance10","drawdown30","runup30","close_location10","close_location30","breakout_high20","breakout_low20"),
"flow":("volume_z20","trades_z20","dollar_volume_z20","flow_accel5","flow_z20","amihud10","volume_price_corr20","flow_return_corr20","body_pressure20","up_volume_ratio20","signed_volume_pressure20","trade_size_z20","signed_flow_accel20"),
"derivatives":("basis_x_vol","oi_x_return5","funding_x_oi"),
"dependence":("autocorr5",),
}
assert set(FEATURES)=={f for g in FAMILY_GROUPS.values() for f in g}
assert set().union(*[set(g) for g in FAMILY_GROUPS.values()])==set(FEATURES)
FAMILY_NAMES=tuple(FAMILY_GROUPS)
PATTERN_COUNT=2**len(FAMILY_NAMES)-1

def metrics(y,p):
    yi=np.asarray([CLASSES.index(v) for v in y],dtype=int)
    p=np.clip(np.asarray(p,dtype=float),1e-7,1.0)
    p/=p.sum(axis=1,keepdims=True)
    hit=(p.argmax(axis=1)==yi).astype(float)
    conf=p.max(axis=1)
    ece=0.0
    for k in range(10):
        lo,hi=k/10,(k+1)/10
        m=(conf>=lo)&((conf<=hi) if hi==1 else (conf<hi))
        if np.any(m):
            ece+=float(m.mean())*abs(float(hit[m].mean())-float(conf[m].mean()))
    one=np.eye(3)[yi]
    return {"n":int(len(y)),"accuracy":float(hit.mean()),
            "logloss":float(log_loss(yi,p,labels=[0,1,2])),
            "brier":float(np.mean(np.sum((p-one)**2,axis=1))),
            "ece":float(ece)}

def load():
    if not PANEL.is_file(): raise RuntimeError("aligned_panel.csv missing; run historical research first")
    with PANEL.open(encoding="utf-8",newline="") as fh:
        rows=list(csv.DictReader(fh))
        fields=set(fh.fieldnames or ())
    missing=sorted(set(FEATURES)-fields)
    if missing: raise RuntimeError("aligned_panel feature schema incomplete: "+",".join(missing))
    ts=np.asarray([int(r["timestamp"]) for r in rows],dtype=np.int64)
    price=np.asarray([float(r["price"]) for r in rows],dtype=float)
    X=np.asarray([[float(r[f]) for f in FEATURES] for r in rows],dtype=float)
    if len(rows)==0 or not np.isfinite(price).all() or not np.isfinite(X).all():
        raise RuntimeError("aligned_panel contains invalid numeric data")
    return ts,price,X

def labels(ts,price,h):
    lookup={int(t):i for i,t in enumerate(ts)}
    keep=[]; y=[]
    for i,t in enumerate(ts):
        j=lookup.get(int(t)+h*60_000)
        if j is None: continue
        if not (price[i]>0 and np.isfinite(price[i]) and np.isfinite(price[j])): continue
        rbps=(price[j]/price[i]-1.0)*10000.0
        y.append(direction_from_return(rbps / 10000.0))
        keep.append(i)
    return np.asarray(keep,dtype=np.int64),np.asarray(y,dtype=object)

def folds(n):
    if n<MIN_TRAIN+EMBARGO+6000: raise RuntimeError(f"insufficient labeled rows: {n}")
    ends=(int(n*.60),int(n*.70),int(n*.80)); out=[]
    for i,tr in enumerate(ends):
        hi=ends[i+1] if i+1<len(ends) else n
        lo=tr+EMBARGO
        if tr>=MIN_TRAIN and lo<hi: out.append((tr,lo,hi))
    if len(out)!=3: raise RuntimeError(f"expected 3 folds, got {len(out)}")
    return out

def pattern_features(mask):
    families=[]; selected=set()
    for i,name in enumerate(FAMILY_NAMES):
        if mask&(1<<i):
            families.append(name); selected.update(FAMILY_GROUPS[name])
    return families,tuple(f for f in FEATURES if f in selected)

def development_selection_key(record):
    """Rank patterns on development OOS only, prioritizing probabilistic quality."""
    metrics_row=record["development_metrics"]
    return (
        float(metrics_row["logloss"]),
        float(metrics_row["brier"]),
        float(metrics_row["ece"]),
        -float(metrics_row["accuracy"]),
        int(record["feature_count"]),
        str(record["pattern_id"]),
    )

def screen_with_slices(X,y,fold_spec):
    """Fit each fold once and return full/development/frozen-holdout metrics."""
    ps=[]; ys=[]; per_fold=[]
    for tr,lo,hi in fold_spec:
        model=Pipeline([("scale",StandardScaler()),("model",LogisticRegression(C=.3,max_iter=450,solver="lbfgs"))])
        model.fit(X[:tr],y[:tr])
        raw=model.predict_proba(X[lo:hi])
        aligned=np.full((len(raw),3),1e-7,dtype=float)
        for j,c in enumerate(model.classes_): aligned[:,CLASSES.index(str(c))]=raw[:,j]
        aligned/=aligned.sum(axis=1,keepdims=True)
        yy=y[lo:hi]
        ps.append(aligned); ys.append(yy); per_fold.append(metrics(yy,aligned))
    def aggregate(indices):
        if not indices:
            raise ValueError("empty fold slice")
        return metrics(
            np.concatenate([ys[i] for i in indices]),
            np.vstack([ps[i] for i in indices]),
        )
    all_metrics=aggregate(list(range(len(fold_spec))))
    dev_metrics=aggregate(list(range(max(0,len(fold_spec)-1))))
    holdout_metrics=aggregate([len(fold_spec)-1])
    return all_metrics,dev_metrics,holdout_metrics,per_fold

def screen(X,y,fold_spec):
    all_metrics,_,_,per_fold=screen_with_slices(X,y,fold_spec)
    return all_metrics,per_fold

def run_single_feature_ablation(ts,price,X):
    """Measure every single frontier add and every single base-feature removal."""
    # Use exact same label construction/folds as family screening. The purpose is
    # attribution, not model selection; all outputs remain research-only.
    result={}
    for horizon,minutes in HORIZONS.items():
        idx,y=labels(ts,price,minutes)
        XX=X[idx]
        fs=folds(len(y))
        base_cols=np.arange(len(BASE_FEATURES),dtype=np.int64)
        base_all,base_dev,base_holdout,_=screen_with_slices(XX[:,base_cols],y,fs)
        add_records=[]
        for feature in FRONTIER_FEATURES:
            col=FEATURES.index(feature)
            cols=np.concatenate((base_cols,np.asarray([col],dtype=np.int64)))
            mm,dm,hm,ff=screen_with_slices(XX[:,cols],y,fs)
            add_records.append({
                "feature":feature,
                "metrics":mm,
                "development_metrics":dm,
                "frozen_holdout_metrics":hm,
                "fold_metrics":ff,
                "delta_vs_base":{k:float(mm[k]-base_all[k]) for k in ("logloss","accuracy","brier","ece")},
                "development_delta_vs_base":{k:float(dm[k]-base_dev[k]) for k in ("logloss","accuracy","brier","ece")},
                "frozen_holdout_delta_vs_base":{k:float(hm[k]-base_holdout[k]) for k in ("logloss","accuracy","brier","ece")},
                "status":"OK",
            })
        remove_records=[]
        for feature in BASE_FEATURES:
            cols=np.asarray([i for i,f in enumerate(BASE_FEATURES) if f!=feature],dtype=np.int64)
            mm,dm,hm,ff=screen_with_slices(XX[:,cols],y,fs)
            remove_records.append({
                "feature":feature,
                "metrics":mm,
                "development_metrics":dm,
                "frozen_holdout_metrics":hm,
                "fold_metrics":ff,
                "delta_vs_base":{k:float(mm[k]-base_all[k]) for k in ("logloss","accuracy","brier","ece")},
                "development_delta_vs_base":{k:float(dm[k]-base_dev[k]) for k in ("logloss","accuracy","brier","ece")},
                "frozen_holdout_delta_vs_base":{k:float(hm[k]-base_holdout[k]) for k in ("logloss","accuracy","brier","ece")},
                "status":"OK",
            })
        add_rank=sorted(add_records,key=lambda r:(r["development_delta_vs_base"]["logloss"],-r["development_delta_vs_base"]["accuracy"]))
        remove_rank=sorted(remove_records,key=lambda r:(-r["development_delta_vs_base"]["logloss"],r["development_delta_vs_base"]["accuracy"]))
        result[horizon]={
            "sample_n":int(len(y)),
            "base_metrics":base_all,
            "frontier_single_add_count":len(add_records),
            "base_single_remove_count":len(remove_records),
            "frontier_single_add":add_records,
            "base_single_remove":remove_records,
            "best_single_add_by_logloss":add_rank[:20],
            "least_harmful_single_remove_by_logloss":remove_rank[:20],
        }
    return result

def main():
    ts,price,X=load()
    result={
        "schema_version":1,
        "experiment_id":"btc_feature_pattern_exhaustive_v2",
        "protocol_version":"feature-family-expanding-wfo-v1",
        "status":"RUNNING",
        "research_only":True,
        "production_changed":False,
        "promotion_effect":"none",
        "promotion_evidence_eligible":False,
        "pit_evidence_status":"NON_STRICT_ARCHIVE_TIMING",
        "pit_policy_note":"Binance Vision/archive publication timing is not independently proven at feature-record level; the 3-day completed-data boundary is a conservative operational buffer, not PIT proof.",
        "source_lineage":{
            "primary":"Binance USD-M futures/spot/mark/premium archives",
            "derivatives_context":"Binance funding/open-interest archives where available",
            "cross_asset":"Binance ETH/SOL futures",
            "independence_note":"same Binance upstream is not counted as independent evidence"
        },
        "target_contract":{
            "version":"label-policy-v1",
            "horizons_minutes":dict(HORIZONS),
            "neutral_bps":float(NEUTRAL_BPS),
            "classes":list(CLASSES),
            "label_function":"label_policy.direction_from_return",
            "future_join":"exact_timestamp_plus_horizon",
            "missing_future_policy":"exclude"
        },
        "feature_schema_sha256":hashlib.sha256("|".join(FEATURES).encode("utf-8")).hexdigest(),
        "github_sha":os.environ.get("GITHUB_SHA"),
        "search_scope":"exhaustive_nonempty_combinations_of_7_disjoint_feature_families",
        "exact_individual_feature_subset_space":int(2**len(FEATURES)),
        "exact_individual_feature_subset_space_is_computationally_intractable":True,
        "family_count":len(FAMILY_NAMES),
        "pattern_count_expected":PATTERN_COUNT,
        "families":{n:{"feature_count":len(g),"features":list(g)} for n,g in FAMILY_GROUPS.items()},
        "feature_count":len(FEATURES),
        "base_feature_count":len(BASE_FEATURES),
        "frontier_feature_count":len(FRONTIER_FEATURES),
        "fold_contract":{"fold_count":3,"embargo_rows":EMBARGO,"chronological":True,"random_split":False,"selection_folds":[0,1],"frozen_holdout_fold":2,"holdout_excluded_from_selection":True},
        "model_role":"screening_only_logistic_regression",
        "selection_leakage_guard":"rank_candidates_on_development_folds_only; evaluate_latest_fold_as_frozen_holdout",
        "selection_metric_order":["development_logloss","development_brier","development_ece","development_accuracy","feature_count"],
        "fine_grained_single_feature_ablation":True,
        "horizons":{}
    }
    for h,minutes in HORIZONS.items():
        idx,y=labels(ts,price,minutes); XX=X[idx]; fs=folds(len(y)); records=[]; failures=0
        for mask in range(1,PATTERN_COUNT+1):
            fam,fn=pattern_features(mask); cols=np.asarray([FEATURES.index(f) for f in fn],dtype=np.int64)
            try:
                mm,dm,hm,ff=screen_with_slices(XX[:,cols],y,fs)
                records.append({"pattern_id":f"mask_{mask:03d}","mask":mask,"families":fam,"feature_count":len(fn),"metrics":mm,"development_metrics":dm,"frozen_holdout_metrics":hm,"fold_metrics":ff,"status":"OK"})
            except (ValueError,RuntimeError,np.linalg.LinAlgError) as exc:
                failures+=1
                records.append({"pattern_id":f"mask_{mask:03d}","mask":mask,"families":fam,"feature_count":len(fn),"status":"FAILED","error":f"{type(exc).__name__}:{exc}"})
        base=next((r for r in records if r["families"]==["base"]),None)
        valid=[r for r in records if r["status"]=="OK"]
        if base and base.get("metrics"):
            bm=base["metrics"]; bd=base["development_metrics"]; bh=base["frozen_holdout_metrics"]
            for r in valid:
                r["delta_vs_base"]={k:float(r["metrics"][k]-bm[k]) for k in ("logloss","accuracy","brier","ece")}
                r["development_delta_vs_base"]={k:float(r["development_metrics"][k]-bd[k]) for k in ("logloss","accuracy","brier","ece")}
                r["frozen_holdout_delta_vs_base"]={k:float(r["frozen_holdout_metrics"][k]-bh[k]) for k in ("logloss","accuracy","brier","ece")}
        # Rank only on development folds. The latest fold remains a frozen holdout
        # and is intentionally excluded from candidate selection.
        ranked=sorted(valid,key=development_selection_key)
        result["horizons"][h]={
            "samples":int(len(y)),
            "folds":fs,
            "patterns_expected":PATTERN_COUNT,
            "patterns_completed":len(valid),
            "pattern_failures":failures,
            "status":"COMPLETE" if len(valid)==PATTERN_COUNT else "PARTIAL_FAILURE",
            "base_pattern":base,
            "top_20_by_development_logloss":ranked[:20],
            "all_patterns":records,
            "selection_protocol":{"selection_folds":[0,1],"frozen_holdout_fold":2,"holdout_touched_by_ranking":False}
        }
    result["fine_grained_ablation"]=run_single_feature_ablation(ts,price,X)
    result["status"]="COMPLETE" if all(result["horizons"][h]["status"]=="COMPLETE" for h in HORIZONS) else "PARTIAL_FAILURE"
    result["completed_utc"]=datetime.now(timezone.utc).isoformat()
    OUT.write_text(json.dumps(result,indent=2,sort_keys=True),encoding="utf-8")
    print(json.dumps({"status":result["status"],"pattern_count_expected":PATTERN_COUNT,"5m_completed":result["horizons"]["5m"]["patterns_completed"],"10m_completed":result["horizons"]["10m"]["patterns_completed"]},indent=2))
    if result["status"]!="COMPLETE": raise SystemExit("exhaustive feature-family screen incomplete; fail closed")

if __name__=="__main__": main()
