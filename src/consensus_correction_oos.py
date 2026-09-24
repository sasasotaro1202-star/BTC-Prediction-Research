"""Research-only deterministic consensus correction OOS.

Baseline is a walk-forward RF champion-family model. Candidate corrections are
applied only when non-RF experts agree against RF:
- 2-of-3 consensus: at least 2 of LR/ExtraTrees/HGB share a direction different
  from RF.
- 3-of-3 consensus: all LR/ExtraTrees/HGB share a direction different from RF.

No learned router or threshold tuning is used. This isolates whether a simple,
low-variance disagreement rule can improve per-event accuracy. All evaluation
is chronological with purge/embargo and a protected final holdout.
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from model_compare import CLASSES, PURGE_BARS, EMBARGO_BARS, load_archive_research_rows, metrics

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "historical_research" / "consensus_correction_oos.json"
HORIZONS = ("5m", "10m")
MIN_TRAIN = 3000
TEST_BLOCK = 600
FINAL_HOLDOUT_FRAC = 0.20
MAX_ROWS = 12000
EPS = 1e-7
NON_RF = ("logreg", "extra_trees", "hgb")


def _factory(name: str):
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=180, max_depth=10, min_samples_leaf=15,
            max_features="sqrt", random_state=42, n_jobs=-1
        )
    if name == "logreg":
        return Pipeline([
            ("scale", StandardScaler()),
            ("model", LogisticRegression(C=0.3, max_iter=2500)),
        ])
    if name == "extra_trees":
        return ExtraTreesClassifier(
            n_estimators=140, max_depth=10, min_samples_leaf=15,
            max_features="sqrt", random_state=42, n_jobs=-1
        )
    if name == "hgb":
        return HistGradientBoostingClassifier(
            max_iter=180, max_leaf_nodes=15, learning_rate=0.04,
            l2_regularization=1.5, random_state=42
        )
    raise ValueError(f"unknown_model:{name}")


def _align(model, rows):
    raw = np.asarray(model.predict_proba(np.asarray([r["x"] for r in rows], dtype=float)), dtype=float)
    out = np.full((len(rows), 3), EPS, dtype=float)
    idx = {c: i for i, c in enumerate(CLASSES)}
    for j, cls in enumerate(model.classes_):
        if str(cls) in idx:
            out[:, idx[str(cls)]] = raw[:, j]
    out = np.clip(out, EPS, 1.0)
    return out / out.sum(axis=1, keepdims=True)


def _fit_models(train):
    y = np.asarray([r["y"] for r in train], dtype=str)
    if len(train) < MIN_TRAIN or len(np.unique(y)) < 3:
        raise ValueError("insufficient_training_rows")
    return {name: _factory(name).fit(np.asarray([r["x"] for r in train], dtype=float), y)
            for name in ("random_forest", *NON_RF)}


def _metrics(rows, probs):
    m = metrics([r["y"] for r in rows], probs)
    return {
        "n": int(len(rows)),
        "accuracy": float(m["accuracy"]),
        "logloss": float(m["logloss"]),
        "brier": float(m["brier"]),
        "ece": float(m.get("ece", m.get("calibration_error", math.nan))),
    }


def _apply(probs):
    rf = probs["random_forest"]
    rf_pred = np.argmax(rf, axis=1)
    non_rf_pred = np.stack([np.argmax(probs[name], axis=1) for name in NON_RF], axis=1)
    counts = np.sum(non_rf_pred != rf_pred[:, None], axis=1)
    candidate = {}

    for k in (2, 3):
        active = counts >= k
        out = rf.copy()
        if np.any(active):
            # If the qualifying non-RF experts share a single alternative class,
            # average their probabilities. For k=2, exactly two or three may
            # disagree; majority among the disagreeing experts must be unique.
            for i in np.where(active)[0]:
                disagree_dirs = [int(non_rf_pred[i, j]) for j in range(3)
                                 if int(non_rf_pred[i, j]) != int(rf_pred[i])]
                if not disagree_dirs:
                    continue
                values, counts2 = np.unique(disagree_dirs, return_counts=True)
                winner = int(values[np.argmax(counts2)])
                if int(np.max(counts2)) < k:
                    continue
                selected = [j for j in range(3) if int(non_rf_pred[i, j]) == winner]
                mix = np.mean([probs[NON_RF[j]][i] for j in selected], axis=0)
                mix = np.clip(mix, EPS, 1.0)
                mix /= mix.sum()
                out[i] = mix
        candidate[f"consensus_{k}of3"] = (out, active)
    return candidate, counts


def _ci(values, seed=42, draws=2000):
    arr = np.asarray(values, dtype=float)
    if len(arr) < 4:
        return {"n": int(len(arr)), "mean": float(arr.mean()) if len(arr) else None}
    rng = np.random.default_rng(seed)
    boot = rng.choice(arr, size=(draws, len(arr)), replace=True).mean(axis=1)
    return {"n": int(len(arr)), "mean": float(arr.mean()),
            "lower": float(np.quantile(boot, 0.025)),
            "upper": float(np.quantile(boot, 0.975))}


def _causal_train(rows, test_start, horizon):
    cutoff = datetime.fromisoformat(str(test_start).replace("Z", "+00:00")).astimezone(timezone.utc)
    gap = cutoff - timedelta(minutes=int(PURGE_BARS[horizon] + EMBARGO_BARS[horizon]))
    out=[]
    for row in rows:
        created=datetime.fromisoformat(str(row["created"]).replace("Z","+00:00")).astimezone(timezone.utc)
        target=datetime.fromisoformat(str(row["target"]).replace("Z","+00:00")).astimezone(timezone.utc)
        if created < target and target < gap:
            out.append(row)
    return out


def evaluate(horizon):
    rows=load_archive_research_rows(horizon, MAX_ROWS)
    if len(rows) < MIN_TRAIN + TEST_BLOCK + 100:
        return {"status":"DEFERRED","research_only":True,"production_changed":False,
                "promotion_allowed":False,"n":len(rows),"reason":"insufficient_rows"}
    split=int(len(rows)*(1-FINAL_HOLDOUT_FRAC))
    dev,holdout=rows[:split],rows[split:]
    blocks={"consensus_2of3":[],"consensus_3of3":[]}

    for end in range(MIN_TRAIN, len(dev), TEST_BLOCK):
        test=dev[end:min(end+TEST_BLOCK,len(dev))]
        if len(test)<TEST_BLOCK//2: continue
        train=_causal_train(dev[:end],test[0]["created"],horizon)
        if len(train)<MIN_TRAIN: continue
        models=_fit_models(train)
        probs={name:_align(models[name],test) for name in ("random_forest",*NON_RF)}
        candidates,disagreement=_apply(probs)
        base=_metrics(test,probs["random_forest"])
        for name,(cp,active) in candidates.items():
            cm=_metrics(test,cp)
            high=active
            high_stats={}
            if np.any(high):
                y=np.asarray([CLASSES.index(r["y"]) for r in test],dtype=int)
                high_stats={
                    "n":int(high.sum()),
                    "baseline_accuracy":float(np.mean(np.argmax(probs["random_forest"][high],axis=1)==y[high])),
                    "candidate_accuracy":float(np.mean(np.argmax(cp[high],axis=1)==y[high])),
                    "baseline_logloss":float(-np.mean(np.log(np.clip(probs["random_forest"][high,y[high]],EPS,1.0)))),
                    "candidate_logloss":float(-np.mean(np.log(np.clip(cp[high,y[high]],EPS,1.0)))),
                }
            blocks[name].append({
                "n":len(test),
                "baseline":base,
                "candidate":cm,
                "delta":{"accuracy":cm["accuracy"]-base["accuracy"],
                         "logloss":cm["logloss"]-base["logloss"],
                         "brier":cm["brier"]-base["brier"],
                         "ece":cm["ece"]-base["ece"]},
                "coverage":float(active.mean()),
                "high_disagreement":high_stats,
            })

    results={}
    for name,bs in blocks.items():
        if len(bs)<8:
            results[name]={"status":"DEFERRED","blocks":len(bs)}
            continue
        ll=np.asarray([b["delta"]["logloss"] for b in bs]); br=np.asarray([b["delta"]["brier"] for b in bs]); ac=np.asarray([b["delta"]["accuracy"] for b in bs]); ece=np.asarray([b["delta"]["ece"] for b in bs])
        hold_train=_causal_train(dev,holdout[0]["created"],horizon)
        if len(hold_train)<MIN_TRAIN:
            results[name]={"status":"DEFERRED","blocks":len(bs),"reason":"insufficient_holdout_training"}
            continue
        models=_fit_models(hold_train)
        hp={m:_align(models[m],holdout) for m in ("random_forest",*NON_RF)}
        hc,_=_apply(hp)
        cand=hc[name][0]
        base_h=_metrics(holdout,hp["random_forest"]); cand_h=_metrics(holdout,cand)
        eligible=bool(
            (ac.mean()>=0.03 or ll.mean()<=-0.03)
            and br.mean()<=-0.006
            and float(np.mean(ac>=-0.005))>=0.70
            and cand_h["logloss"]<=base_h["logloss"]
            and cand_h["brier"]<=base_h["brier"]
            and cand_h["accuracy"]>=base_h["accuracy"]-0.005
        )
        results[name]={
            "status":"OK","blocks":len(bs),"development":{
                "mean_delta":{"accuracy":float(ac.mean()),"logloss":float(ll.mean()),"brier":float(br.mean()),"ece":float(ece.mean())},
                "ci95":{"accuracy":_ci(ac),"logloss":_ci(ll),"brier":_ci(br),"ece":_ci(ece)},
                "non_worse_accuracy_ratio":float(np.mean(ac>=-0.005)),
                "improved_logloss_ratio":float(np.mean(ll<0)),
                "improved_brier_ratio":float(np.mean(br<0)),
                "samples":int(sum(b["n"] for b in bs)),
            },
            "final_holdout":{"n":len(holdout),"coverage":float(hc[name][1].mean()),
                             "baseline":base_h,"candidate":cand_h,
                             "delta":{k:cand_h[k]-base_h[k] for k in ("accuracy","logloss","brier","ece")}},
            "eligibility_candidate":eligible,
        }
    return {"status":"OK","schema_version":1,"research_only":True,"production_changed":False,
            "promotion_allowed":False,"final_holdout_protected":True,"horizon":horizon,
            "n":len(rows),"development_n":len(dev),"final_holdout_n":len(holdout),
            "config":{"non_rf":list(NON_RF),"purge_bars":int(PURGE_BARS[horizon]),"embargo_bars":int(EMBARGO_BARS[horizon]),
                      "rules":["2of3","3of3"],"thresholds_tuned":False},
            "candidates":results}


def main():
    payload={"schema_version":1,"research_only":True,"production_changed":False,
             "horizons":{h:evaluate(h) for h in HORIZONS}}
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps(payload,indent=2,sort_keys=True))


if __name__=="__main__":
    main()
