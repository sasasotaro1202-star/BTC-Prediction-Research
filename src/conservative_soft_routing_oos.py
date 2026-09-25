"""Research-only conservative soft routing for BTC short-horizon direction.

Champion remains the default. Alternative experts receive a capped soft weight
only on current observations that are both high-uncertainty and in disagreement
with at least one alternative. Expert weights come only from a strictly prior
meta block and are shrunken toward zero, making this a conservative challenger.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from model_compare import (
    CLASSES,
    EMBARGO_BARS,
    PURGE_BARS,
    load_archive_research_rows,
    metrics as core_metrics,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/historical_research/conservative_soft_routing_oos.json"

HORIZONS = ("5m", "10m")
ALTS = ("logreg", "extra_trees", "hgb")

MIN_TRAIN = 3000
META_BLOCK = 500
TEST_BLOCK = 500
MIN_META_ROWS = 250
MAX_ROWS = 12000
FINAL_HOLDOUT_FRAC = 0.20
UNCERTAINTY_QUANTILE = 0.75

MAX_ALT_MASS = 0.25
ADVANTAGE_SCALE = 0.04
MIN_ADVANTAGE = 0.005
MIN_DISAGREE = 1
EPS = 1e-7


def _metrics(y: list[str], p: np.ndarray) -> dict[str, float | int]:
    raw = core_metrics(y, p)
    return {
        "n": int(len(y)),
        "accuracy": float(raw["accuracy"]),
        "logloss": float(raw["logloss"]),
        "brier": float(raw["brier"]),
        "ece": float(raw["calibration_error"]),
        "mean_confidence": float(np.max(p, axis=1).mean()),
    }


def _align(model: Any, rows: list[dict[str, Any]]) -> np.ndarray:
    X = np.asarray([r["x"] for r in rows], dtype=float)
    raw = np.asarray(model.predict_proba(X), dtype=float)
    out = np.full((len(rows), 3), EPS, dtype=float)
    idx = {c: i for i, c in enumerate(CLASSES)}
    for j, cls in enumerate(getattr(model, "classes_", [])):
        if str(cls) in idx:
            out[:, idx[str(cls)]] = raw[:, j]
    if out.shape != (len(rows), 3) or not np.isfinite(out).all():
        raise ValueError("expert_probability_invalid")
    out = np.clip(out, EPS, 1.0)
    out /= out.sum(axis=1, keepdims=True)
    return out


def _champion_probs(rows: list[dict[str, Any]], horizon: str) -> np.ndarray:
    model_path = ROOT / "models" / f"{horizon}.joblib"
    meta_path = ROOT / "models" / f"{horizon}.json"
    if not model_path.is_file() or not meta_path.is_file():
        raise FileNotFoundError(f"production_champion_artifact_missing:{horizon}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("candidate") is True:
        raise ValueError(f"production_candidate_artifact_rejected:{horizon}")
    model = joblib.load(model_path)
    X = np.asarray([r["x"] for r in rows], dtype=float)
    raw = np.asarray(model.predict_proba(X), dtype=float)
    out = np.full((len(rows), 3), EPS, dtype=float)
    idx = {c: i for i, c in enumerate(CLASSES)}
    for j, cls in enumerate(getattr(model, "classes_", [])):
        if str(cls) in idx:
            out[:, idx[str(cls)]] = raw[:, j]
    if out.shape != (len(rows), 3) or not np.isfinite(out).all():
        raise ValueError(f"production_champion_probability_invalid:{horizon}")
    out = np.clip(out, EPS, 1.0)
    out /= out.sum(axis=1, keepdims=True)
    return out


def _factories() -> dict[str, Any]:
    return {
        "logreg": lambda: Pipeline([
            ("scale", StandardScaler()),
            ("model", LogisticRegression(C=0.3, max_iter=2500)),
        ]),
        "extra_trees": lambda: ExtraTreesClassifier(
            n_estimators=160, max_depth=10, min_samples_leaf=15,
            max_features="sqrt", random_state=42, n_jobs=-1
        ),
        "hgb": lambda: HistGradientBoostingClassifier(
            max_iter=180, max_leaf_nodes=15, learning_rate=0.04,
            l2_regularization=1.5, random_state=42
        ),
    }


def _fit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if len(rows) < MIN_TRAIN or len({r["y"] for r in rows}) < 3:
        raise ValueError("insufficient_or_single_class_training_rows")
    X = np.asarray([r["x"] for r in rows], dtype=float)
    y = np.asarray([r["y"] for r in rows], dtype=str)
    out = {}
    for name, factory in _factories().items():
        m = factory()
        m.fit(X, y)
        out[name] = m
    return out


def _expert_probs(models: dict[str, Any], rows: list[dict[str, Any]], horizon: str) -> dict[str, np.ndarray]:
    out = {name: _align(model, rows) for name, model in models.items()}
    out["production"] = _champion_probs(rows, horizon)
    return out


def _uncertainty(probs: dict[str, np.ndarray]) -> np.ndarray:
    champion = probs["production"]
    ordered = np.sort(champion, axis=1)[:, ::-1]
    margin = ordered[:, 0] - ordered[:, 1]
    entropy = -np.sum(champion * np.log(np.clip(champion, EPS, 1.0)), axis=1) / math.log(3.0)
    stack = np.stack([probs[e] for e in ALTS], axis=0)
    disagreement = np.mean(np.sum((stack - champion[None, :, :]) ** 2, axis=2), axis=0)
    vote_disagreement = 1.0 - np.mean(
        np.argmax(stack, axis=2) == np.argmax(champion, axis=1)[None, :], axis=0
    )
    return np.clip(
        0.45 * entropy
        + 0.30 * np.clip(disagreement / 0.10, 0.0, 1.0)
        + 0.15 * (1.0 - margin)
        + 0.10 * vote_disagreement,
        0.0, 1.0
    )


def _gate_threshold(meta_uncertainty: np.ndarray) -> float:
    vals=np.asarray(meta_uncertainty,dtype=float)
    vals=vals[np.isfinite(vals)]
    if len(vals)<100: raise ValueError("insufficient_meta_uncertainty")
    q=float(np.quantile(vals,UNCERTAINTY_QUANTILE))
    if not 0.0<=q<=1.0: raise ValueError("uncertainty_threshold_invalid")
    return q


def _advantage_weights(
    meta_probs: dict[str, np.ndarray],
    meta_rows: list[dict[str, Any]],
    meta_gate: np.ndarray,
) -> dict[str, float]:
    y=np.asarray([CLASSES.index(r["y"]) for r in meta_rows],dtype=int)
    champion=meta_probs["production"]
    weights={e:0.0 for e in ALTS}
    for e in ALTS:
        p=meta_probs[e]
        champion_loss=-np.log(np.clip(champion[np.arange(len(meta_rows)),y],EPS,1.0))
        alt_loss=-np.log(np.clip(p[np.arange(len(meta_rows)),y],EPS,1.0))
        if int(meta_gate.sum()) < 50:
            idx=np.arange(len(meta_rows))
        else:
            idx=np.flatnonzero(meta_gate)
        advantage=float(np.mean(champion_loss[idx]-alt_loss[idx])) if len(idx) else 0.0
        if advantage <= MIN_ADVANTAGE:
            weights[e]=0.0
        else:
            weights[e]=float(np.clip(advantage/ADVANTAGE_SCALE,0.0,1.0))
    total=sum(weights.values())
    if total>0:
        for e in ALTS:
            weights[e]=(weights[e]/total)*MAX_ALT_MASS
    return weights


def _mix(
    probs: dict[str, np.ndarray],
    gate: np.ndarray,
    weights: dict[str, float],
) -> tuple[np.ndarray,np.ndarray]:
    n=len(probs["production"])
    final=probs["production"].copy()
    routing=np.zeros((n,4),dtype=float)
    routing[:,0]=1.0
    for i in range(n):
        if not gate[i] or sum(weights.values())<=0:
            continue
        total_alt=sum(weights.values())
        p=total_alt*0.0 + (1.0-total_alt)*probs["production"][i]
        for j,e in enumerate(ALTS, start=1):
            p += weights[e]*probs[e][i]
            routing[i,j]=weights[e]
        routing[i,0]=1.0-total_alt
        final[i]=p/np.sum(p)
    return final,routing


def _evaluate_development(rows: list[dict[str,Any]],horizon:str) -> dict[str,Any]:
    gap=int(PURGE_BARS[horizon]+EMBARGO_BARS[horizon])
    blocks=[]
    for test_start in range(MIN_TRAIN+META_BLOCK+gap,len(rows),TEST_BLOCK):
        test_end=min(test_start+TEST_BLOCK,len(rows))
        test=rows[test_start:test_end]
        meta_end=test_start-gap
        meta_start=meta_end-META_BLOCK
        train=rows[:meta_start]
        meta=rows[meta_start:meta_end]
        if len(test)<TEST_BLOCK//2 or len(meta)<MIN_META_ROWS or len(train)<MIN_TRAIN:
            continue
        meta_models=_fit(train)
        meta_probs=_expert_probs(meta_models,meta,horizon)
        meta_unc=_uncertainty(meta_probs)
        threshold=_gate_threshold(meta_unc)
        meta_gate=meta_unc>=threshold
        weights=_advantage_weights(meta_probs,meta,meta_gate)

        test_models=_fit(rows[:test_start-gap])
        test_probs=_expert_probs(test_models,test,horizon)
        test_unc=_uncertainty(test_probs)
        alt_votes=np.stack([np.argmax(test_probs[e],axis=1) for e in ALTS],axis=0)
        champ_vote=np.argmax(test_probs["production"],axis=1)
        disagreement_count=np.sum(alt_votes!=champ_vote[None,:],axis=0)
        gate=(test_unc>=threshold)&(disagreement_count>=MIN_DISAGREE)

        final,routing=_mix(test_probs,gate,weights)
        y=[r["y"] for r in test]
        base=_metrics(y,test_probs["production"]); cand=_metrics(y,final)
        gate_idx=np.flatnonzero(gate)
        high_base=_metrics([y[i] for i in gate_idx],test_probs["production"][gate]) if len(gate_idx) else None
        high_cand=_metrics([y[i] for i in gate_idx],final[gate]) if len(gate_idx) else None
        blocks.append({
            "n":len(test),"gate_n":int(gate.sum()),"gate_rate":float(gate.mean()),
            "uncertainty_threshold":threshold,"weights":weights,
            "baseline":base,"candidate":cand,
            "delta":{k:float(cand[k]-base[k]) for k in ("accuracy","logloss","brier","ece")},
            "high_uncertainty_disagreement":{
                "n":int(gate.sum()),"baseline":high_base,"candidate":high_cand
            },
        })
    if not blocks:
        return {"status":"DEFERRED","reason":"no_valid_oos_blocks"}
    b=_agg(blocks,"baseline"); c=_agg(blocks,"candidate")
    d={k:float(c[k]-b[k]) for k in ("accuracy","logloss","brier","ece")}
    return {
        "status":"OK","blocks":len(blocks),"samples":b["n"],
        "baseline":b,"candidate":c,"delta":d,
        "relative_improvement":{
            "accuracy":d["accuracy"]/max(b["accuracy"],EPS),
            "logloss":(b["logloss"]-c["logloss"])/max(b["logloss"],EPS),
            "brier":(b["brier"]-c["brier"])/max(b["brier"],EPS)
        },
        "stability":{
            "improved_logloss_ratio":float(np.mean([x["delta"]["logloss"]<0 for x in blocks])),
            "improved_brier_ratio":float(np.mean([x["delta"]["brier"]<0 for x in blocks])),
            "non_worse_accuracy_ratio":float(np.mean([x["delta"]["accuracy"]>=-0.005 for x in blocks]))
        },
        "gate_mean_rate":float(np.mean([x["gate_rate"] for x in blocks])),
        "blocks_detail":blocks
    }


def _agg(blocks,side):
    total=sum(int(x["n"]) for x in blocks)
    return {k:float(sum(int(x["n"])*x[side][k] for x in blocks)/total) for k in ("accuracy","logloss","brier","ece","mean_confidence")} | {"n":total}


def _evaluate_holdout(rows,horizon,dev):
    split=int(len(rows)*(1-FINAL_HOLDOUT_FRAC))
    development=rows[:split]; holdout=rows[split:]
    gap=int(PURGE_BARS[horizon]+EMBARGO_BARS[horizon])
    meta_end=len(development)-gap
    meta_start=max(MIN_TRAIN,meta_end-META_BLOCK)
    train=development[:meta_start]; meta=development[meta_start:meta_end]
    if len(train)<MIN_TRAIN or len(meta)<MIN_META_ROWS or len(holdout)<200:
        return {"status":"DEFERRED","reason":"insufficient_holdout_training"}
    meta_models=_fit(train)
    meta_probs=_expert_probs(meta_models,meta,horizon)
    threshold=_gate_threshold(_uncertainty(meta_probs))
    weights=_advantage_weights(meta_probs,meta,_uncertainty(meta_probs)>=threshold)
    hold_models=_fit(development)
    hold_probs=_expert_probs(hold_models,holdout,horizon)
    unc=_uncertainty(hold_probs)
    votes=np.stack([np.argmax(hold_probs[e],axis=1) for e in ALTS],axis=0)
    champ=np.argmax(hold_probs["production"],axis=1)
    disagreement=np.sum(votes!=champ[None,:],axis=0)
    gate=(unc>=threshold)&(disagreement>=MIN_DISAGREE)
    final,routing=_mix(hold_probs,gate,weights)
    y=[r["y"] for r in holdout]
    base=_metrics(y,hold_probs["production"]); cand=_metrics(y,final)
    gate_idx=np.flatnonzero(gate)
    return {
        "status":"OK","n":len(holdout),"gate_n":int(gate.sum()),"gate_rate":float(gate.mean()),
        "threshold":threshold,"weights":weights,
        "baseline":base,"candidate":cand,
        "delta":{k:float(cand[k]-base[k]) for k in ("accuracy","logloss","brier","ece")},
        "high_uncertainty_disagreement":{
            "n":int(gate.sum()),
            "baseline":_metrics([y[i] for i in gate_idx],hold_probs["production"][gate]) if len(gate_idx) else None,
            "candidate":_metrics([y[i] for i in gate_idx],final[gate]) if len(gate_idx) else None
        },
        "used_for_selection":False
    }


def evaluate(horizon):
    rows=load_archive_research_rows(horizon,MAX_ROWS)
    if len(rows)<MIN_TRAIN+META_BLOCK+TEST_BLOCK+200:
        return {"status":"DEFERRED","reason":"insufficient_archive_rows","n":len(rows)}
    dev_end=int(len(rows)*(1-FINAL_HOLDOUT_FRAC))
    dev=_evaluate_development(rows[:dev_end],horizon)
    hold=_evaluate_holdout(rows,horizon,dev)
    eligible=False
    if dev.get("status")=="OK":
        ri=dev["relative_improvement"]; st=dev["stability"]
        eligible=bool(
            ri["logloss"]>=0.03 and ri["brier"]>=0.01 and st["improved_logloss_ratio"]>=0.70
            and st["improved_brier_ratio"]>=0.70 and st["non_worse_accuracy_ratio"]>=0.70
        )
    return {
        "status":"OK","schema_version":1,"research_only":True,"production_changed":False,
        "final_holdout_protected":True,"final_holdout_used_for_selection":False,
        "strict_point_in_time_archive_replay":False,"horizon":horizon,"n":len(rows),
        "config":{"max_alt_mass":MAX_ALT_MASS,"advantage_scale":ADVANTAGE_SCALE,"min_advantage":MIN_ADVANTAGE,
                  "uncertainty_quantile":UNCERTAINTY_QUANTILE,"min_disagreement":MIN_DISAGREE},
        "development":dev,"final_holdout":hold,"eligibility":eligible
    }


def main():
    payload={"schema_version":1,"research_only":True,"production_changed":False,
             "horizons":{h:evaluate(h) for h in HORIZONS}}
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps(payload,indent=2,sort_keys=True))

if __name__=="__main__":
    main()
