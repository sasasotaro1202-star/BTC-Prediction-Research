"""Research-only selective prediction / abstention analysis for BTC.

Selection thresholds are chosen only on the development walk-forward output.
The protected final holdout is then evaluated once with frozen thresholds.
No production artifact is modified.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"data"/"historical_research"/"selective_prediction_report.json"

def _metrics(y,p):
    y=np.asarray(y); p=np.asarray(p,float)
    pred=p.argmax(1); conf=p.max(1)
    if not len(y): return {"coverage":0.0,"accuracy":None,"logloss":None}
    # Unthresholded summary: every observation is eligible.
    return {"coverage":1.0,"n":int(len(y)),"accuracy":float((pred==y).mean())}

def evaluate(y,p,threshold):
    y=np.asarray(y); p=np.asarray(p,float)
    pred=p.argmax(1); conf=p.max(1)
    mask=conf>=threshold
    if not mask.any():
        return {"coverage":0.0,"n":0,"accuracy":None,"mean_confidence":None}
    return {"coverage":float(mask.mean()),"n":int(mask.sum()),
            "accuracy":float((pred[mask]==y[mask]).mean()),
            "mean_confidence":float(conf[mask].mean())}

def choose_threshold(y,p,target_coverages=(0.50,0.60,0.70,0.80,0.90)):
    y=np.asarray(y); p=np.asarray(p,float); conf=p.max(1)
    candidates=np.unique(np.round(conf,6))
    chosen={}
    for target in target_coverages:
        valid=[(abs(m["coverage"]-target),-m["accuracy"],float(c),m)
               for c in candidates if (m:=evaluate(y,p,float(c)))["n"]>=max(100,int(len(y)*0.05))]
        if valid:
            chosen[str(target)]=min(valid,key=lambda z:z[:2])[3] | {"threshold":min(valid,key=lambda z:z[:2])[2]}
    return chosen

def main():
    # This module is an analysis primitive. It intentionally requires prediction
    # arrays from a caller/research runner rather than fetching data itself.
    raise SystemExit("selective_prediction.py is a research library; call choose_threshold/evaluate from a chronological runner")

if __name__=="__main__": main()
