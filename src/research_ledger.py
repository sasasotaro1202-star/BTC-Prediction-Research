"""Compact durable summary for a completed BTC 9H research run.

Descriptive only: this module never selects or promotes production models.
"""
from __future__ import annotations
import json, os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT=Path(__file__).resolve().parents[1]
E=ROOT/"data/historical_research"
OUT=E/"9h_latest_summary.json"
LANES={
"context_router":"context_router_oos.json","interaction":"interaction_oos_report.json",
"robustness":"robustness_oos_report.json","adaptive_ensemble":"adaptive_ensemble_oos.json",
"risk_aware_dynamic":"risk_aware_dynamic_oos.json","rolling_challenger":"rolling_challenger_oos.json",
"extended_features":"extended_features_oos.json","model_zoo_regime":"model_zoo_regime_oos.json",
"recency_weighted":"recency_weighted_oos.json",
"situation_meta":"situation_meta_oos.json",
}
def load(name:str)->dict[str,Any]:
    p=E/name
    if not p.is_file() or p.stat().st_size<=0: raise SystemExit(f"missing research evidence: {p}")
    try: obj=json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc: raise SystemExit(f"invalid research evidence {p}: {type(exc).__name__}") from exc
    if not isinstance(obj,dict): raise SystemExit(f"research evidence root must be object: {p}")
    return obj
def compact(obj:dict[str,Any])->dict[str,Any]:
    out={k:obj.get(k) for k in ("status","research_only","production_changed","final_holdout_protected","promotion_evidence_eligible","policy")}
    if isinstance(obj.get("summary"),dict): out["summary"]=obj["summary"]
    if isinstance(obj.get("final_holdout"),dict): out["final_holdout"]=obj["final_holdout"]
    return out
def build():
    report,pit,ria,artifact,gate=[load(x) for x in ("report.json","pit_oos_audit.json","research_input_audit.json","production_artifact_audit.json","promotion_gate.json")]
    return {
      "schema_version":1,"ledger_type":"btc_9h_research_summary",
      "generated_at_utc":datetime.now(timezone.utc).isoformat(),
      "github_sha":os.environ.get("GITHUB_SHA",""),"workflow_run_id":int(os.environ.get("GITHUB_RUN_ID","0")),
      "workflow_run_attempt":int(os.environ.get("GITHUB_RUN_ATTEMPT","0")),
      "research":{"protocol_version":report.get("protocol_version"),"rows":report.get("rows"),"days":report.get("days"),
        "features_count":len(report.get("features",[])) if isinstance(report.get("features"),list) else None,
        "horizons":{h:{"samples":report.get("horizons",{}).get(h,{}).get("samples"),
                       "class_counts":report.get("horizons",{}).get(h,{}).get("class_counts")}
                   for h in ("5m","10m")}},
      "pit_oos":{k:pit.get(k) for k in ("ok","pit_verified","violation_count","checked_predictions","verified_primary_predictions","min_strict_pit_rows")},
      "research_input_audit":{k:ria.get(k) for k in ("ok","schema_version","data_source")},
      "production_artifact_audit":{"artifacts":artifact.get("artifacts"),"policy":artifact.get("policy")},
      "promotion_gate":{k:gate.get(k) for k in ("production_safety_gate","promotion_allowed","promotion_status","reason","candidate_status","pit_oos_verified","calibration_verified","research_input_audit_verified")},
      "lanes":{lane:compact(load(filename)) for lane,filename in LANES.items()},
    }
def main():
    payload=build()
    if not payload["github_sha"] or payload["workflow_run_id"]<=0: raise SystemExit("research run identity is incomplete")
    OUT.parent.mkdir(parents=True,exist_ok=True); OUT.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps({"ok":True,"bytes":OUT.stat().st_size,"path":str(OUT)},sort_keys=True))
if __name__=="__main__": main()
