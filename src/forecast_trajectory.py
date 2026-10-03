"""Chart-ready BTC forecast trajectories from the canonical prediction ledger."""
from __future__ import annotations
import argparse, json, sqlite3
from datetime import datetime, timezone
from pathlib import Path
try:
    from db import DB, init_db
    from horizon_registry import ALL_HORIZONS, PRIMARY_HORIZONS, EXTENDED_RESEARCH_HORIZONS, HORIZON_MINUTES
except ModuleNotFoundError:
    from src.db import DB, init_db
    from src.horizon_registry import ALL_HORIZONS, PRIMARY_HORIZONS, EXTENDED_RESEARCH_HORIZONS, HORIZON_MINUTES

OUT = Path(DB).parent / "historical_research" / "forecast_trajectory.json"

def _num(v):
    if v is None: return None
    try: x=float(v)
    except (TypeError, ValueError): return None
    return x if x == x and abs(x) != float("inf") else None

def _ensure_horizon_columns(con, horizons):
    """Make the chart exporter tolerant of restored pre-extension DB archives."""
    existing = {str(row[1]) for row in con.execute("PRAGMA table_info(predictions)").fetchall()}
    for horizon in horizons:
        for prefix, sql_type in (
            ("target_", "TEXT"),
            ("p_up_", "REAL"),
            ("p_down_", "REAL"),
            ("p_flat_", "REAL"),
            ("actual_price_", "REAL"),
            ("actual_direction_", "TEXT"),
            ("correct_", "INTEGER"),
        ):
            name = prefix + horizon
            if name not in existing:
                con.execute(f"ALTER TABLE predictions ADD COLUMN {name} {sql_type}")
                existing.add(name)
        settlement_name = f"settled_{horizon}_at_utc"
        if settlement_name not in existing:
            con.execute(f"ALTER TABLE predictions ADD COLUMN {settlement_name} TEXT")
            existing.add(settlement_name)


def build(limit=2880, horizons=ALL_HORIZONS):
    init_db()
    horizons=tuple(horizons)
    bad=[h for h in horizons if h not in ALL_HORIZONS]
    if bad: raise ValueError("unsupported_horizon:" + ",".join(bad))
    with sqlite3.connect(DB) as con:
        _ensure_horizon_columns(con, horizons)
        sql=("SELECT prediction_id,created_at_utc,base_price," +
             ",".join(f"target_{h},p_down_{h},p_flat_{h},p_up_{h},actual_price_{h},actual_direction_{h},correct_{h},settled_{h}_at_utc" for h in horizons) +
             " FROM predictions ORDER BY prediction_id DESC" +
             ("" if limit is None else f" LIMIT {int(limit)}"))
        rows=con.execute(sql).fetchall()
    rows.reverse()
    points={h:[] for h in horizons}
    for row in rows:
        prediction_id,created_at,base_price=row[:3]; pos=3
        for h in horizons:
            target,down,flat,up,actual_price,actual_direction,correct,settled=row[pos:pos+8]; pos+=8
            probs=[_num(down),_num(flat),_num(up)]
            if any(v is None for v in probs): continue
            total=sum(probs)
            if total<=0: continue
            probs=[v/total for v in probs]
            points[h].append({
                "prediction_id":int(prediction_id),"prediction_time_utc":created_at,
                "target_time_utc":target,"horizon_minutes":int(HORIZON_MINUTES[h]),
                "base_price":_num(base_price),"p_down":probs[0],"p_flat":probs[1],"p_up":probs[2],
                "predicted_direction":("DOWN","FLAT","UP")[max(range(3),key=lambda i:probs[i])],
                "actual_price":_num(actual_price),"actual_direction":actual_direction,
                "correct":None if correct is None else int(correct),"settled_at_utc":settled,
                "settled":actual_direction is not None,
                "research_only":h in EXTENDED_RESEARCH_HORIZONS,
                "production_priority":0 if h=="5m" else 1 if h=="10m" else 2,
            })
    return {
        "schema_version":1,"generated_at_utc":datetime.now(timezone.utc).isoformat(),
        "source":"predictions.db","source_of_truth":"canonical_prediction_ledger",
        "primary_horizons":list(PRIMARY_HORIZONS),
        "extended_research_horizons":list(EXTENDED_RESEARCH_HORIZONS),
        "horizon_order":list(ALL_HORIZONS),"default_window_predictions":limit,
        "no_missing_as_zero":True,"research_only_extended":True,
        "chart_semantics":{"x":"prediction_time_utc","probability_series":["p_down","p_flat","p_up"],
                           "outcome_series":["actual_price","actual_direction","correct"],"target":"target_time_utc"},
        "horizons":{h:{"horizon_minutes":int(HORIZON_MINUTES[h]),
                       "production_priority":"primary_5m" if h=="5m" else "primary_10m" if h=="10m" else "research_extended",
                       "research_only":h in EXTENDED_RESEARCH_HORIZONS,
                       "availability_status":"AVAILABLE" if points[h] else "PENDING_FIRST_LIVE_SAMPLE",
                       "points":points[h],"point_count":len(points[h])} for h in horizons}
    }

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--limit",type=int,default=2880)
    ap.add_argument("--horizon",action="append",choices=ALL_HORIZONS)
    ap.add_argument("--output",type=Path,default=OUT)
    a=ap.parse_args()
    limit=None if a.limit==0 else max(1,a.limit)
    hs=tuple(a.horizon) if a.horizon else ALL_HORIZONS
    payload=build(limit,hs); a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(payload,ensure_ascii=False,separators=(",",":")),encoding="utf-8")
    print(json.dumps({"output":str(a.output),"horizon_order":payload["horizon_order"],
                      "point_counts":{h:payload["horizons"][h]["point_count"] for h in hs},"limit":limit},
                     ensure_ascii=False))

if __name__=="__main__": main()
