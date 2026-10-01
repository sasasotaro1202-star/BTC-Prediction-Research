"""Autonomous BTC data frontier: discover, acquire, validate, and select free research data."""
from __future__ import annotations
import hashlib,json,os,re,time
from datetime import datetime,timezone
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request,urlopen
from src.btc_source_frontier_catalog import SOURCES

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"data/historical_research/data_frontier.json"
RUN_OUT=ROOT/"data/historical_research/data_frontier_run.json"
STATE_OUT=ROOT/"data/historical_research/data_frontier_state.json"
SNAPSHOT_DIR=ROOT/"data/historical_research/source_snapshots"
MAX_PAYLOAD_BYTES=120_000
MAX_SNAPSHOTS=240
DISCOVERY_RESULTS=6
PROBES={
 "hyperliquid_ws":("POST","https://api.hyperliquid.xyz/info",{"type":"metaAndAssetCtxs"}),
 "bitget_public_ws":("GET","https://api.bitget.com/api/v3/market/tickers?category=USDT-FUTURES&symbol=BTCUSDT",None),
 "binance_options_public":("GET","https://eapi.binance.com/eapi/v1/exchangeInfo",None),
 "deribit_public":("GET","https://www.deribit.com/api/v2/public/get_ticker?instrument_name=BTC-PERPETUAL",None),
 "mempool_space":("GET","https://mempool.space/api/v1/fees/recommended",None),
 "brk_bitview":("GET","https://bitview.space/",None),
 "us_treasury_yield_curve":("GET","https://home.treasury.gov/resource-center/data-chart-center/interest-rates/pages/xml",None),
}
DISCOVERY_QUERIES=("bitcoin dataset orderbook historical","bitcoin futures funding open interest dataset","bitcoin onchain dataset historical","BTC options historical dataset","crypto market microstructure dataset","bitcoin news events dataset timestamp")

def now_utc(): return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
def _norm(value): return re.sub(r"\\s+", " ", str(value or "")).strip().lower()
def _get(url,method="GET",body=None,token=None):
 headers={"User-Agent":"BTC-Prediction-Research-data-frontier/1.0","Accept":"application/json,text/plain,*/*"}
 if token:
  headers["Authorization"]="Bearer "+token
 data=None
 if method=="POST": headers["Content-Type"]="application/json"; data=json.dumps(body or {}).encode()
 req=Request(url,headers=headers,method=method,data=data)
 with urlopen(req,timeout=20) as r:
  raw=r.read(MAX_PAYLOAD_BYTES+1)
  if len(raw)>MAX_PAYLOAD_BYTES: raw=raw[:MAX_PAYLOAD_BYTES]
  if "json" in r.headers.get("content-type","") or raw[:1] in (b"{",b"["): return json.loads(raw.decode())
  return {"_text":raw.decode("utf-8",errors="replace")[:MAX_PAYLOAD_BYTES]}
def _sha(v): return hashlib.sha256(json.dumps(v,ensure_ascii=False,sort_keys=True,separators=(",",":")).encode()).hexdigest()
def _count(v):
 if isinstance(v,list): return len(v)
 if isinstance(v,dict):
  for k in ("data","result","events","rows","candles"):
   if isinstance(v.get(k),list): return len(v[k])
 return 1 if v else 0
def _source_ts(v):
 vals=[]
 if isinstance(v,dict):
  vals += [v.get(k) for k in ("ts","timestamp","time","requestTime","serverTime")]
  d=v.get("data")
  if isinstance(d,dict): vals += [d.get(k) for k in ("ts","timestamp","time","requestTime","serverTime")]
  if isinstance(d,list) and d and isinstance(d[0],dict): vals += [d[0].get(k) for k in ("ts","timestamp","time")]
 for x in vals:
  try:
   n=float(x)
   if n>10_000_000_000: return datetime.fromtimestamp(n/1000,timezone.utc).isoformat()
   if n>1_000_000_000: return datetime.fromtimestamp(n,timezone.utc).isoformat()
  except (TypeError,ValueError,OverflowError): pass
 return None
def _empty_frontier():
 return {"schema_version":1,"candidates":{},"source_state":{},"history":[]}

def load_frontier():
 recovery=[]
 if OUT.exists():
  raw=OUT.read_text(encoding="utf-8")
  if raw.strip():
   p=json.loads(raw)
   if not isinstance(p,dict) or p.get("schema_version") not in {1,2}: raise RuntimeError("invalid data frontier schema")
  else:
   p=_empty_frontier()
   recovery.append("empty_durable_frontier_reset")
 else:
  p=_empty_frontier()
 if STATE_OUT.exists():
  raw_state=STATE_OUT.read_text(encoding="utf-8")
  if raw_state.strip():
   state=json.loads(raw_state)
   if not isinstance(state,dict) or state.get("schema_version")!=1: raise RuntimeError("invalid data frontier state schema")
   p["source_state"]=dict(state.get("source_state") or {})
   p["history"]=list(state.get("history") or [])
  else:
   recovery.append("empty_selector_state_ignored")
 p.setdefault("candidates",{}); p.setdefault("source_state",{}); p.setdefault("history",[])
 p["_recovery_events"]=recovery
 return p
def current_gap():
 pth=ROOT/"data/historical_research/pit_oos_audit.json"
 default={
  "strict_primary":0,"target":300,"gap":300,"pit_verified":False,"legacy_unverified":0,
  "situation_meta_ready_min":0,"situation_meta_target":3000,
  "online_expert_ready_min":0,"online_expert_target":140,
 }
 if not pth.is_file():
  return default
 try:
  p=json.loads(pth.read_text(encoding="utf-8"))
  coverage=p.get("coverage") or {}
  situation=[int((coverage.get(h) or {}).get("situation_meta_ready",0)) for h in ("5m","10m")]
  online=[int((coverage.get(h) or {}).get("online_expert_ready",0)) for h in ("5m","10m")]
  strict=int(p.get("verified_primary_predictions",0))
  target=max(300,int(p.get("min_strict_pit_rows",300)))
  return {
   "strict_primary":strict,"target":target,"gap":max(0,target-strict),
   "pit_verified":bool(p.get("pit_verified")),
   "legacy_unverified":int(p.get("legacy_unverified_count",0)),
   "situation_meta_ready_min":min(situation) if situation else 0,
   "situation_meta_target":3000,
   "online_expert_ready_min":min(online) if online else 0,
   "online_expert_target":140,
  }
 except (OSError,ValueError,TypeError,json.JSONDecodeError):
  return default
def probe(sid):
 t=time.monotonic(); retrieved=now_utc(); method,url,body=PROBES[sid]
 try:
  payload=_get(url,method,body); st=_source_ts(payload)
  return {"source_id":sid,"status":"OK","url":url,"retrieved_at":retrieved,"available_at":retrieved,"event_time":st,"temporal_basis":"source_timestamp" if st else "retrieval_snapshot","record_count":_count(payload),"payload_sha256":_sha(payload),"latency_sec":round(time.monotonic()-t,3),"payload":payload}
 except Exception as e:
  return {"source_id":sid,"status":"ERROR","url":url,"retrieved_at":retrieved,"latency_sec":round(time.monotonic()-t,3),"error":f"{type(e).__name__}:{e}"}
def discover_public_sources():
 found={}
 failures=[]
 for query in DISCOVERY_QUERIES:
  url="https://api.github.com/search/repositories?q="+quote(query)+"&sort=updated&order=desc&per_page="+str(DISCOVERY_RESULTS)
  try:
   payload=_get(url,token=os.getenv("GITHUB_TOKEN","").strip() or None)
   for item in payload.get("items",[]) if isinstance(payload,dict) else []:
    name=str(item.get("full_name") or "")
    if name:
     found["github:"+name]={"candidate_id":"github:"+name,"platform":"github","name":name,"url":str(item.get("html_url") or ""),"description":str(item.get("description") or ""),"query":query,"license":((item.get("license") or {}).get("spdx") if isinstance(item.get("license"),dict) else None),"stars":int(item.get("stargazers_count") or 0),"status":"DISCOVERED_UNVERIFIED","pit_status":"UNVERIFIED","production_eligible":False}
  except Exception as exc:
   failures.append({"query":query,"platform":"github","error":f"{type(exc).__name__}:{exc}"})

 for query in DISCOVERY_QUERIES:
  url="https://huggingface.co/api/datasets?search="+quote(query)+"&limit="+str(DISCOVERY_RESULTS)
  try:
   payload=_get(url)
   for item in payload if isinstance(payload,list) else []:
    name=str(item.get("id") or "")
    if not name: continue
    found["huggingface:"+name]={
     "candidate_id":"huggingface:"+name,
     "platform":"huggingface",
     "name":name,
     "url":"https://huggingface.co/datasets/"+name,
     "description":str(item.get("description") or ""),
     "query":query,
     "license":None,
     "stars":int(item.get("likes") or 0),
     "status":"DISCOVERED_UNVERIFIED",
     "pit_status":"UNVERIFIED",
     "production_eligible":False,
    }
  except Exception as exc:
   failures.append({"query":query,"platform":"huggingface","error":f"{type(exc).__name__}:{exc}"})
 return list(found.values()), failures
def score(source,state,gap,pr):
 score=100-float(source.priority)*8
 score += 18 if source.access=="public_free" else 8
 score += 15 if source.pit in {"low","medium"} else -20
 score += 8 if source.realtime else 0
 score += 8 if source.historical else 0
 score += 20 if pr and pr.get("status")=="OK" else (-18 if pr else 0)
 score += min(10,int(state.get("successful_probes",0))*.5)-min(12,int(state.get("consecutive_failures",0))*2)
 if gap["gap"]>0 and source.family in {"exchange_derivatives","options","bitcoin_network","bitcoin_onchain","institutional_derivatives","capital_flow"}: score+=15
 return round(score,3)
def select_sources(frontier,gap,results):
 rows=[(score(s,frontier["source_state"].get(s.source_id,{}),gap,results.get(s.source_id)),s.source_id,s.family) for s in SOURCES]
 for row in frontier.get("candidates",{}).values():
  if row.get("production_eligible") is not False or row.get("status")!="DISCOVERED_UNVERIFIED":
   continue
  text_value=_norm(" ".join((row.get("name",""),row.get("description",""),row.get("query",""))))
  value=40.0
  if "bitcoin" in text_value or re.search(r"\bbtc\b",text_value): value+=20
  if any(k in text_value for k in ("timestamp","event","publication","api","websocket")): value+=15
  if any(k in text_value for k in ("dataset","historical","archive","csv","parquet")): value+=10
  if row.get("license"): value+=5
  rows.append((value,row["candidate_id"],"discovered"))
 rows.sort(key=lambda x:(-x[0],x[1]))
 discovered_rows=[
  row for row in rows
  if row[1] in frontier.get("candidates",{})
  and (frontier["candidates"].get(row[1]) or {}).get("status")=="DISCOVERED_UNVERIFIED"
 ]
 reserved_discovered=None
 if discovered_rows:
  best_discovered=max(discovered_rows,key=lambda x:(x[0],x[1]))
  if best_discovered[0] >= 60:
   reserved_discovered=best_discovered[1]

 # Reserve one exploration slot on every cycle when a sufficiently relevant
 # discovered candidate exists. The candidate remains research-only/unverified.
 selected=[]; families=set()
 if reserved_discovered is not None:
  selected.append(reserved_discovered); families.add("discovered")
 for _,sid,family in rows:
  if family=="discovered" and sid!=reserved_discovered:
   continue
  if family not in families and len(selected)<8:
   selected.append(sid); families.add(family)
 for _,sid,family in rows:
  if len(selected)>=8: break
  if family=="discovered" and sid!=reserved_discovered:
   continue
  if sid not in selected: selected.append(sid)
 return selected
def persist_snapshot(result):
 if result.get("status")!="OK": return None
 SNAPSHOT_DIR.mkdir(parents=True,exist_ok=True); stamp=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
 p=SNAPSHOT_DIR/f"{stamp}_{result['source_id']}.json"; p.write_text(json.dumps(result,ensure_ascii=False,sort_keys=True)+"\n",encoding="utf-8")
 return str(p.relative_to(ROOT))
def run():
 frontier=load_frontier(); recovery_events=list(frontier.pop("_recovery_events",[]) or [])
 gap=current_gap(); cycle=int(datetime.now(timezone.utc).timestamp()//900)
 ids=[s.source_id for s in SOURCES if s.source_id in PROBES]; offset=cycle%max(1,len(ids)); probe_ids=(ids[offset:]+ids[:offset])[:min(5,len(ids))]
 results={sid:probe(sid) for sid in probe_ids}; snapshots=[]
 for sid,r in results.items():
  state=frontier["source_state"].setdefault(sid,{})
  if r["status"]=="OK":
   state["successful_probes"]=int(state.get("successful_probes",0))+1; state["consecutive_failures"]=0; state["last_ok_at"]=r["retrieved_at"]; state["last_snapshot"]=persist_snapshot(r)
  else: state["consecutive_failures"]=int(state.get("consecutive_failures",0))+1
  state["last_probe_at"]=r["retrieved_at"]; state["last_status"]=r["status"]; state["last_latency_sec"]=r["latency_sec"]
  if r.get("event_time"): state["last_source_event_time"]=r["event_time"]
  snapshots.append({k:v for k,v in r.items() if k!="payload"})
 discovered, discovery_failures=discover_public_sources()
 for c in discovered:
  old=frontier["candidates"].get(c["candidate_id"],{}); old.update(c); old["first_seen"]=old.get("first_seen",now_utc()); old["last_seen"]=now_utc(); frontier["candidates"][c["candidate_id"]]=old
 selected=select_sources(frontier,gap,results)
 selected_discovered=[
  sid for sid in selected
  if sid in frontier.get("candidates",{})
  and (frontier["candidates"].get(sid) or {}).get("status")=="DISCOVERED_UNVERIFIED"
 ]
 for sid in selected_discovered:
  cand=frontier["candidates"][sid]
  cand["last_selected_at"]=now_utc()
  cand["selection_count"]=int(cand.get("selection_count",0))+1
 for s in SOURCES:
  st=frontier["source_state"].setdefault(s.source_id,{})
  st["selected_for_next_cycle"]=s.source_id in selected
  if s.source_id in selected: st["selection_reason"]="data_gap_and_source_diversity" if gap["gap"]>0 else "rotating_frontier_coverage"
 runrec={
 "run_at":now_utc(),"cycle":cycle+1,"gap":gap,
 "probed_source_ids":probe_ids,"selected_source_ids":selected,
 "successful_probes":sum(r["status"]=="OK" for r in results.values()),
 "failed_probes":sum(r["status"]=="ERROR" for r in results.values()),
 "new_discovered_candidates":len(discovered),
 "discovery_failures":discovery_failures,
 "production_changed":False,"unknown_pit_policy":"FAIL_CLOSED","free_only":True,
 "snapshots":snapshots,
 secondary_gaps={
  "strict_primary":max(0,int(gap.get("target",300))-int(gap.get("strict_primary",0))),
  "situation_meta_ready":max(0,int(gap.get("situation_meta_target",3000))-int(gap.get("situation_meta_ready_min",0))),
  "online_expert_ready":max(0,int(gap.get("online_expert_target",140))-int(gap.get("online_expert_ready_min",0))),
 }
 repeat_until_data_sufficient=any(value>0 for value in secondary_gaps.values())
 if secondary_gaps["strict_primary"]>0:
  next_action="collect_live_and_refresh_pit"
 elif secondary_gaps["situation_meta_ready"]>0:
  next_action="warm_binance_ws_and_collect_context"
 elif secondary_gaps["online_expert_ready"]>0:
  next_action="continue_live_cycles_for_online_expert"
 else:
  next_action="discover_and_reselect_frontier"
 runrec["data_sufficiency"]={
  "remaining":secondary_gaps,
  "repeat_until_data_sufficient":repeat_until_data_sufficient,
  "stop_when_all_targets_met":True,
 }
 runrec["next_best_action"]=next_action
 runrec["actions"]={
  "collect_live":secondary_gaps["strict_primary"]>0,
  "warm_binance_ws":secondary_gaps["strict_primary"]>0 or secondary_gaps["situation_meta_ready"]>0,
  "refresh_pit_audit":secondary_gaps["strict_primary"]>0,
  "continue_discovery":True,
  "continue_selection":True,
  "repeat_until_data_sufficient":repeat_until_data_sufficient,
  "recompute_after_collection":True,
  "reselect_after_acquisition":True,
 }
 frontier["candidate_count"]=len(frontier["candidates"])
 frontier["durable_change"]=len(discovered)>0
 frontier["history"]=list(frontier.get("history") or [])
 frontier["history"].append({"run_at":runrec["run_at"],"cycle":runrec["cycle"],"gap":gap,"probed_source_ids":probe_ids,"selected_source_ids":selected,"selected_discovered_source_ids":selected_discovered,"successful_probes":runrec["successful_probes"],"failed_probes":runrec["failed_probes"],"new_discovered_candidates":runrec["new_discovered_candidates"]})
 frontier["history"]=frontier["history"][-96:]
 durable={
  "schema_version":1,
  "updated_at":now_utc() if discovered else frontier.get("updated_at"),
  "candidate_count":len(frontier["candidates"]),
  "candidates":dict(sorted(frontier["candidates"].items())),
  "policy":{"free_only":True,"production_promotion":False,"unknown_pit":"FAIL_CLOSED"},
 }
 OUT.parent.mkdir(parents=True,exist_ok=True)
 OUT.write_text(json.dumps(durable,ensure_ascii=False,indent=2,sort_keys=True)+"\n",encoding="utf-8")
 STATE_OUT.write_text(json.dumps({"schema_version":1,"updated_at":now_utc(),"source_state":dict(sorted(frontier["source_state"].items())),"history":frontier["history"],"policy":{"free_only":True,"production_promotion":False,"unknown_pit":"FAIL_CLOSED"}},ensure_ascii=False,indent=2,sort_keys=True)+"\n",encoding="utf-8")
 RUN_OUT.write_text(json.dumps(runrec,ensure_ascii=False,indent=2,sort_keys=True)+"\n",encoding="utf-8")
 files=sorted(SNAPSHOT_DIR.glob("*.json"),key=lambda p:p.stat().st_mtime,reverse=True) if SNAPSHOT_DIR.exists() else []
 for stale in files[MAX_SNAPSHOTS:]: stale.unlink(missing_ok=True)
 return runrec
if __name__=="__main__": print(json.dumps(run(),ensure_ascii=False,indent=2))
