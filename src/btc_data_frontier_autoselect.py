"""Autonomous BTC data frontier: discover, acquire, validate, and select free research data."""
from __future__ import annotations
import hashlib,json,math,os,re,time
from datetime import datetime,timezone
from pathlib import Path
from urllib.parse import quote
from urllib.error import HTTPError
from urllib.request import Request,urlopen
from src.btc_source_frontier_catalog import SOURCES

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"data/historical_research/data_frontier.json"
RUN_OUT=ROOT/"data/historical_research/data_frontier_run.json"
STATE_OUT=ROOT/"data/historical_research/data_frontier_state.json"
SNAPSHOT_DIR=ROOT/"data/historical_research/source_snapshots"
MAX_PAYLOAD_BYTES=120_000
MAX_SNAPSHOTS=240
HYPERLIQUID_HISTORY_BATCH_CANDLES=200
HYPERLIQUID_HISTORY_MAX_RETRIES=2
HYPERLIQUID_HISTORY_RETRY_BACKOFF_SEC=1.0
DERIBIT_HISTORY_BATCH_CANDLES=200
DERIBIT_HISTORY_MAX_RETRIES=2
DERIBIT_HISTORY_RETRY_BACKOFF_SEC=1.0
DISCOVERY_RESULTS=8
STRICT_PRIMARY_ACCUMULATION_TARGET=600
HISTORICAL_ACQUISITION_MIN_INTERVAL_SEC=900
MAX_ACQUISITION_FILES=48
ACQUISITION_DIR=ROOT/"data/historical_research/frontier_acquisitions"
ACQUISITION_SOURCE_IDS=("bitget_public_ws","hyperliquid_ws","deribit_public")

# Discovery is deliberately broader than automatic acquisition. These gates
# keep cost/licence uncertainty and PIT uncertainty fail-closed.
FREE_ACCESS_VALUES={"public_free","free_limited","public","free","free_registration"}
BLOCKED_PROVIDER_TERMS={
 "bloomberg","factset","lseg","refinitiv","morningstar","sp global",
 "standard and poor","pitchbook","third bridge","polygon.io",
 "alphavantage","alpha vantage","quandl",
}
DISCOVERY_RESEARCH_STATUSES={"DISCOVERED_UNVERIFIED","ACQUIRED_RESEARCH_ONLY"}

PROBES={
 "hyperliquid_ws":("POST","https://api.hyperliquid.xyz/info",{"type":"metaAndAssetCtxs"}),
 "bitget_public_ws":("GET","https://api.bitget.com/api/v3/market/tickers?category=USDT-FUTURES&symbol=BTCUSDT",None),
 "binance_options_public":("GET","https://eapi.binance.com/eapi/v1/exchangeInfo",None),
 "deribit_public":("GET","https://www.deribit.com/api/v2/public/get_ticker?instrument_name=BTC-PERPETUAL",None),
 "mempool_space":("GET","https://mempool.space/api/v1/fees/recommended",None),
 "brk_bitview":("GET","https://bitview.space/",None),
 "us_treasury_yield_curve":("GET","https://home.treasury.gov/resource-center/data-chart-center/interest-rates/pages/xml",None),
}
DISCOVERY_QUERIES=("bitcoin dataset orderbook historical","bitcoin futures funding open interest dataset","bitcoin onchain dataset historical","BTC options historical dataset","crypto market microstructure dataset","bitcoin news events dataset timestamp","bitcoin liquidation historical dataset public API","bitcoin funding rate historical dataset public","bitcoin open interest historical dataset public","bitcoin cross exchange spread historical dataset","bitcoin 5m OHLCV historical public API","bitcoin block fees mempool historical dataset")
CODE_DISCOVERY_QUERIES=("BTCUSDT filename:csv","bitcoin orderbook filename:parquet","bitcoin funding filename:csv","bitcoin open interest filename:csv","bitcoin liquidation filename:csv","bitcoin OHLCV filename:parquet")
SOURCES_BY_ID={s.source_id:s for s in SOURCES}

def now_utc(): return datetime.now(timezone.utc).replace(microsecond=0).isoformat()

def _atomic_write_json(path: Path, payload: object) -> None:
 tmp = path.with_suffix(path.suffix + ".tmp")
 tmp.write_text(json.dumps(payload,ensure_ascii=False,indent=2,sort_keys=True)+"\n",encoding="utf-8")
 tmp.replace(path)


def _candidate_lifecycle(candidate):
 text_value=_norm(" ".join(
  str(candidate.get(k,""))
  for k in ("name","description","full_name","repository","url","html_url")
 ))
 blocked=any(term in text_value for term in BLOCKED_PROVIDER_TERMS)
 source_url=str(candidate.get("url") or candidate.get("html_url") or "").strip()
 if blocked:
  eligibility="REJECTED_BLOCKED_PROVIDER"
 elif not source_url.startswith("https://"):
  eligibility="REJECTED_NON_HTTPS"
 else:
  eligibility="ELIGIBLE_FOR_RESEARCH_REVIEW"
 free_status=str(candidate.get("free_status") or "").strip().lower()
 if free_status in FREE_ACCESS_VALUES:
  cost_status="VERIFIED_FREE"
 else:
  cost_status="UNCONFIRMED"
 acquisition_adapter=bool(candidate.get("acquisition_adapter_available"))
 if str(candidate.get("candidate_id","")) in {"bitget_public_ws","hyperliquid_ws","deribit_public"}:
  acquisition_adapter=True
 if blocked:
  acquisition_status="REJECTED"
 elif not acquisition_adapter:
  acquisition_status="BLOCKED_UNTIL_VERIFIED_AND_ADAPTER"
 else:
  acquisition_status="RESEARCH_ACQUISITION_ALLOWED"
 return {
  "stage":"REJECTED" if blocked else "DISCOVERED",
  "eligibility":eligibility,
  "cost_status":cost_status,
  "data_feasibility":"METADATA_ONLY",
  "pit_status":"UNVERIFIED",
  "acquisition_status":acquisition_status,
  "research_selection_eligible":eligibility=="ELIGIBLE_FOR_RESEARCH_REVIEW",
  "adoption_status":"RESEARCH_CANDIDATE_ONLY",
  "next_test":(
   "verify_cost_license_access_pit_and_add_safe_adapter"
   if not blocked else
   "no_further_research_unless_policy_changes"
  ),
 }

def _decorate_discovery_candidate(candidate):
 row=dict(candidate)
 row["lifecycle"]=_candidate_lifecycle(row)
 return row

def _discovered_candidate_score(candidate):
 text_value=_norm(" ".join(
  str(candidate.get(k,""))
  for k in ("name","description","query")
 ))
 value=40.0
 if "bitcoin" in text_value or re.search(r"\bbtc\b",text_value): value+=20
 if any(k in text_value for k in ("timestamp","event","publication","api","websocket")): value+=15
 if any(k in text_value for k in ("dataset","historical","archive","csv","parquet")): value+=10
 if candidate.get("license"): value+=5
 # Exploration debt: repeatedly selecting the same candidate loses priority.
 # This is persisted through selection_count in the durable/state branch.
 count=max(0,int(candidate.get("selection_count",0) or 0))
 value+=max(0,12-2*count)
 if not candidate.get("last_selected_at"): value+=5
 return round(value,3)

def _discovery_debt():
 pth=OUT
 if not pth.is_file():
  return 0
 try:
  obj=json.loads(pth.read_text(encoding="utf-8"))
  return sum(
   1 for row in (obj.get("candidates") or {}).values()
   if row.get("production_eligible") is False
   and row.get("status") in DISCOVERY_RESEARCH_STATUSES
   and _candidate_lifecycle(row).get("research_selection_eligible")
  )
 except (OSError,ValueError,TypeError,json.JSONDecodeError):
  return 0

def plan_for_gap(gap):
 gate_gap=max(0,int(gap.get("target",300))-int(gap.get("strict_primary",0)))
 accumulation_gap=max(0,STRICT_PRIMARY_ACCUMULATION_TARGET-int(gap.get("strict_primary",0)))
 secondary={
  "strict_primary_gate":gate_gap,
  "strict_primary_accumulation":accumulation_gap,
  "situation_meta_ready":max(0,int(gap.get("situation_meta_target",3000))-int(gap.get("situation_meta_ready_min",0))),
  "online_expert_ready":max(0,int(gap.get("online_expert_target",140))-int(gap.get("online_expert_ready_min",0))),
  "discovery_pending":max(0,int(gap.get("discovery_pending",0))),
 }
 blocking_reasons=[]
 acquisition_reasons=[]
 if gate_gap>0:
  blocking_reasons.append("strict_primary_gate")
 if accumulation_gap>0:
  acquisition_reasons.append("strict_primary_accumulation")
 if secondary["situation_meta_ready"]>0:
  acquisition_reasons.append("situation_meta_ready")
 if secondary["online_expert_ready"]>0:
  acquisition_reasons.append("online_expert_ready")
 if secondary["discovery_pending"]>0:
  acquisition_reasons.append("discovery_candidate_review")
 needs_more=bool(blocking_reasons or acquisition_reasons)
 if gate_gap>0:
  next_action="collect_live_and_refresh_pit"
 elif accumulation_gap>0:
  next_action="collect_live_and_refresh_pit_for_evidence_margin"
 elif secondary["situation_meta_ready"]>0:
  next_action="warm_binance_ws_and_collect_context"
 elif secondary["online_expert_ready"]>0:
  next_action="continue_live_cycles_for_online_expert"
 elif secondary["discovery_pending"]>0:
  next_action="discover_and_reselect_frontier"
 else:
  next_action="discover_and_reselect_frontier"
 secondary["blocking_reasons"]=blocking_reasons
 secondary["acquisition_reasons"]=acquisition_reasons
 return secondary,needs_more,next_action
def _norm(value): return re.sub(r"\\s+", " ", str(value or "")).strip().lower()

DISCOVERY_DIRECT_SIGNAL_TERMS=(
 "bitcoin","btc","btcusdt","crypto","binance","bitget","hyperliquid","deribit",
 "orderbook","ohlcv","onchain","funding","open interest","open_interest","liquidation",
)

def _has_direct_discovery_signal(candidate):
 text_value=_norm(" ".join(
  str(candidate.get(k,""))
  for k in ("name","description","repository","path","full_name")
 ))
 return any(term in text_value for term in DISCOVERY_DIRECT_SIGNAL_TERMS)

def _get(url,method="GET",body=None,token=None):
 headers={"User-Agent":"BTC-Prediction-Research-data-frontier/1.0","Accept":"application/json,text/plain,*/*"}
 if token:
  headers["Authorization"]="Bearer "+token
 data=None
 if method=="POST": headers["Content-Type"]="application/json"; data=json.dumps(body or {}).encode()
 req=Request(url,headers=headers,method=method,data=data)
 with urlopen(req,timeout=20) as r:
  raw=r.read(MAX_PAYLOAD_BYTES+1)
  if len(raw)>MAX_PAYLOAD_BYTES:
   raise RuntimeError("response_too_large")
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
  "discovery_pending":_discovery_debt(),
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
   "discovery_pending":_discovery_debt(),
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
def _bitget_history_url(end_ms=None, limit=200):
 if end_ms is None:
  end_ms=int(time.time()*1000)
 end_ms=int(end_ms)
 start_ms=end_ms-(5*60*1000*int(limit))
 return ("https://api.bitget.com/api/v2/mix/market/history-candles?"
         "symbol=BTCUSDT&productType=USDT-FUTURES&granularity=5m"
         f"&startTime={start_ms}&endTime={end_ms}&limit={int(limit)}")


def _acquisition_due(state, now=None):
 raw=state.get("last_historical_acquisition_at")
 if not raw:
  return True
 try:
  last=datetime.fromisoformat(str(raw).replace("Z","+00:00"))
  current=datetime.now(timezone.utc) if now is None else now
  return (current-last).total_seconds() >= HISTORICAL_ACQUISITION_MIN_INTERVAL_SEC
 except (TypeError,ValueError):
  return True


def acquire_bitget_history(end_ms=None):
 retrieved=now_utc()
 url=_bitget_history_url(end_ms=end_ms)
 try:
  payload=_get(url)
  rows=payload.get("data") if isinstance(payload,dict) else None
  if not isinstance(rows,list):
   raise RuntimeError("bitget_history_missing_data")
  retrieved_ms=int(time.time()*1000)
  normalized=[]
  for row in rows:
   if not isinstance(row,list) or len(row)<7:
    continue
   try:
    event_ms=int(row[0])
    close_ms=event_ms+5*60*1000
    if event_ms<=0 or close_ms>retrieved_ms:
     continue
    values=[float(row[i]) for i in range(1,7)]
    if not all(math.isfinite(v) for v in values):
     continue
    normalized.append({
     "event_time":datetime.fromtimestamp(event_ms/1000,timezone.utc).isoformat(),
     "close_time":datetime.fromtimestamp(close_ms/1000,timezone.utc).isoformat(),
     "open":values[0],"high":values[1],"low":values[2],"close":values[3],
     "volume":values[4],"quote_volume":values[5],
    })
   except (TypeError,ValueError,OverflowError,IndexError):
    continue
  if not normalized:
   raise RuntimeError("bitget_history_no_closed_rows")
  normalized.sort(key=lambda row:row["event_time"])
  payload_sha=_sha(payload)
  ACQUISITION_DIR.mkdir(parents=True,exist_ok=True)
  stamp=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
  path=ACQUISITION_DIR/f"{stamp}_bitget_5m_history.json"
  record={
   "schema_version":1,"source_id":"bitget_public_ws","transport":"rest_historical_candles",
   "url":url,"status":"OK","research_only":True,"production_eligible":False,
   "pit_status":"UNVERIFIED_POSTHOC","temporal_basis":"posthoc_historical_endpoint",
   "retrieved_at":retrieved,"available_at":retrieved,"prediction_cutoff":retrieved,
   "payload_sha256":payload_sha,"record_count":len(normalized),
   "first_event_time":normalized[0]["event_time"],"last_event_time":normalized[-1]["event_time"],
   "rows":normalized,
  }
  _atomic_write_json(path,record)
  files=sorted(ACQUISITION_DIR.glob("*.json"),key=lambda p:p.stat().st_mtime,reverse=True)
  for stale in files[MAX_ACQUISITION_FILES:]:
   stale.unlink(missing_ok=True)
  return {
   "source_id":"bitget_public_ws","status":"OK","transport":"rest_historical_candles",
   "pit_status":"UNVERIFIED_POSTHOC","production_eligible":False,"retrieved_at":retrieved,
   "record_count":len(normalized),"first_event_time":normalized[0]["event_time"],
   "last_event_time":normalized[-1]["event_time"],"payload_sha256":payload_sha,
   "next_cursor_ms":min(int(datetime.fromisoformat(row["event_time"].replace("Z","+00:00")).timestamp()*1000) for row in normalized),
   "path":str(path.relative_to(ROOT)),
  }
 except Exception as exc:
  return {"source_id":"bitget_public_ws","status":"ERROR","retrieved_at":retrieved,
          "production_eligible":False,"error":f"{type(exc).__name__}:{exc}"}



def _persist_candle_batch(source_id,url,normalized,transport,temporal_basis,retrieved_at):
 normalized=sorted(normalized,key=lambda row:row["event_time"])
 if not normalized:
  raise RuntimeError(f"{source_id}_no_closed_rows")
 payload_sha=_sha({"source_id":source_id,"rows":normalized})
 ACQUISITION_DIR.mkdir(parents=True,exist_ok=True)
 stamp=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
 path=ACQUISITION_DIR/f"{stamp}_{source_id}_history.json"
 record={
  "schema_version":1,"source_id":source_id,"transport":transport,"url":url,
  "status":"OK","research_only":True,"production_eligible":False,
  "pit_status":"UNVERIFIED_POSTHOC","temporal_basis":temporal_basis,
  "retrieved_at":retrieved_at,"available_at":retrieved_at,"prediction_cutoff":retrieved_at,
  "payload_sha256":payload_sha,"record_count":len(normalized),
  "first_event_time":normalized[0]["event_time"],"last_event_time":normalized[-1]["event_time"],
  "rows":normalized,
 }
 _atomic_write_json(path,record)
 files=sorted(ACQUISITION_DIR.glob("*.json"),key=lambda p:p.stat().st_mtime,reverse=True)
 for stale in files[MAX_ACQUISITION_FILES:]:
  stale.unlink(missing_ok=True)
 return {
  "source_id":source_id,"status":"OK","transport":transport,
  "pit_status":"UNVERIFIED_POSTHOC","production_eligible":False,
  "retrieved_at":retrieved_at,"record_count":len(normalized),
  "first_event_time":normalized[0]["event_time"],"last_event_time":normalized[-1]["event_time"],
  "payload_sha256":payload_sha,
  "next_cursor_ms":min(int(datetime.fromisoformat(row["event_time"].replace("Z","+00:00")).timestamp()*1000) for row in normalized),
  "path":str(path.relative_to(ROOT)),
 }


def summarize_acquisition_evidence():
 """Audit research acquisitions without allowing them into model inputs."""
 result={
  "schema_version":1,
  "generated_at":now_utc(),
  "scope":"research_acquisition_evidence_integrity",
  "note":"COMPLETE/PARTIAL/MISSING describe evidence integrity for acquired batches, not historical-dataset completeness.",
  "by_source":{},
  "cross_source":{"unique_event_times":0,"duplicate_event_rows":0,"overlap_event_times":0},
  "invalid_files":[],
 }
 files=sorted(ACQUISITION_DIR.glob("*_history.json")) if ACQUISITION_DIR.exists() else []
 event_sources={}
 for sid in ACQUISITION_SOURCE_IDS:
  result["by_source"][sid]={
   "status":"MISSING","file_count":0,"record_count":0,"valid_row_count":0,
   "invalid_row_count":0,"unique_event_times":0,"duplicate_event_rows":0,
   "first_event_time":None,"last_event_time":None,
  }
 loaded=[]
 for path in files:
  try:
   obj=json.loads(path.read_text(encoding="utf-8"))
   if not isinstance(obj,dict): raise ValueError("record_not_object")
   sid=str(obj.get("source_id") or "")
   if sid not in result["by_source"]:
    continue
   loaded.append((path,obj))
  except (OSError,ValueError,TypeError,json.JSONDecodeError) as exc:
   result["invalid_files"].append({"path":str(path.relative_to(ROOT)),"error":f"{type(exc).__name__}:{exc}"})
 for sid in ACQUISITION_SOURCE_IDS:
  src_state=result["by_source"][sid]
  source_files=[(path,obj) for path,obj in loaded if str(obj.get("source_id") or "")==sid]
  src_state["file_count"]=len(source_files)
  source_event_times=set()
  provenance_complete=True
  for path,obj in source_files:
   src_state["record_count"]+=int(obj.get("record_count",0) or 0)
   if (
    obj.get("status")!="OK"
    or obj.get("research_only") is not True
    or obj.get("production_eligible") is not False
    or obj.get("pit_status")!="UNVERIFIED_POSTHOC"
   ):
    provenance_complete=False
   rows=obj.get("rows") if isinstance(obj.get("rows"),list) else []
   for row in rows:
    try:
     event_time=str(row.get("event_time") or "").strip()
     parsed=datetime.fromisoformat(event_time.replace("Z","+00:00"))
     if parsed.tzinfo is None:
      raise ValueError("event_time_naive")
     values=[float(row[k]) for k in ("open","high","low","close","volume")]
     if not all(math.isfinite(v) for v in values):
      raise ValueError("nonfinite_ohlcv")
    except (AttributeError,KeyError,TypeError,ValueError,OverflowError):
     src_state["invalid_row_count"]+=1
     continue
    src_state["valid_row_count"]+=1
    source_event_times.add(event_time)
    event_sources.setdefault(event_time,set()).add(sid)
    src_state["first_event_time"]=event_time if src_state["first_event_time"] is None else min(src_state["first_event_time"],event_time)
    src_state["last_event_time"]=event_time if src_state["last_event_time"] is None else max(src_state["last_event_time"],event_time)
  src_state["unique_event_times"]=len(source_event_times)
  src_state["duplicate_event_rows"]=max(0,src_state["valid_row_count"]-src_state["unique_event_times"])
  if src_state["file_count"]==0:
   src_state["status"]="MISSING"
  elif src_state["invalid_row_count"]>0 or not provenance_complete or src_state["valid_row_count"]==0:
   src_state["status"]="PARTIAL"
  else:
   src_state["status"]="COMPLETE"
 result["invalid_files_count"]=len(result["invalid_files"])
 result["cross_source"]["unique_event_times"]=len(event_sources)
 result["cross_source"]["overlap_event_times"]=sum(1 for sources in event_sources.values() if len(sources)>1)
 result["cross_source"]["duplicate_event_rows"]=sum(max(0,len(sources)-1) for sources in event_sources.values())
 return result


def _get_hyperliquid_history_payload(body):
 url="https://api.hyperliquid.xyz/info"
 last_error=None
 for attempt in range(HYPERLIQUID_HISTORY_MAX_RETRIES+1):
  try:
   return _get(url,"POST",body)
  except HTTPError as exc:
   if exc.code!=429 or attempt>=HYPERLIQUID_HISTORY_MAX_RETRIES:
    raise
   retry_after=exc.headers.get("Retry-After") if exc.headers else None
   try:
    delay=float(retry_after)
   except (TypeError,ValueError):
    delay=HYPERLIQUID_HISTORY_RETRY_BACKOFF_SEC*(2**attempt)
   time.sleep(min(8.0,max(0.1,delay)))
   last_error=exc
 if last_error is not None:
  raise last_error
 raise RuntimeError("hyperliquid_history_request_failed")


def acquire_hyperliquid_history(end_ms=None,interval="5m"):
 retrieved=now_utc()
 end_ms=int(end_ms or time.time()*1000)
 interval_ms={"1m":60_000,"3m":180_000,"5m":300_000,"15m":900_000}.get(interval,300_000)
 batch_candles=HYPERLIQUID_HISTORY_BATCH_CANDLES
 url="https://api.hyperliquid.xyz/info"
 try:
  for _ in range(3):
   start_ms=end_ms-(interval_ms*batch_candles)
   body={"type":"candleSnapshot","req":{"coin":"BTC","interval":interval,"startTime":start_ms,"endTime":end_ms}}
   try:
    payload=_get_hyperliquid_history_payload(body)
    break
   except RuntimeError as exc:
    if str(exc)!="response_too_large" or batch_candles<=25:
     raise
    batch_candles=max(25,batch_candles//2)
  else:
   raise RuntimeError("hyperliquid_history_payload_unavailable")
  if not isinstance(payload,list):
   raise RuntimeError("hyperliquid_history_invalid_payload")
  retrieved_ms=int(time.time()*1000)
  normalized=[]
  for row in payload:
   if not isinstance(row,dict):
    continue
   try:
    event_ms=int(row.get("t",0))
    close_ms=int(row.get("T",event_ms+interval_ms))
    if event_ms<=0 or close_ms>retrieved_ms:
     continue
    values=[float(row[k]) for k in ("o","h","l","c","v")]
    if not all(math.isfinite(v) for v in values):
     continue
    normalized.append({
     "event_time":datetime.fromtimestamp(event_ms/1000,timezone.utc).isoformat(),
     "close_time":datetime.fromtimestamp(close_ms/1000,timezone.utc).isoformat(),
     "open":values[0],"high":values[1],"low":values[2],"close":values[3],
     "volume":values[4],"trade_count":int(row.get("n",0) or 0),
    })
   except (TypeError,ValueError,OverflowError):
    continue
  result=_persist_candle_batch(
   "hyperliquid_ws",url,normalized,"rest_historical_candles","source_event_time",retrieved
  )
  result["history_window_start_ms"]=start_ms
  result["history_window_end_ms"]=end_ms
  result["history_requested_candles"]=batch_candles
  result["retrieved_at"]=retrieved
  return result
 except Exception as exc:
  return {"source_id":"hyperliquid_ws","status":"ERROR","retrieved_at":retrieved,
          "production_eligible":False,"error":f"{type(exc).__name__}:{exc}"}


def _get_deribit_history_payload(params):
 url="https://www.deribit.com/api/v2/public/get_tradingview_chart_data"
 last_error=None
 for attempt in range(DERIBIT_HISTORY_MAX_RETRIES+1):
  try:
   query="&".join(f"{quote(str(k))}={quote(str(v))}" for k,v in params.items())
   payload=_get(url+"?"+query)
   if not isinstance(payload,dict) or not isinstance(payload.get("result"),dict):
    raise RuntimeError("deribit_history_invalid_payload")
   return payload["result"]
  except HTTPError as exc:
   if exc.code!=429 or attempt>=DERIBIT_HISTORY_MAX_RETRIES:
    raise
   time.sleep(min(8.0,DERIBIT_HISTORY_RETRY_BACKOFF_SEC*(2**attempt)))
   last_error=exc
 if last_error is not None:
  raise last_error
 raise RuntimeError("deribit_history_request_failed")


def acquire_deribit_history(end_ms=None,resolution="5"):
 retrieved=now_utc()
 end_ms=int(end_ms or time.time()*1000)
 interval_ms={"1":60_000,"3":180_000,"5":300_000,"15":900_000}.get(str(resolution),300_000)
 batch_candles=DERIBIT_HISTORY_BATCH_CANDLES
 url="https://www.deribit.com/api/v2/public/get_tradingview_chart_data"
 try:
  for _ in range(3):
   start_ms=end_ms-(interval_ms*batch_candles)
   result=_get_deribit_history_payload({
    "instrument_name":"BTC-PERPETUAL",
    "start_timestamp":start_ms,
    "end_timestamp":end_ms,
    "resolution":resolution,
   })
   ticks=result.get("ticks") or []
   if ticks:
    break
   if batch_candles<=25:
    break
   batch_candles=max(25,batch_candles//2)
  ticks=result.get("ticks") or []
  closes=result.get("close") or []
  opens=result.get("open") or []
  highs=result.get("high") or []
  lows=result.get("low") or []
  volumes=result.get("volume") or []
  if not ticks:
   raise RuntimeError("deribit_history_no_rows")
  normalized=[]
  for i,event_ms in enumerate(ticks):
   try:
    event_ms=int(event_ms)
    close_ms=event_ms+interval_ms
    values=[float(opens[i]),float(highs[i]),float(lows[i]),float(closes[i]),float(volumes[i] if i<len(volumes) else 0.0)]
    if event_ms<=0 or close_ms>int(time.time()*1000) or not all(math.isfinite(v) for v in values):
     continue
    normalized.append({
     "event_time":datetime.fromtimestamp(event_ms/1000,timezone.utc).isoformat(),
     "close_time":datetime.fromtimestamp(close_ms/1000,timezone.utc).isoformat(),
     "open":values[0],"high":values[1],"low":values[2],"close":values[3],
     "volume":values[4],"trade_count":0,
    })
   except (IndexError,TypeError,ValueError,OverflowError):
    continue
  result=_persist_candle_batch("deribit_public",url,normalized,"rest_historical_candles","source_event_time",retrieved)
  result["history_window_start_ms"]=start_ms
  result["history_window_end_ms"]=end_ms
  result["history_requested_candles"]=batch_candles
  result["instrument_name"]="BTC-PERPETUAL"
  result["retrieved_at"]=retrieved
  return result
 except Exception as exc:
  return {"source_id":"deribit_public","status":"ERROR","retrieved_at":retrieved,
          "production_eligible":False,"error":f"{type(exc).__name__}:{exc}"}


def select_auto_acquisition_sources(frontier,selected,gap):
 primary_gap=bool(gap.get("gap",0)>0 or gap.get("strict_primary",0)<STRICT_PRIMARY_ACCUMULATION_TARGET)
 out=[]
 if primary_gap:
  for sid in ("bitget_public_ws","hyperliquid_ws","deribit_public"):
   if sid in selected:
    out.append(sid)
  for sid in ("bitget_public_ws","hyperliquid_ws","deribit_public"):
   if sid not in out:
    out.append(sid)
 for sid in selected:
  row=frontier.get("candidates",{}).get(sid)
  if not row:
   continue
  lifecycle=row.get("lifecycle") or _candidate_lifecycle(row)
  if lifecycle.get("cost_status")=="VERIFIED_FREE" and lifecycle.get("acquisition_adapter_available"):
   out.append(sid)
 return list(dict.fromkeys(out))


def acquire_selected_research_data(frontier,gap,selected):
 candidate_ids=select_auto_acquisition_sources(frontier,selected,gap)
 out=[]
 for sid in candidate_ids[:3]:
  state=frontier["source_state"].setdefault(sid,{})
  if sid=="deribit_public":
   if not _acquisition_due(state):
    out.append({"source_id":sid,"status":"SKIPPED_COOLDOWN","production_eligible":False,
                "last_historical_acquisition_at":state.get("last_historical_acquisition_at")})
    continue
   cursor_ms=int(state.get("last_historical_cursor_ms",0) or 0)
   result=acquire_deribit_history(end_ms=(cursor_ms-1) if cursor_ms>0 else None)
   if result.get("status")=="OK":
    state["last_historical_acquisition_at"]=result["retrieved_at"]
    state["last_historical_record_count"]=int(result.get("record_count",0))
    state["historical_batches_acquired"]=int(state.get("historical_batches_acquired",0))+1
    state["historical_total_records_acquired"]=int(state.get("historical_total_records_acquired",0))+int(result.get("record_count",0))
    state["historical_earliest_event_time"]=state.get("historical_earliest_event_time") or result.get("first_event_time")
    state["last_historical_event_time"]=result.get("last_event_time")
    state["last_historical_cursor_ms"]=int(result.get("next_cursor_ms",0) or 0)
    state["last_historical_payload_sha256"]=result.get("payload_sha256")
    state["historical_acquisition_failures"]=0
   else:
    state["historical_acquisition_failures"]=int(state.get("historical_acquisition_failures",0))+1
   out.append(result)
   continue
  if sid=="hyperliquid_ws":
   if not _acquisition_due(state):
    out.append({"source_id":sid,"status":"SKIPPED_COOLDOWN","production_eligible":False,
                "last_historical_acquisition_at":state.get("last_historical_acquisition_at")})
    continue
   cursor_ms=int(state.get("last_historical_cursor_ms",0) or 0)
   result=acquire_hyperliquid_history(end_ms=(cursor_ms-1) if cursor_ms>0 else None)
   if result.get("status")=="OK":
    state["last_historical_acquisition_at"]=result["retrieved_at"]
    state["last_historical_record_count"]=int(result.get("record_count",0))
    state["historical_batches_acquired"]=int(state.get("historical_batches_acquired",0))+1
    state["historical_total_records_acquired"]=int(state.get("historical_total_records_acquired",0))+int(result.get("record_count",0))
    state["historical_earliest_event_time"]=state.get("historical_earliest_event_time") or result.get("first_event_time")
    state["last_historical_event_time"]=result.get("last_event_time")
    state["last_historical_cursor_ms"]=int(result.get("next_cursor_ms",0) or 0)
    state["last_historical_payload_sha256"]=result.get("payload_sha256")
    state["historical_acquisition_failures"]=0
   else:
    state["historical_acquisition_failures"]=int(state.get("historical_acquisition_failures",0))+1
   out.append(result)
   continue
  if not _acquisition_due(state):
   out.append({"source_id":sid,"status":"SKIPPED_COOLDOWN","production_eligible":False,
               "last_historical_acquisition_at":state.get("last_historical_acquisition_at")})
   continue
  if sid!="bitget_public_ws":
   out.append({"source_id":sid,"status":"SKIPPED_NO_ADAPTER","production_eligible":False})
   continue
  cursor_ms=int(state.get("last_historical_cursor_ms",0) or 0)
  result=acquire_bitget_history(end_ms=(cursor_ms-1) if cursor_ms>0 else None)
  if result.get("status")=="OK":
   state["last_historical_acquisition_at"]=result["retrieved_at"]
   state["last_historical_record_count"]=int(result.get("record_count",0))
   state["historical_batches_acquired"]=int(state.get("historical_batches_acquired",0))+1
   state["historical_total_records_acquired"]=int(state.get("historical_total_records_acquired",0))+int(result.get("record_count",0))
   state["historical_earliest_event_time"]=state.get("historical_earliest_event_time") or result.get("first_event_time")
   state["last_historical_event_time"]=result.get("last_event_time")
   state["last_historical_cursor_ms"]=int(result.get("next_cursor_ms",0) or 0)
   state["last_historical_payload_sha256"]=result.get("payload_sha256")
   state["historical_acquisition_failures"]=0
  else:
   state["historical_acquisition_failures"]=int(state.get("historical_acquisition_failures",0))+1
  out.append(result)
 return out


def _quarantine_discovery_noise(frontier):
 quarantined=0
 for candidate in frontier.get("candidates",{}).values():
  if (
   candidate.get("platform")=="github_code"
   and candidate.get("status") in DISCOVERY_RESEARCH_STATUSES
   and not _has_direct_discovery_signal(candidate)
  ):
   candidate["status"]="REJECTED_DISCOVERY_NOISE"
   candidate["production_eligible"]=False
   candidate["rejection_reason"]="low_signal_code_search_match"
   candidate["lifecycle"]={
    "stage":"REJECTED",
    "eligibility":"REJECTED_DISCOVERY_NOISE",
    "cost_status":"UNCONFIRMED",
    "data_feasibility":"METADATA_ONLY",
    "pit_status":"UNVERIFIED",
    "acquisition_status":"REJECTED",
    "research_selection_eligible":False,
    "adoption_status":"RESEARCH_CANDIDATE_ONLY",
    "next_test":"none_until_new_direct_signal",
   }
   quarantined+=1
 return quarantined

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

 for query in CODE_DISCOVERY_QUERIES:
  url="https://api.github.com/search/code?q="+quote(query)+"&per_page="+str(DISCOVERY_RESULTS)
  try:
   payload=_get(url,token=os.getenv("GITHUB_TOKEN","").strip() or None)
   for item in payload.get("items",[]) if isinstance(payload,dict) else []:
    repo_name=str((item.get("repository") or {}).get("full_name") or "")
    item_path=str(item.get("path") or item.get("name") or "")
    if not repo_name or not item_path: continue
    candidate_id=f"github-code:{repo_name}:{item_path}"
    candidate_probe={
     "candidate_id":candidate_id,
     "name":f"{repo_name}/{item_path}",
     "description":"GitHub code-search lead; file-level candidate requiring acquisition/PIT validation",
     "repository":repo_name,
     "path":item_path,
    }
    if not _has_direct_discovery_signal(candidate_probe):
     continue
    found[candidate_id]={
     "candidate_id":candidate_id,
     "platform":"github_code",
     "name":f"{repo_name}/{item_path}",
     "url":str(item.get("html_url") or ""),
     "description":"GitHub code-search lead; file-level candidate requiring acquisition/PIT validation",
     "query":query,
     "repository":repo_name,
     "path":item_path,
     "license":None,
     "stars":0,
     "status":"DISCOVERED_UNVERIFIED",
     "pit_status":"UNVERIFIED",
     "production_eligible":False,
    }
  except Exception as exc:
   failures.append({"query":query,"platform":"github_code","error":f"{type(exc).__name__}:{exc}"})

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
 return [_decorate_discovery_candidate(row) for row in found.values()], failures

def score(source,state,gap,pr):
 score=100-float(source.priority)*8
 score += 18 if source.access=="public_free" else 8
 score += min(8, max(0, 6-int(state.get("selection_count",0))))
 score += 5 if not state.get("last_probe_at") else 0
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
  if row.get("production_eligible") is not False or row.get("status") not in DISCOVERY_RESEARCH_STATUSES:
   continue
  lifecycle=row.get("lifecycle") or _candidate_lifecycle(row)
  if lifecycle.get("eligibility")!="ELIGIBLE_FOR_RESEARCH_REVIEW":
   continue
  # Discovery selection is research-only. Unknown cost/PIT may be investigated,
  # but cannot auto-enter acquisition or production.
  value=_discovered_candidate_score(row)
  rows.append((value,row["candidate_id"],"discovered"))
 rows.sort(key=lambda x:(-x[0],x[1]))
 discovered_rows=[
  row for row in rows
  if row[1] in frontier.get("candidates",{})
  and (frontier["candidates"].get(row[1]) or {}).get("status") in DISCOVERY_RESEARCH_STATUSES
 ]
 reserved_discovered=None
 if discovered_rows:
  best_discovered=max(discovered_rows,key=lambda x:(x[0],x[1]))
  if best_discovered[0] >= 60:
   reserved_discovered=best_discovered[1]

 # Reserve one exploration slot when qualified discovery debt exists.
 # selection_count lowers priority on repeated picks, preventing starvation.
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
 quarantined_discovery_noise=_quarantine_discovery_noise(frontier)
 for c in discovered:
  old=frontier["candidates"].get(c["candidate_id"],{}); old.update(c); old["first_seen"]=old.get("first_seen",now_utc()); old["last_seen"]=now_utc(); frontier["candidates"][c["candidate_id"]]=old
 selected_before_acquisition=select_sources(frontier,gap,results)
 acquisitions=acquire_selected_research_data(frontier,gap,selected_before_acquisition)
 selected=select_sources(frontier,gap,results)
 selected_discovered=[
  sid for sid in selected
  if sid in frontier.get("candidates",{})
  and (frontier["candidates"].get(sid) or {}).get("status") in {"DISCOVERED_UNVERIFIED","ACQUIRED_RESEARCH_ONLY"}
 ]
 for sid in selected_discovered:
  cand=frontier["candidates"][sid]
  cand["last_selected_at"]=now_utc()
  cand["selection_count"]=int(cand.get("selection_count",0))+1
 for s in SOURCES:
  st=frontier["source_state"].setdefault(s.source_id,{})
  st["selected_for_next_cycle"]=s.source_id in selected
  if s.source_id in selected:
   st["selection_reason"]="data_gap_and_source_diversity" if gap["gap"]>0 else "rotating_frontier_coverage"
   st["selection_count"]=int(st.get("selection_count",0))+1
 acquisition_evidence=summarize_acquisition_evidence()
 secondary_gaps,repeat_until_data_sufficient,next_action=plan_for_gap(gap)
 runrec={
 "run_at":now_utc(),"cycle":cycle+1,"gap":gap,
 "probed_source_ids":probe_ids,
 "selected_source_ids_before_acquisition":selected_before_acquisition,
 "selected_source_ids":selected,
 "reselected_after_acquisition":True,
 "selected_source_ids_after_acquisition":selected,
 "successful_probes":sum(r["status"]=="OK" for r in results.values()),
 "failed_probes":sum(r["status"]=="ERROR" for r in results.values()),
 "new_discovered_candidates":len(discovered),
 "quarantined_discovery_noise":quarantined_discovery_noise,
 "discovery_failures":discovery_failures,
 "production_changed":False,"unknown_pit_policy":"FAIL_CLOSED","free_only":True,
 "snapshots":snapshots,
 "acquisitions":acquisitions,
 "acquisition_evidence":acquisition_evidence,
 "acquisition_totals":{
  "batches":sum(int(frontier["source_state"].get(sid,{}).get("historical_batches_acquired",0)) for sid in ACQUISITION_SOURCE_IDS),
  "records":sum(int(frontier["source_state"].get(sid,{}).get("historical_total_records_acquired",0)) for sid in ACQUISITION_SOURCE_IDS),
  "by_source":{
   sid:{
    "batches":int(frontier["source_state"].get(sid,{}).get("historical_batches_acquired",0)),
    "records":int(frontier["source_state"].get(sid,{}).get("historical_total_records_acquired",0)),
    "earliest_event_time":frontier["source_state"].get(sid,{}).get("historical_earliest_event_time"),
   } for sid in ACQUISITION_SOURCE_IDS
  },
 },
 "data_sufficiency":{
  "promotion_gate_target":int(gap.get("target",300)),
  "accumulation_target":STRICT_PRIMARY_ACCUMULATION_TARGET,
  "remaining":secondary_gaps,
  "blocking_reasons":list(secondary_gaps.get("blocking_reasons",[])),
  "acquisition_reasons":list(secondary_gaps.get("acquisition_reasons",[])),
  "repeat_until_data_sufficient":repeat_until_data_sufficient,
  "stop_when_all_targets_met":True,
 },
 "selection_policy":{
  "reselect_after_acquisition":True,
  "reselect_after_new_data":True,
  "continue_discovery":True,
  "continue_selection":True,
  "persistent_selection_count":True,
  "anti_starvation":"selection_count_penalty",
 },
 "next_best_action":next_action,
 "actions":{
  "collect_live":secondary_gaps["strict_primary_accumulation"]>0 or secondary_gaps["situation_meta_ready"]>0 or secondary_gaps["online_expert_ready"]>0,
  "warm_binance_ws":secondary_gaps["strict_primary_accumulation"]>0 or secondary_gaps["situation_meta_ready"]>0,
  "refresh_pit_audit":secondary_gaps["strict_primary_accumulation"]>0,
  "acquire_historical_archive":secondary_gaps["strict_primary_gate"]>0 or secondary_gaps["strict_primary_accumulation"]>0,
  "acquire_frontier_research_data":bool(acquisitions),
  "continue_discovery":True,
  "continue_selection":True,
  "repeat_until_data_sufficient":repeat_until_data_sufficient,
  "pending_discovery_review":secondary_gaps.get("discovery_pending",0)>0,
  "recompute_after_collection":True,
  "reselect_after_acquisition":True,
 },
 }
 frontier["acquisition_evidence"]=acquisition_evidence
 frontier["discovery_quarantined_total"]=int(frontier.get("discovery_quarantined_total",0))+quarantined_discovery_noise
 frontier["candidate_count"]=len(frontier["candidates"])
 frontier["durable_change"]=bool(discovered) or bool(quarantined_discovery_noise)
 frontier["history"]=list(frontier.get("history") or [])
 frontier["history"].append({"run_at":runrec["run_at"],"cycle":runrec["cycle"],"gap":gap,"probed_source_ids":probe_ids,"selected_source_ids_before_acquisition":selected_before_acquisition,"selected_source_ids":selected,"reselected_after_acquisition":True,"selected_discovered_source_ids":selected_discovered,"successful_probes":runrec["successful_probes"],"failed_probes":runrec["failed_probes"],"new_discovered_candidates":runrec["new_discovered_candidates"],"quarantined_discovery_noise":runrec["quarantined_discovery_noise"],"acquisition_statuses":[x.get("status") for x in acquisitions]})
 frontier["history"]=frontier["history"][-96:]
 durable={
  "schema_version":1,
  "updated_at":now_utc() if discovered else frontier.get("updated_at"),
  "candidate_count":len(frontier["candidates"]),
  "discovery_quarantined_total":int(frontier.get("discovery_quarantined_total",0)),
  "candidates":dict(sorted(frontier["candidates"].items())),
  "acquisition_evidence":acquisition_evidence,
  "policy":{"free_only":True,"production_promotion":False,"unknown_pit":"FAIL_CLOSED"},
 }
 OUT.parent.mkdir(parents=True,exist_ok=True)
 _atomic_write_json(OUT,durable)
 _atomic_write_json(STATE_OUT,{"schema_version":1,"updated_at":now_utc(),"source_state":dict(sorted(frontier["source_state"].items())),"history":frontier["history"],"policy":{"free_only":True,"production_promotion":False,"unknown_pit":"FAIL_CLOSED"}})
 _atomic_write_json(RUN_OUT,runrec)
 files=sorted(SNAPSHOT_DIR.glob("*.json"),key=lambda p:p.stat().st_mtime,reverse=True) if SNAPSHOT_DIR.exists() else []
 for stale in files[MAX_SNAPSHOTS:]: stale.unlink(missing_ok=True)
 return runrec
if __name__=="__main__": print(json.dumps(run(),ensure_ascii=False,indent=2))