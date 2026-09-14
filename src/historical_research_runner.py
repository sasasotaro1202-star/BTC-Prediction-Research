"""Fail-safe launcher for BTC historical research."""
from __future__ import annotations
import csv, hashlib, io, json, urllib.parse, urllib.request, zipfile
from datetime import datetime, timezone, timedelta
from pathlib import Path
import historical_research as hr

USER_AGENT="BTC-Prediction-Research/11.5"
ARCHIVE_BASES=("https://data.binance.vision","https://s3-ap-northeast-1.amazonaws.com/data.binance.vision")
ARCHIVE_SAFETY_DAYS=3
ARCHIVE_CACHE=Path("/tmp/btc_prediction_archive_cache"); ARCHIVE_CACHE.mkdir(parents=True,exist_ok=True)
CORE_ENDPOINT="klines"; OPTIONAL_ENDPOINTS={"markPriceKlines","premiumIndexKlines"}; FALLBACK_ENDPOINTS={CORE_ENDPOINT,*OPTIONAL_ENDPOINTS}
RETRYABLE_HTTP={403,429,451,500,502,503,504}
_ORIGINAL_REQ_JSON=hr.req_json

def _download(url,timeout=90):
    req=urllib.request.Request(url,headers={"User-Agent":USER_AGENT})
    with urllib.request.urlopen(req,timeout=timeout) as r:return r.read()

def _archive_path(symbol,interval,endpoint,day,monthly):
    if monthly:
        ym=day.strftime("%Y-%m"); return f"data/futures/um/monthly/{endpoint}/{symbol}/{interval}/{symbol}-{interval}-{ym}.zip"
    d=day.isoformat(); return f"data/futures/um/daily/{endpoint}/{symbol}/{interval}/{symbol}-{interval}-{d}.zip"

def _candidate_urls(symbol,interval,endpoint,day,monthly):
    p=_archive_path(symbol,interval,endpoint,day,monthly); return [f"{b}/{p}" for b in ARCHIVE_BASES]

def _funding_candidate_urls(symbol,day):
    stamp=day.strftime("%Y-%m"); p=f"data/futures/um/monthly/fundingRate/{symbol}/{symbol}-fundingRate-{stamp}.zip"
    return [f"{b}/{p}" for b in ARCHIVE_BASES]

def _cache_path(url):return ARCHIVE_CACHE/(hashlib.sha256(url.encode()).hexdigest()+".zip")

def _zip_valid(payload):
    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as zf:
            names=[n for n in zf.namelist() if not n.endswith("/")]
            return bool(names) and zf.testzip() is None
    except Exception:return False

def _checksum(url,payload):
    for suffix in (".CHECKSUM",".sha256"):
        try:
            text=_download(url+suffix,30).decode("utf-8",errors="replace"); expected=text.split()[0].strip().lower()
            if len(expected)==64:return hashlib.sha256(payload).hexdigest().lower()==expected
        except Exception:continue
    return _zip_valid(payload)

def _get_zip(urls):
    last=None
    for url in urls:
        try:
            cache=_cache_path(url); payload=cache.read_bytes() if cache.exists() else _download(url)
            if not _checksum(url,payload):raise RuntimeError("archive integrity validation failed")
            if not cache.exists():cache.write_bytes(payload)
            return payload,url
        except Exception as exc:last=exc
    raise RuntimeError(f"archive unavailable: {urls[0]}: {last}")

def _zip_rows(urls,start_ms,end_ms):
    payload,_=_get_zip(urls); rows=[]
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        names=[n for n in zf.namelist() if not n.endswith("/")]
        if not names:return []
        with zf.open(names[0]) as fh:
            text=io.TextIOWrapper(fh,encoding="utf-8",newline="")
            for row in csv.reader(text):
                if not row:continue
                try:ts=int(float(row[0]))
                except (ValueError,TypeError):continue
                if start_ms<=ts<end_ms:rows.append(row)
    return rows

def _funding_zip_rows(urls,start_ms,end_ms):
    payload,_=_get_zip(urls); rows=[]
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        names=[n for n in zf.namelist() if not n.endswith("/")]
        if not names:return []
        with zf.open(names[0]) as fh:
            text=io.TextIOWrapper(fh,encoding="utf-8",newline="")
            reader=csv.reader(text); header=None
            for raw in reader:
                if not raw:continue
                lowered=[str(x).strip().lower() for x in raw]
                if header is None and any("funding" in x or "calc_time" in x for x in lowered):
                    header=lowered; continue
                if header:
                    try:
                        def col(names):
                            for n in names:
                                if n in header:return raw[header.index(n)]
                            return None
                        ts_raw=col(["fundingtime","funding_time","calc_time","timestamp"]); rate_raw=col(["last_funding_rate","fundingrate","funding_rate"])
                        if ts_raw is None or rate_raw is None:continue
                        ts=float(ts_raw)
                        if ts<10_000_000_000:ts*=1000
                        ts=int(ts)
                        if start_ms<=ts<end_ms:rows.append({"symbol":"BTCUSDT","fundingTime":ts,"fundingRate":str(rate_raw)})
                    except (ValueError,TypeError,IndexError):continue
                else:
                    try:
                        ts=float(raw[0]); rate=raw[-1]
                        if ts<10_000_000_000:ts*=1000
                        ts=int(ts)
                        if start_ms<=ts<end_ms:rows.append({"symbol":"BTCUSDT","fundingTime":ts,"fundingRate":str(rate)})
                    except (ValueError,TypeError,IndexError):continue
    dedup={int(r["fundingTime"]):r for r in rows}; return [dedup[k] for k in sorted(dedup)]

def _safe_end():
    now=datetime.now(timezone.utc).replace(second=0,microsecond=0); return int((now-timedelta(days=ARCHIVE_SAFETY_DAYS)).timestamp()*1000)

def _archive_fallback(url,optional=False):
    parsed=urllib.parse.urlsplit(url); qs=urllib.parse.parse_qs(parsed.query)
    symbol=qs.get("symbol",[None])[0]; interval=qs.get("interval",["1m"])[0]; start_ms=int(qs.get("startTime",[0])[0]); requested_end_ms=int(qs.get("endTime",[0])[0]); endpoint=parsed.path.split("/fapi/v1/",1)[-1]
    if not symbol or not start_ms or not requested_end_ms or endpoint not in FALLBACK_ENDPOINTS:
        if optional:return []
        raise RuntimeError("invalid archive fallback request")
    end_ms=min(requested_end_ms,_safe_end())
    if start_ms>=end_ms:return []
    start_day=datetime.fromtimestamp(start_ms/1000,timezone.utc).date(); end_day=datetime.fromtimestamp((end_ms-1)/1000,timezone.utc).date(); current_month=datetime.now(timezone.utc).date().replace(day=1)
    rows=[]; day=start_day
    while day<=end_day:
        month_start=day.replace(day=1); month_end=(month_start+timedelta(days=32)).replace(day=1)
        month_a=max(start_ms,int(datetime.combine(month_start,datetime.min.time(),tzinfo=timezone.utc).timestamp()*1000)); month_b=min(end_ms,int(datetime.combine(month_end,datetime.min.time(),tzinfo=timezone.utc).timestamp()*1000))
        if month_end<=current_month:
            try:rows.extend(_zip_rows(_candidate_urls(symbol,interval,endpoint,month_start,True),month_a,month_b)); day=month_end; continue
            except Exception as exc:print(f"[WARN] monthly archive unavailable: {symbol} {endpoint} {month_start}: {exc}")
        d=day; daily_end=min(end_day,month_end-timedelta(days=1))
        while d<=daily_end:
            a=max(start_ms,int(datetime.combine(d,datetime.min.time(),tzinfo=timezone.utc).timestamp()*1000)); b=min(end_ms,int(datetime.combine(d+timedelta(days=1),datetime.min.time(),tzinfo=timezone.utc).timestamp()*1000))
            try:rows.extend(_zip_rows(_candidate_urls(symbol,interval,endpoint,d,False),a,b))
            except Exception as exc:
                if not optional:raise RuntimeError(f"no verified Binance archive for {symbol} {endpoint} {d}: {exc}") from exc
                print(f"[WARN] optional archive unavailable: {symbol} {endpoint} {d}: {exc}")
            d+=timedelta(days=1)
        day=month_end
    dedup={int(r[0]):r for r in rows}; return [dedup[k] for k in sorted(dedup)]

def _archive_funding_fallback(url):
    """Use completed monthly Binance Vision fundingRate archives only.

    Binance publishes fundingRate as monthly archives. The current month is
    deliberately treated as unavailable; returning the completed history lets
    the research panel carry forward the last observed funding event instead
    of inventing current-month data or failing the entire research run.
    """
    parsed=urllib.parse.urlsplit(url); qs=urllib.parse.parse_qs(parsed.query)
    symbol=qs.get("symbol",["BTCUSDT"])[0]; start_ms=int(qs.get("startTime",[0])[0]); requested_end_ms=int(qs.get("endTime",[0])[0])
    if not start_ms or not requested_end_ms:raise RuntimeError("funding archive fallback requires startTime/endTime")
    end_ms=min(requested_end_ms,_safe_end())
    if start_ms>=end_ms:return []
    start_day=datetime.fromtimestamp(start_ms/1000,timezone.utc).date(); end_day=datetime.fromtimestamp((end_ms-1)/1000,timezone.utc).date(); current_month=datetime.now(timezone.utc).date().replace(day=1)
    out=[]; month=start_day.replace(day=1)
    while month<=end_day:
        month_end=(month+timedelta(days=32)).replace(day=1)
        if month>=current_month:
            print(f"[WARN] Binance Vision fundingRate current-month archive not published; no new funding observations for {month}.")
            month=month_end; continue
        a=max(start_ms,int(datetime.combine(month,datetime.min.time(),tzinfo=timezone.utc).timestamp()*1000)); b=min(end_ms,int(datetime.combine(month_end,datetime.min.time(),tzinfo=timezone.utc).timestamp()*1000))
        try:out.extend(_funding_zip_rows(_funding_candidate_urls(symbol,month),a,b))
        except Exception as exc:print(f"[WARN] monthly funding archive unavailable: {symbol} {month}: {exc}")
        month=month_end
    dedup={int(r["fundingTime"]):r for r in out}; return [dedup[k] for k in sorted(dedup)]

def _bybit_funding_fallback(url):
    parsed=urllib.parse.urlsplit(url); qs=urllib.parse.parse_qs(parsed.query); symbol=qs.get("symbol",["BTCUSDT"])[0]; start_ms=int(qs.get("startTime",[0])[0]); end_ms=int(qs.get("endTime",[0])[0])
    if not start_ms or not end_ms:raise RuntimeError("funding fallback requires startTime/endTime")
    out=[]; cursor_end=end_ms
    for _ in range(20):
        q=urllib.parse.urlencode({"category":"linear","symbol":symbol,"startTime":start_ms,"endTime":cursor_end,"limit":200}); payload=json.loads(_download(f"https://api.bybit.com/v5/market/funding/history?{q}",30))
        if payload.get("retCode") not in (0,None):raise RuntimeError(f"Bybit funding API error: {payload.get('retCode')} {payload.get('retMsg')}")
        batch=payload.get("result",{}).get("list",[]) or []
        if not batch:break
        for r in batch:
            ts=int(r["fundingRateTimestamp"])
            if start_ms<=ts<end_ms:out.append({"symbol":symbol,"fundingTime":ts,"fundingRate":r["fundingRate"]})
        oldest=min(int(r["fundingRateTimestamp"]) for r in batch)
        if oldest<=start_ms or len(batch)<200:break
        cursor_end=oldest-1
    dedup={int(r["fundingTime"]):r for r in out}; return [dedup[k] for k in sorted(dedup)]

def resilient_req_json(url,timeout=30,retries=5):
    try:return _ORIGINAL_REQ_JSON(url,timeout=timeout,retries=retries)
    except RuntimeError as exc:
        message=str(exc)
        if "/fapi/v1/" not in url or not any(f"HTTP Error {c}" in message for c in RETRYABLE_HTTP):raise
        endpoint=url.split("/fapi/v1/",1)[1].split("?",1)[0]
        if endpoint=="fundingRate":
            print("[WARN] Binance fundingRate unavailable; using verified free Binance Vision monthly funding archive.")
            archived=_archive_funding_fallback(url)
            if archived:return archived
            print("[WARN] No completed Binance funding archive covers this window; trying free Bybit funding-history fallback.")
            return _bybit_funding_fallback(url)
        if endpoint==CORE_ENDPOINT:return _archive_fallback(url,optional=False)
        if endpoint in OPTIONAL_ENDPOINTS:return _archive_fallback(url,optional=True)
        raise

hr.req_json=resilient_req_json
if __name__=="__main__":hr.main()
