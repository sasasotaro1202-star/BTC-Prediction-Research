"""Fail-safe launcher for BTC historical research."""
from __future__ import annotations
import concurrent.futures, csv, hashlib, io, urllib.parse, urllib.request, zipfile
from datetime import datetime, timezone, timedelta
from pathlib import Path
import historical_research as hr

USER_AGENT="BTC-Prediction-Research/11.1"
ARCHIVE_BASES=("https://data.binance.vision","https://s3-ap-northeast-1.amazonaws.com/data.binance.vision")
ARCHIVE_SAFETY_DAYS=3
ARCHIVE_CACHE=Path("data/historical_research/archive_cache"); ARCHIVE_CACHE.mkdir(parents=True,exist_ok=True)
CORE_ENDPOINT="klines"; OPTIONAL_ENDPOINTS={"markPriceKlines","premiumIndexKlines"}; FALLBACK_ENDPOINTS={CORE_ENDPOINT,*OPTIONAL_ENDPOINTS}
RETRYABLE_HTTP={403,429,451,500,502,503,504}
_ORIGINAL_REQ_JSON=hr.req_json
_ZIP_CACHE={}

def _download(url,timeout=90):
    req=urllib.request.Request(url,headers={"User-Agent":USER_AGENT})
    with urllib.request.urlopen(req,timeout=timeout) as r:return r.read()

def _archive_path(symbol,interval,endpoint,day,monthly):
    if monthly:
        ym=day.strftime("%Y-%m"); return f"data/futures/um/monthly/{endpoint}/{symbol}/{interval}/{symbol}-{interval}-{ym}.zip"
    d=day.isoformat(); return f"data/futures/um/daily/{endpoint}/{symbol}/{interval}/{symbol}-{interval}-{d}.zip"

def _candidate_urls(symbol,interval,endpoint,day,monthly):
    p=_archive_path(symbol,interval,endpoint,day,monthly); return [f"{b}/{p}" for b in ARCHIVE_BASES]

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
            _ZIP_CACHE[url]=payload; return payload,url
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

def _safe_end():
    now=datetime.now(timezone.utc).replace(second=0,microsecond=0); return int((now-timedelta(days=ARCHIVE_SAFETY_DAYS)).timestamp()*1000)

def _archive_fallback(url,optional=False):
    parsed=urllib.parse.urlsplit(url); qs=urllib.parse.parse_qs(parsed.query)
    symbol=qs.get("symbol",[None])[0]; interval=qs.get("interval",["1m"])[0]
    start_ms=int(qs.get("startTime",[0])[0]); requested_end_ms=int(qs.get("endTime",[0])[0])
    endpoint=parsed.path.split("/fapi/v1/",1)[-1]
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
            try:
                rows.extend(_zip_rows(_candidate_urls(symbol,interval,endpoint,month_start,True),month_a,month_b)); day=month_end; continue
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

def resilient_req_json(url,timeout=30,retries=5):
    try:return _ORIGINAL_REQ_JSON(url,timeout=timeout,retries=retries)
    except RuntimeError as exc:
        message=str(exc)
        if "/fapi/v1/" not in url or not any(f"HTTP Error {c}" in message for c in RETRYABLE_HTTP):raise
        endpoint=url.split("/fapi/v1/",1)[1].split("?",1)[0]
        if endpoint==CORE_ENDPOINT:return _archive_fallback(url,optional=False)
        if endpoint in OPTIONAL_ENDPOINTS:return _archive_fallback(url,optional=True)
        raise

hr.req_json=resilient_req_json
if __name__=="__main__":hr.main()
