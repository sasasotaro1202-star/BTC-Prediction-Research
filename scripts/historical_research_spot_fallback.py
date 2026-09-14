"""Historical research launcher with a verified Binance Vision spot fallback."""
from __future__ import annotations
import csv, hashlib, io, urllib.parse, urllib.request, zipfile
from datetime import datetime, timezone, timedelta
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import historical_research as hr
import historical_research_runner as runner

ARCHIVE_BASES=("https://data.binance.vision","https://s3-ap-northeast-1.amazonaws.com/data.binance.vision")
CACHE=Path("/tmp/btc_prediction_archive_cache"); CACHE.mkdir(parents=True,exist_ok=True)
_ORIGINAL_CHUNK=hr.fetch_klines_chunk


def _download(url,timeout=90):
    req=urllib.request.Request(url,headers={"User-Agent":"BTC-Prediction-Research/12.0"})
    with urllib.request.urlopen(req,timeout=timeout) as r:return r.read()


def _urls(symbol,interval,day,monthly):
    if monthly:
        stamp=day.strftime("%Y-%m")
        path=f"data/spot/monthly/klines/{symbol}/{interval}/{symbol}-{interval}-{stamp}.zip"
    else:
        path=f"data/spot/daily/klines/{symbol}/{interval}/{symbol}-{interval}-{day.isoformat()}.zip"
    return [f"{base}/{path}" for base in ARCHIVE_BASES]


def _cache(url):return CACHE/(hashlib.sha256(url.encode()).hexdigest()+".zip")


def _get(urls):
    last=None
    for url in urls:
        try:
            p=_cache(url); payload=p.read_bytes() if p.exists() else _download(url)
            with zipfile.ZipFile(io.BytesIO(payload)) as zf:
                if zf.testzip() is not None:raise RuntimeError("corrupt zip")
            if not p.exists():p.write_bytes(payload)
            return payload
        except Exception as exc:last=exc
    raise RuntimeError(f"verified Binance Vision spot archive unavailable: {last}")


def _rows(urls,start_ms,end_ms):
    payload=_get(urls); out=[]
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        names=[n for n in zf.namelist() if not n.endswith("/")]
        with zf.open(names[0]) as fh:
            text=io.TextIOWrapper(fh,encoding="utf-8",newline="")
            for row in csv.reader(text):
                if not row:continue
                try:ts=int(float(row[0]))
                except (ValueError,TypeError):continue
                if start_ms<=ts<end_ms:out.append(row)
    return out


def _spot_archive_chunk(symbol,start_ms,end_ms,day,limit):
    current_month=datetime.now(timezone.utc).date().replace(day=1)
    d=datetime.fromtimestamp(start_ms/1000,timezone.utc).date()
    end_day=datetime.fromtimestamp((end_ms-1)/1000,timezone.utc).date()
    rows=[]
    while d<=end_day:
        month_start=d.replace(day=1); month_end=(month_start+timedelta(days=32)).replace(day=1)
        a=max(start_ms,int(datetime.combine(d,datetime.min.time(),tzinfo=timezone.utc).timestamp()*1000))
        b=min(end_ms,int(datetime.combine(d+timedelta(days=1),datetime.min.time(),tzinfo=timezone.utc).timestamp()*1000))
        if month_end<=current_month:
            try:
                ma=max(start_ms,int(datetime.combine(month_start,datetime.min.time(),tzinfo=timezone.utc).timestamp()*1000)); mb=min(end_ms,int(datetime.combine(month_end,datetime.min.time(),tzinfo=timezone.utc).timestamp()*1000))
                rows.extend(_rows(_urls(symbol,"1m",month_start,True),ma,mb)); d=month_end; continue
            except Exception:pass
        rows.extend(_rows(_urls(symbol,"1m",d,False),a,b)); d+=timedelta(days=1)
    dedup={int(r[0]):r for r in rows}; return [dedup[k] for k in sorted(dedup)]


def patched_chunk(symbol,start_ms,end_ms,endpoint,kind,day,limit):
    if endpoint=="/api/v3/klines":
        path=hr._cache_path(kind,symbol,day)
        if path.exists():
            try:return __import__("json").loads(path.read_text())
            except Exception:pass
        try:
            return _spot_archive_chunk(symbol,start_ms,end_ms,day,limit)
        except Exception as archive_exc:
            print(f"[WARN] Binance Vision spot fallback failed for {day}: {archive_exc}; trying original endpoint")
            return _ORIGINAL_CHUNK(symbol,start_ms,end_ms,endpoint,kind,day,limit)
    return _ORIGINAL_CHUNK(symbol,start_ms,end_ms,endpoint,kind,day,limit)


hr.fetch_klines_chunk=patched_chunk
hr.req_json=runner.resilient_req_json
hr.spot_proxy=False

if __name__=="__main__":hr.main()
