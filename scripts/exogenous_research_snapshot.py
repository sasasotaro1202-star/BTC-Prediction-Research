"""Collect free, research-only exogenous BTC information snapshots.

No production model/state is touched. GDELT's indexed/seen timestamp is used
as the conservative availability timestamp. BLS observations are tagged with
retrieval time, so they are safe for live diagnostics but are NOT historical
release-time truth.
"""
from __future__ import annotations
import argparse, json, urllib.parse, urllib.request
from datetime import datetime, timezone
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from exogenous_information import InformationEvent, source_metadata
from exogenous_sources import _parse_seen_time

UA="BTC-Prediction-Research/exogenous-research"
OUT=Path("data/exogenous_research")
GDELT="https://api.gdeltproject.org/api/v2/doc/doc"
BLS="https://api.bls.gov/publicAPI/v1/timeseries/data/"

def _get_json(url: str) -> dict:
    req=urllib.request.Request(url,headers={"User-Agent":UA})
    with urllib.request.urlopen(req,timeout=30) as r:
        return json.loads(r.read().decode("utf-8"))

def _now() -> datetime:
    return datetime.now(timezone.utc)

def fetch_news() -> list[InformationEvent]:
    params={"query":"(bitcoin OR btc OR cryptocurrency)","mode":"ArtList","maxrecords":"75","timespan":"1h","sort":"datedesc","format":"json"}
    payload=_get_json(GDELT+"?"+urllib.parse.urlencode(params))
    events=[]
    for item in payload.get("articles",[]):
        seen=item.get("seendate"); url=item.get("url")
        available=_parse_seen_time(seen) if seen else None
        if available is None or not url:
            continue
        events.append(InformationEvent(
            source="gdelt_doc_2", event_id=url, published_at=available,
            available_at=available, event_type="news", importance=1.0))
    return events

def fetch_bls() -> list[InformationEvent]:
    year=str(_now().year)
    payload=_get_json(BLS+"?"+urllib.parse.urlencode({
        "seriesid":"CUSR0000SA0,LNS14000000","startyear":year,"endyear":year}))
    available=_now(); events=[]
    for series in payload.get("Results",{}).get("series",[]):
        sid=series.get("seriesID")
        for item in series.get("data",[]):
            period=item.get("period","")
            if not sid or not period.startswith(("M","S")):
                continue
            events.append(InformationEvent(
                source="bls_public_api_v1",
                event_id=f"{sid}:{item.get('year')}:{period}",
                published_at=available, available_at=available,
                event_type="macro", importance=0.8))
    return events

def write_events(name: str, events: list[InformationEvent]) -> None:
    OUT.mkdir(parents=True,exist_ok=True)
    path=OUT/name
    rows=[]
    for event in events:
        event.validate()
        rows.append({**source_metadata(event),"importance":event.importance,
                     "sentiment":event.sentiment,"surprise":event.surprise})
    path.write_text("\n".join(json.dumps(x,sort_keys=True) for x in rows)+
                    ("\n" if rows else ""))

def main() -> None:
    p=argparse.ArgumentParser()
    p.add_argument("--news",action="store_true")
    p.add_argument("--macro",action="store_true")
    args=p.parse_args()
    if not args.news and not args.macro: args.news=args.macro=True
    if args.news: write_events("news.jsonl",fetch_news())
    if args.macro: write_events("macro.jsonl",fetch_bls())

if __name__=="__main__":
    main()
