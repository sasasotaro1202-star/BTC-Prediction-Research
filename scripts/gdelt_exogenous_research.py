"""Research-only GDELT GKG -> PIT-safe BTC information events.

The collector never mutates production prediction state or production feature
schemas. GDELT 2.0 GKG is a free public 15-minute news stream. We use the
GKG slice timestamp as a conservative availability bound and the article
timestamp as published_at. Records without trustworthy timestamps are dropped.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import html
import io
import json
import re
import urllib.request
import zipfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

BASE = "https://data.gdeltproject.org/gdeltv2"
KEYWORDS = (
    "bitcoin", "btc", "crypto", "cryptocurrency", "ethereum", "eth",
    "binance", "coinbase", "digital asset", "spot etf", "bitcoin etf",
    "fed", "federal reserve", "fomc", "interest rate", "inflation", "cpi",
    "jobs report", "nonfarm", "treasury", "sec", "regulation", "regulatory",
    "tariff", "sanction", "stablecoin",
)
POLICY_WORDS = ("sec", "regulation", "regulatory", "sanction", "tariff",
                "stablecoin", "legislation", "law", "ban")
MACRO_WORDS = ("fed", "federal reserve", "fomc", "interest rate", "inflation",
               "cpi", "jobs report", "nonfarm", "treasury", "yield")

def _stamp(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y%m%d%H%M%S")

def _parse_gdelt_dt(value: str) -> datetime | None:
    value = (value or "").strip()
    if not re.fullmatch(r"\d{14}", value):
        return None
    try:
        return datetime.strptime(value, "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
    except ValueError:
        return None

def _download_slice(stamp: datetime, attempts: int = 3) -> bytes:
    url = f"{BASE}/{_stamp(stamp)}.gkg.csv.zip"
    req = urllib.request.Request(url, headers={"User-Agent": "BTC-Prediction-Research/1.0"})
    last = None
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read()
        except Exception as exc:
            last = exc
            if attempt + 1 < attempts:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"download failed after {attempts} attempts: {last}")

def _title(extras: str) -> str:
    m = re.search(r"<PAGE_TITLE>(.*?)</PAGE_TITLE>", extras or "", flags=re.S)
    return html.unescape(m.group(1)).strip() if m else ""

def _tone(v: str) -> float | None:
    try:
        x = float((v or "").split(",")[0])
    except (ValueError, IndexError):
        return None
    # GDELT tone is not bounded to [-1,1]; normalize conservatively.
    return max(-1.0, min(1.0, x / 10.0))

def parse_slice(raw: bytes, available_at: datetime) -> list[dict]:
    out: list[dict] = []
    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        names = zf.namelist()
        if not names:
            raise ValueError("GDELT zip contains no file")
        with zf.open(names[0]) as fh:
            text = io.TextIOWrapper(fh, encoding="utf-8", errors="replace")
            for row in csv.reader(text, delimiter="\t"):
                if len(row) < 27:
                    continue
                published = _parse_gdelt_dt(row[1])
                if published is None or available_at < published:
                    # Fail closed rather than inventing chronology.
                    continue
                url = row[4].strip()
                title = _title(row[26])
                blob = f"{title} {url} {row[8]} {row[17]} {row[18]}".lower()
                if not any(k in blob for k in KEYWORDS):
                    continue
                if any(k in blob for k in POLICY_WORDS):
                    event_type = "policy"
                elif any(k in blob for k in MACRO_WORDS):
                    event_type = "macro"
                else:
                    event_type = "news"
                event_id = hashlib.sha256(
                    f"gdelt-gkg|{row[0]}|{url}".encode("utf-8")
                ).hexdigest()
                out.append({
                    "source": "GDELT_GKG",
                    "event_id": event_id,
                    "published_at": published.isoformat(),
                    "available_at": available_at.isoformat(),
                    "event_type": event_type,
                    "importance": 0.5,
                    "sentiment": _tone(row[15]),
                    "surprise": None,
                    "title": title,
                    "url": url,
                    "source_name": row[3].strip(),
                    "research_only": True,
                    "production_changed": False,
                    "final_holdout_protected": True,
                })
    return out

def _coverage_path(output: Path) -> Path:
    return output.with_name("gdelt_coverage.json")


def _load_coverage(path: Path) -> set[str]:
    if not path.exists():
        return set()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return {str(x) for x in data.get("successful_slices", [])}
    except Exception as exc:
        raise RuntimeError(f"existing GDELT coverage manifest is malformed: {exc}") from exc


def _slice_stamps_for_window(prediction_time: datetime, lookback: timedelta = timedelta(hours=1)) -> list[str]:
    if prediction_time.tzinfo is None or prediction_time.utcoffset() is None:
        raise ValueError("prediction_time must be timezone-aware")
    if lookback.total_seconds() < 0:
        raise ValueError("lookback must be non-negative")
    end = prediction_time.astimezone(timezone.utc).replace(second=0, microsecond=0)
    end = end.replace(minute=(end.minute // 15) * 15)
    start = (end - lookback).replace(minute=((end - lookback).minute // 15) * 15)
    cur = start
    out = []
    while cur <= end:
        out.append(_stamp(cur))
        cur += timedelta(minutes=15)
    return out


def coverage_complete(coverage: set[str], prediction_time: datetime, lookback: timedelta = timedelta(hours=1)) -> bool:
    return set(_slice_stamps_for_window(prediction_time, lookback)).issubset(coverage)


def collect(start: datetime, end: datetime, output: Path) -> dict:
    if start.tzinfo is None or end.tzinfo is None:
        raise ValueError("start/end must be timezone-aware")
    start = start.astimezone(timezone.utc)
    end = end.astimezone(timezone.utc)
    if end < start:
        raise ValueError("end before start")
    start = start.replace(minute=(start.minute // 15) * 15, second=0, microsecond=0)
    end = end.replace(minute=(end.minute // 15) * 15, second=0, microsecond=0)
    output.parent.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    records: list[dict] = []
    coverage_path = _coverage_path(output)
    coverage = _load_coverage(coverage_path)
    # Reuse the restored research cache so hourly runs accumulate a bounded
    # historical event archive instead of discarding prior slices.
    if output.exists():
        try:
            with output.open(encoding="utf-8") as fh:
                for line in fh:
                    if not line.strip():
                        continue
                    item = json.loads(line)
                    event_id = item.get("event_id")
                    if event_id and item.get("source") == "GDELT_GKG":
                        seen.add(event_id)
                        records.append(item)
        except Exception as exc:
            raise RuntimeError(f"existing exogenous cache is malformed: {exc}") from exc
    cur = start
    attempted = 0
    missing = 0
    while cur <= end:
        stamp = _stamp(cur)
        # Incremental collection: never redownload a successfully verified
        # slice. Missing/failed slices remain retryable on the next run.
        if stamp in coverage:
            cur += timedelta(minutes=15)
            continue
        attempted += 1
        try:
            batch = parse_slice(_download_slice(cur), cur)
            coverage.add(stamp)
        except Exception as exc:
            # A missing slice is not a data point; record it for audit and continue.
            missing += 1
            print(f"WARN: unavailable GDELT slice {_stamp(cur)}: {exc}")
            cur += timedelta(minutes=15)
            continue
        for item in batch:
            if item["event_id"] not in seen:
                seen.add(item["event_id"])
                records.append(item)
        cur += timedelta(minutes=15)
    retention_cutoff = datetime.now(timezone.utc) - timedelta(days=45)
    records = [item for item in records if datetime.fromisoformat(item["available_at"].replace("Z", "+00:00")) >= retention_cutoff]
    coverage = {
        stamp for stamp in coverage
        if _parse_gdelt_dt(stamp) is not None and _parse_gdelt_dt(stamp) >= retention_cutoff
    }
    # Coverage is an independent provenance contract: missing slices are never
    # interpreted as zero-event observations.
    records.sort(key=lambda x: (x["available_at"], x["event_id"]))
    with output.open("w", encoding="utf-8") as fh:
        for item in records:
            fh.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
    manifest = {
        "schema_version": 1,
        "source": "GDELT_GKG",
        "research_only": True,
        "production_changed": False,
        "final_holdout_protected": True,
        "start_utc": start.isoformat(),
        "end_utc": end.isoformat(),
        "slices_attempted": attempted,
        "slices_unavailable": missing,
        "complete": missing == 0,
        "events": len(records),
    }
    if missing == attempted:
        raise RuntimeError("all requested GDELT slices were unavailable; refusing empty research input")
    coverage_path.write_text(
        json.dumps({
            "schema_version": 1,
            "source": "GDELT_GKG",
            "retention_days": 45,
            "successful_slices": sorted(coverage),
        }, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest["coverage_slices"] = len(coverage)
    manifest["coverage_min_utc"] = min(coverage) if coverage else None
    manifest["coverage_max_utc"] = max(coverage) if coverage else None
    output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=int, default=1)
    ap.add_argument("--output", default="data/exogenous/gdelt_events.jsonl")
    args = ap.parse_args()
    if args.hours < 1 or args.hours > 24:
        raise SystemExit("--hours must be between 1 and 24")
    end = datetime.now(timezone.utc) - timedelta(minutes=15)
    start = end - timedelta(hours=args.hours)
    manifest = collect(start, end, Path(args.output))
    print(json.dumps(manifest, sort_keys=True))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
