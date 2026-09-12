from __future__ import annotations

import hashlib
import io
import urllib.request
import zipfile
from datetime import datetime, timezone, timedelta

BASES = ("https://data.binance.vision", "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision")
UA = "BTC-Prediction-Research/11.2"
SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT")


def get(url: str, timeout: int = 60) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def valid_zip(payload: bytes) -> bool:
    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as zf:
            names = [n for n in zf.namelist() if not n.endswith("/")]
            if not names or zf.testzip() is not None:
                return False
            with zf.open(names[0]) as fh:
                return bool(fh.readline())
    except Exception:
        return False


def verified(url: str) -> bool:
    try:
        payload = get(url)
        for suffix in (".CHECKSUM", ".sha256"):
            try:
                text = get(url + suffix, 30).decode("utf-8", errors="replace")
                expected = text.split()[0].strip().lower()
                if len(expected) == 64:
                    return expected == hashlib.sha256(payload).hexdigest().lower() and valid_zip(payload)
            except Exception:
                pass
        return valid_zip(payload)
    except Exception:
        return False


def main() -> None:
    # Binance publishes new daily data on the following day. Try a short
    # completed-data window so a delayed public-data upload never blocks CI.
    for age in range(3, 8):
        day = (datetime.now(timezone.utc) - timedelta(days=age)).date()
        stamp = day.isoformat()
        ok = True
        for symbol in SYMBOLS:
            path = f"data/futures/um/daily/klines/{symbol}/1m/{symbol}-1m-{stamp}.zip"
            urls = [f"{base}/{path}" for base in BASES]
            if not any(verified(u) for u in urls):
                ok = False
                break
        if ok:
            print(f"OK common completed archive day: {stamp}")
            return
    raise RuntimeError("No common verified BTC/ETH/SOL USD-M 1m daily archive found in the last 7 completed UTC days")


if __name__ == "__main__":
    main()
