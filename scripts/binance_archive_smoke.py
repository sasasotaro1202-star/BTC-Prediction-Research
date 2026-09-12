from __future__ import annotations

import hashlib
import io
import urllib.request
import zipfile
from datetime import datetime, timezone, timedelta

BASE = "https://data.binance.vision"
UA = "BTC-Prediction-Research/10.0"
SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT")


def get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def main() -> None:
    day = (datetime.now(timezone.utc) - timedelta(days=3)).date()
    stamp = day.isoformat()
    for symbol in SYMBOLS:
        path = f"data/futures/um/daily/klines/{symbol}/1m/{symbol}-1m-{stamp}.zip"
        url = f"{BASE}/{path}"
        payload = get(url)
        checksum = get(url + ".CHECKSUM").decode("utf-8", errors="replace").split()[0]
        actual = hashlib.sha256(payload).hexdigest()
        if checksum.lower() != actual.lower():
            raise RuntimeError(f"checksum mismatch: {path}")
        with zipfile.ZipFile(io.BytesIO(payload)) as zf:
            names = [n for n in zf.namelist() if not n.endswith("/")]
            if not names:
                raise RuntimeError(f"empty archive: {path}")
            with zf.open(names[0]) as fh:
                first = fh.readline()
                if not first:
                    raise RuntimeError(f"empty CSV: {path}")
        print(f"OK {symbol} {stamp} {len(payload)} bytes")


if __name__ == "__main__":
    main()
