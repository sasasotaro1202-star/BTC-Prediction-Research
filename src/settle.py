import json, sqlite3
from datetime import datetime, timezone
from urllib.request import urlopen, Request
from db import DB, init_db

THRESHOLD = 0.00015


def target_price(target_iso):
    target=datetime.fromisoformat(target_iso.replace('Z','+00:00'))
    # Binance 1m candle close is the price at the end of that minute.
    # Query the candle immediately before the target so its close is target time.
    target_ms=int(target.timestamp()*1000)
    start_ms=target_ms-60000
    url=(f'https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=1m'
         f'&startTime={start_ms}&endTime={target_ms}&limit=2')
    req=Request(url,headers={'User-Agent':'btc-prediction-research/1.1'})
    with urlopen(req,timeout=15) as r:
        rows=json.loads(r.read())
    if not rows:
        return None
    # Prefer the candle whose close time equals the target timestamp.
    for row in rows:
        if int(row[6]) >= target_ms:
            return float(row[4])
    return float(rows[-1][4])


def direction(base, actual, threshold=THRESHOLD):
    r=actual/base-1
    if r > threshold: return 'UP'
    if r < -threshold: return 'DOWN'
    return 'FLAT'


def settle():
    init_db(); now=datetime.now(timezone.utc)
    with sqlite3.connect(DB) as con:
        rows=con.execute('SELECT * FROM predictions WHERE (actual_price_5m IS NULL AND target_5m <= ?) OR (actual_price_10m IS NULL AND target_10m <= ?)',(now.isoformat(),now.isoformat())).fetchall()
        settled=0
        for r in rows:
            if r[16] is None and r[2] <= now.isoformat():
                px=target_price(r[2])
                if px is not None:
                    d=direction(r[4],px); pred=max((('UP',r[5]),('DOWN',r[6]),('FLAT',r[7])),key=lambda x:x[1])[0]
                    con.execute('UPDATE predictions SET actual_price_5m=?,actual_direction_5m=?,correct_5m=?,settled_5m_at_utc=? WHERE prediction_id=?',(px,d,int(d==pred),now.isoformat(),r[0])); settled+=1
            if r[17] is None and r[3] <= now.isoformat():
                px=target_price(r[3])
                if px is not None:
                    d=direction(r[4],px); pred=max((('UP',r[8]),('DOWN',r[9]),('FLAT',r[10])),key=lambda x:x[1])[0]
                    con.execute('UPDATE predictions SET actual_price_10m=?,actual_direction_10m=?,correct_10m=?,settled_10m_at_utc=? WHERE prediction_id=?',(px,d,int(d==pred),now.isoformat(),r[0])); settled+=1
    print('settled_fields',settled)

if __name__=='__main__': settle()
