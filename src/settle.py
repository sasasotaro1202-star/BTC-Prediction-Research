import json, sqlite3
from datetime import datetime, timezone
from urllib.request import urlopen, Request
from db import DB, init_db


def price():
    req=Request('https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT',headers={'User-Agent':'btc-prediction-research/1.0'})
    with urlopen(req,timeout=15) as r: return float(json.loads(r.read())['price'])

def direction(base, actual, threshold=0.00015):
    r=actual/base-1
    if r > threshold: return 'UP'
    if r < -threshold: return 'DOWN'
    return 'FLAT'

def settle():
    init_db(); now=datetime.now(timezone.utc); px=price()
    with sqlite3.connect(DB) as con:
        rows=con.execute('SELECT * FROM predictions WHERE (actual_price_5m IS NULL AND target_5m <= ?) OR (actual_price_10m IS NULL AND target_10m <= ?)',(now.isoformat(),now.isoformat())).fetchall()
        for r in rows:
            if r[16] is None and r[2] <= now.isoformat():
                d=direction(r[4],px); pred=max(('UP',r[5]),('DOWN',r[6]),('FLAT',r[7]),key=lambda x:x[1])[0]
                con.execute('UPDATE predictions SET actual_price_5m=?,actual_direction_5m=?,correct_5m=?,settled_5m_at_utc=? WHERE prediction_id=?',(px,d,int(d==pred),now.isoformat(),r[0]))
            if r[17] is None and r[3] <= now.isoformat():
                d=direction(r[4],px); pred=max(('UP',r[8]),('DOWN',r[9]),('FLAT',r[10]),key=lambda x:x[1])[0]
                con.execute('UPDATE predictions SET actual_price_10m=?,actual_direction_10m=?,correct_10m=?,settled_10m_at_utc=? WHERE prediction_id=?',(px,d,int(d==pred),now.isoformat(),r[0]))
    print('settled',len(rows),'price',px)

if __name__=='__main__': settle()
