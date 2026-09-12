import json, sqlite3
from datetime import datetime, timezone
from db import DB, init_db
from market_data import target_close_binance

THRESHOLD=0.00020  # 2 bps

def direction(base,actual,threshold=THRESHOLD):
    r=actual/base-1
    if r>threshold:return 'UP'
    if r<-threshold:return 'DOWN'
    return 'FLAT'

def settle():
    init_db(); now=datetime.now(timezone.utc); settled=0; unavailable=0
    with sqlite3.connect(DB) as con:
        rows=con.execute('SELECT * FROM predictions WHERE (actual_price_5m IS NULL AND target_5m <= ?) OR (actual_price_10m IS NULL AND target_10m <= ?)',(now.isoformat(),now.isoformat())).fetchall()
        for r in rows:
            if r[14] is None and r[2] <= now.isoformat():
                px,source=target_close_binance(r[2])
                if px is not None:
                    d=direction(r[4],px); pred=max((('UP',r[5]),('DOWN',r[6]),('FLAT',r[7])),key=lambda x:x[1])[0]
                    con.execute('UPDATE predictions SET actual_price_5m=?,actual_direction_5m=?,correct_5m=?,settled_5m_at_utc=? WHERE prediction_id=?',(px,d,int(d==pred),now.isoformat(),r[0])); settled+=1
                else: unavailable+=1
            if r[15] is None and r[3] <= now.isoformat():
                px,source=target_close_binance(r[3])
                if px is not None:
                    d=direction(r[4],px); pred=max((('UP',r[8]),('DOWN',r[9]),('FLAT',r[10])),key=lambda x:x[1])[0]
                    con.execute('UPDATE predictions SET actual_price_10m=?,actual_direction_10m=?,correct_10m=?,settled_10m_at_utc=? WHERE prediction_id=?',(px,d,int(d==pred),now.isoformat(),r[0])); settled+=1
                else: unavailable+=1
    print('settled_fields',settled,'unavailable_fields',unavailable,'threshold_bps',THRESHOLD*10000)
if __name__=='__main__':settle()
