import math, sqlite3
from datetime import datetime, timezone
from db import DB, init_db


def logloss(y,p): return -math.log(max(1e-12,min(1-1e-12,p))) if y else -math.log(max(1e-12,min(1-1e-12,1-p))

def calibration():
    init_db(); now=datetime.now(timezone.utc)
    with sqlite3.connect(DB) as con:
        rows=con.execute('SELECT p_up_5m,p_down_5m,p_flat_5m,actual_direction_5m,p_up_10m,p_down_10m,p_flat_10m,actual_direction_10m FROM predictions WHERE actual_direction_5m IS NOT NULL').fetchall()
        if not rows: print('No settled predictions yet'); return
        for horizon in (5,10):
            n=len(rows); correct=ll=brier=0.0
            for r in rows:
                off=0 if horizon==5 else 4; ps=r[off:off+3]; y=r[3] if horizon==5 else r[7]
                idx={'UP':0,'DOWN':1,'FLAT':2}[y]; p=ps[idx]
                correct += int(idx==max(range(3),key=lambda i:ps[i]))
                ll += logloss(idx==max(range(3),key=lambda i:ps[i]),p)
                brier += sum((ps[i]-(1 if i==idx else 0))**2 for i in range(3))
            con.execute('INSERT INTO model_metrics(evaluated_at_utc,horizon,model_version,n,accuracy,logloss,brier,calibration_error) VALUES(?,?,?,?,?,?,?,?)',(now.isoformat(),f'{horizon}m','v1.0',n,correct/n,ll/n,brier/n,0.0))
            print(horizon,'m',n,'accuracy',correct/n,'logloss',ll/n,'brier',brier/n)

if __name__=='__main__': calibration()
