import math, sqlite3
from datetime import datetime, timezone
from db import DB, init_db

CLASSES = ['UP','DOWN','FLAT']


def multiclass_metrics(rows, horizon):
    if not rows:
        return None
    probs=[]; ys=[]
    for r in rows:
        ps=[float(r[0]),float(r[1]),float(r[2])]
        y=r[3]
        probs.append(ps); ys.append(y)
    correct=0; ll=0.0; brier=0.0
    confidences=[]; hits=[]
    for ps,y in zip(probs,ys):
        idx=CLASSES.index(y)
        pred=max(range(3),key=lambda i:ps[i])
        correct += int(pred==idx)
        ll += -math.log(max(1e-12,min(1.0,ps[idx])))
        brier += sum((ps[i]-(1.0 if i==idx else 0.0))**2 for i in range(3))
        confidences.append(ps[pred]); hits.append(float(pred==idx))
    ece=0.0
    for lo in [i/10 for i in range(10)]:
        hi=lo+0.1
        bucket=[i for i,c in enumerate(confidences) if c>=lo and (c<hi or (hi==1.0 and c<=hi))]
        if bucket:
            acc=sum(hits[i] for i in bucket)/len(bucket)
            conf=sum(confidences[i] for i in bucket)/len(bucket)
            ece += len(bucket)/len(confidences)*abs(acc-conf)
    n=len(rows)
    return correct/n,ll/n,brier/n,ece


def calibration():
    init_db(); now=datetime.now(timezone.utc)
    with sqlite3.connect(DB) as con:
        for horizon in (5,10):
            actual_col=f'actual_direction_{horizon}m'
            rows=con.execute(f'SELECT p_up_{horizon}m,p_down_{horizon}m,p_flat_{horizon}m,{actual_col} FROM predictions WHERE {actual_col} IS NOT NULL').fetchall()
            if not rows:
                print(horizon,'m: no settled predictions')
                continue
            acc,ll,brier,ece=multiclass_metrics(rows,horizon)
            con.execute('INSERT INTO model_metrics(evaluated_at_utc,horizon,model_version,n,accuracy,logloss,brier,calibration_error) VALUES(?,?,?,?,?,?,?,?)',(now.isoformat(),f'{horizon}m','production',len(rows),acc,ll,brier,ece))
            print(horizon,'m',len(rows),'accuracy',acc,'logloss',ll,'brier',brier,'ece',ece)

if __name__=='__main__': calibration()
