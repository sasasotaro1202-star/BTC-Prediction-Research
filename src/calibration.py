import json, math, sqlite3
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from db import DB, init_db

CLASSES = ['UP','DOWN','FLAT']
MODEL_DIR = Path(DB).parent / 'models'


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


def temperature_scale(rows):
    if len(rows) < 200:
        return 1.0, None
    y=[]; logits=[]
    for r in rows:
        ps=np.clip(np.asarray([float(r[0]),float(r[1]),float(r[2])]),1e-6,1-1e-6)
        ps=ps/ps.sum(); logits.append(np.log(ps)); y.append(CLASSES.index(r[3]))
    logits=np.asarray(logits); y=np.asarray(y)
    best_t=1.0; best=float('inf')
    for t in np.linspace(0.7,2.5,73):
        z=logits/t; z=z-z.max(axis=1,keepdims=True); p=np.exp(z); p/=p.sum(axis=1,keepdims=True)
        loss=float(-np.mean(np.log(np.clip(p[np.arange(len(y)),y],1e-12,1.0))))
        if loss < best:
            best=loss; best_t=float(t)
    return best_t,best


def save_temperature(horizon, temperature, n):
    MODEL_DIR.mkdir(parents=True,exist_ok=True)
    path=MODEL_DIR/f'{horizon}.calibration.json'
    payload={'horizon':horizon,'temperature':float(temperature),'n_settled':int(n),'method':'bounded_temperature_scaling','updated_at_utc':datetime.now(timezone.utc).isoformat()}
    path.write_text(json.dumps(payload,indent=2),encoding='utf-8')


def calibration():
    init_db(); now=datetime.now(timezone.utc)
    for horizon in (5,10):
        actual_col=f'actual_direction_{horizon}m'
        with sqlite3.connect(DB) as con:
            rows=con.execute(f'SELECT p_up_{horizon}m,p_down_{horizon}m,p_flat_{horizon}m,{actual_col} FROM predictions WHERE {actual_col} IS NOT NULL ORDER BY created_at_utc').fetchall()
            if not rows:
                print(horizon,'m: no settled predictions')
                continue
            acc,ll,brier,ece=multiclass_metrics(rows,horizon)
            con.execute('INSERT INTO model_metrics(evaluated_at_utc,horizon,model_version,n,accuracy,logloss,brier,calibration_error) VALUES(?,?,?,?,?,?,?,?)',(now.isoformat(),f'{horizon}m','production',len(rows),acc,ll,brier,ece))
        temperature,fit_ll=temperature_scale(rows)
        save_temperature(f'{horizon}m',temperature,len(rows))
        print(horizon,'m',len(rows),'accuracy',acc,'logloss',ll,'brier',brier,'ece',ece,'temperature',temperature,'fit_logloss',fit_ll)

if __name__=='__main__': calibration()
