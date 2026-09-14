import json, math, sqlite3
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from db import DB, init_db

CLASSES = ['UP','DOWN','FLAT']
MODEL_DIR = Path(DB).parent / 'models'
MIN_CALIBRATION = 300
HOLDOUT_FRACTION = 0.25


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


def _probs_and_labels(rows):
    y=[]; logits=[]
    for r in rows:
        ps=np.clip(np.asarray([float(r[0]),float(r[1]),float(r[2])]),1e-6,1-1e-6)
        ps=ps/ps.sum(); logits.append(np.log(ps)); y.append(CLASSES.index(r[3]))
    return np.asarray(logits), np.asarray(y)


def _logloss_at_temperature(logits,y,t):
    z=logits/t; z=z-z.max(axis=1,keepdims=True); p=np.exp(z); p/=p.sum(axis=1,keepdims=True)
    return float(-np.mean(np.log(np.clip(p[np.arange(len(y)),y],1e-12,1.0))))


def temperature_scale(rows):
    if len(rows) < MIN_CALIBRATION:
        return 1.0, None, None
    logits,y=_probs_and_labels(rows)
    split=max(int(len(rows)*(1-HOLDOUT_FRACTION)),1)
    fit_logits,fit_y=logits[:split],y[:split]
    eval_logits,eval_y=logits[split:],y[split:]
    if len(eval_y)<100 or len(set(fit_y.tolist()))<3:
        return 1.0,None,None
    best_t=1.0; best=float('inf')
    for t in np.linspace(0.7,2.5,73):
        loss=_logloss_at_temperature(fit_logits,fit_y,float(t))
        if loss < best:
            best=loss; best_t=float(t)
    raw_eval=_logloss_at_temperature(eval_logits,eval_y,1.0)
    scaled_eval=_logloss_at_temperature(eval_logits,eval_y,best_t)
    # Only publish a temperature if it improves genuinely unseen data.
    # A tiny tolerance prevents oscillation when the difference is noise.
    if scaled_eval >= raw_eval-0.001:
        return 1.0,best,raw_eval
    return best_t,best,scaled_eval


def save_temperature(horizon, temperature, n, fit_logloss, eval_logloss, holdout_fraction, model_version):
    MODEL_DIR.mkdir(parents=True,exist_ok=True)
    path=MODEL_DIR/f'{horizon}.calibration.json'
    payload={
        'horizon':horizon,
        'temperature':float(temperature),
        'n_settled':int(n),
        'model_version':model_version,
        'method':'bounded_temperature_scaling_current_model_generation_holdout_guard',
        'fit_logloss':None if fit_logloss is None else float(fit_logloss),
        'holdout_logloss':None if eval_logloss is None else float(eval_logloss),
        'holdout_fraction':float(holdout_fraction),
        'updated_at_utc':datetime.now(timezone.utc).isoformat()
    }
    path.write_text(json.dumps(payload,indent=2),encoding='utf-8')


def _current_registry_version(con, horizon):
    row=con.execute('SELECT production_version FROM model_registry WHERE horizon=?',(horizon,)).fetchone()
    return str(row[0]) if row and row[0] else None


def _settled_rows(con, horizon, actual_col, model_version):
    # Calibration must not pool incompatible model generations.  Predictions store
    # both horizon registry versions in one field, so match the relevant prefix.
    if not model_version:
        return []
    prefix=f'{horizon}:{model_version}|%'
    return con.execute(
        f'''SELECT p_up_{horizon}m,p_down_{horizon}m,p_flat_{horizon}m,{actual_col}
            FROM predictions
            WHERE {actual_col} IS NOT NULL
              AND model_version LIKE ?
              AND model_version NOT LIKE 'DEGRADED_NO_FRESH_DATA%'
            ORDER BY created_at_utc''',
        (prefix,)
    ).fetchall()


def calibration():
    init_db(); now=datetime.now(timezone.utc)
    for horizon in (5,10):
        actual_col=f'actual_direction_{horizon}m'
        with sqlite3.connect(DB) as con:
            model_version=_current_registry_version(con,f'{horizon}m')
            rows=_settled_rows(con,horizon,actual_col,model_version)
            if not rows:
                print(horizon,'m: no settled predictions for current model generation',model_version)
                continue
            acc,ll,brier,ece=multiclass_metrics(rows,horizon)
            con.execute('INSERT INTO model_metrics(evaluated_at_utc,horizon,model_version,n,accuracy,logloss,brier,calibration_error) VALUES(?,?,?,?,?,?,?,?)',(now.isoformat(),f'{horizon}m',model_version,len(rows),acc,ll,brier,ece))
        temperature,fit_ll,eval_ll=temperature_scale(rows)
        save_temperature(f'{horizon}m',temperature,len(rows),fit_ll,eval_ll,HOLDOUT_FRACTION,model_version)
        print(horizon,'m',len(rows),'model_version',model_version,'accuracy',acc,'logloss',ll,'brier',brier,'ece',ece,'temperature',temperature,'fit_logloss',fit_ll,'holdout_logloss',eval_ll)

if __name__=='__main__': calibration()
