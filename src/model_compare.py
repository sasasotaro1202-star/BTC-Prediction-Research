import json, math, sqlite3
from datetime import datetime, timezone
import joblib, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss
from db import DB, init_db

HORIZONS={'5m':('actual_direction_5m','p_up_5m','p_down_5m','p_flat_5m'),'10m':('actual_direction_10m','p_up_10m','p_down_10m','p_flat_10m')}
FEATURES=['ret_1m','ret_3m','ret_5m','ret_10m','volatility_10m','volume_ratio']; CLASSES=['DOWN','FLAT','UP']
# Data-first policy: do not research/adopt between these checkpoints.
MILESTONES=(2000,5000,10000)
MIN_TRAIN=1000; MIN_OOS=500; TEST_BLOCK=25; MODEL_DIR=DB.parent/'models'

def now(): return datetime.now(timezone.utc).isoformat()
def safe_json(t):
    try:return json.loads(t)
    except:return {}

def load_rows(h):
    ac,*_=HORIZONS[h]
    with sqlite3.connect(DB) as con:
        rows=con.execute(f'SELECT prediction_id,created_at_utc,feature_json,{ac},p_up_{h},p_down_{h},p_flat_{h},model_version FROM predictions WHERE {ac} IS NOT NULL ORDER BY created_at_utc').fetchall()
    out=[]
    for r in rows:
        f=safe_json(r[2])
        if not all(k in f for k in FEATURES) or r[3] not in CLASSES: continue
        x=[float(f[k]) for k in FEATURES]
        if all(math.isfinite(v) for v in x): out.append({'id':r[0],'created':r[1],'x':x,'y':r[3],'production':[float(r[4]),float(r[5]),float(r[6])],'model_version':r[7]})
    return out

def metrics(ys,probs):
    idx={c:i for i,c in enumerate(CLASSES)}; y=np.array([idx[v] for v in ys]); p=np.clip(np.asarray(probs,float),1e-6,1-1e-6); p/=p.sum(axis=1,keepdims=True); pred=p.argmax(1)
    conf=p.max(1); hit=(pred==y).astype(float); ece=0.0
    for i in range(10):
        lo=i/10; hi=(i+1)/10; m=(conf>=lo)&((conf<hi) if hi<1 else (conf<=hi))
        if m.any(): ece += float(m.mean())*abs(float(hit[m].mean())-float(conf[m].mean()))
    one=np.eye(3)[y]
    return {'accuracy':float((pred==y).mean()),'logloss':float(log_loss(y,p,labels=[0,1,2])),'brier':float(np.mean(np.sum((p-one)**2,axis=1))),'calibration_error':ece}

def aligned(model,X):
    p=model.predict_proba(X); out=np.full((len(X),3),1e-6)
    for j,c in enumerate(model.classes_): out[:,CLASSES.index(c)]=p[:,j]
    return out/out.sum(1,keepdims=True)

def walk_forward(rows,factory):
    if len(rows)<MIN_TRAIN+MIN_OOS:return None
    preds=[]; ys=[]
    for end in range(MIN_TRAIN,len(rows),TEST_BLOCK):
        train=rows[:end]; test=rows[end:min(end+TEST_BLOCK,len(rows))]
        if not test: break
        model=factory(); X=np.array([r['x'] for r in train]); y=np.array([r['y'] for r in train])
        if len(set(y))<3: continue
        model.fit(X,y); preds.extend(aligned(model,np.array([r['x'] for r in test])).tolist()); ys.extend(r['y'] for r in test)
    return metrics(ys,preds) if len(ys)>=MIN_OOS else None

def train_save(rows,h,name,factory):
    X=np.array([r['x'] for r in rows]); y=np.array([r['y'] for r in rows]); model=factory()
    if len(set(y))<3:return None
    model.fit(X,y); MODEL_DIR.mkdir(parents=True,exist_ok=True); joblib.dump(model,MODEL_DIR/f'{h}.joblib')
    meta={'model_version':name,'horizon':h,'classes':list(model.classes_),'features':FEATURES,'artifact':f'{h}.joblib'}; (MODEL_DIR/f'{h}.json').write_text(json.dumps(meta,indent=2),encoding='utf-8'); return meta

def better(c,p): return c['accuracy']>=p['accuracy']-0.01 and c['logloss']<=p['logloss']-0.005 and c['brier']<=p['brier']-0.002 and c['calibration_error']<=p['calibration_error']+0.01

def save_metric(h,v,n,m,milestone):
    with sqlite3.connect(DB) as con:
        con.execute('INSERT INTO model_metrics(evaluated_at_utc,horizon,model_version,n,accuracy,logloss,brier,calibration_error) VALUES(?,?,?,?,?,?,?,?)',(now(),h,f'{v}@{milestone}',n,m['accuracy'],m['logloss'],m['brier'],m['calibration_error']))

def ensure_checkpoint_table():
    with sqlite3.connect(DB) as con:
        con.execute('CREATE TABLE IF NOT EXISTS research_checkpoints (horizon TEXT NOT NULL, milestone INTEGER NOT NULL, evaluated_at_utc TEXT NOT NULL, status TEXT NOT NULL, PRIMARY KEY(horizon,milestone))')

def checkpoint_done(h,milestone):
    with sqlite3.connect(DB) as con:return con.execute('SELECT 1 FROM research_checkpoints WHERE horizon=? AND milestone=?',(h,milestone)).fetchone() is not None

def mark_checkpoint(h,milestone,status):
    with sqlite3.connect(DB) as con:
        con.execute('INSERT OR REPLACE INTO research_checkpoints(horizon,milestone,evaluated_at_utc,status) VALUES(?,?,?,?)',(h,milestone,now(),status))

def latest_milestone(n):
    reached=[m for m in MILESTONES if n>=m]
    return max(reached) if reached else None

def prod_ver(h):
    with sqlite3.connect(DB) as con:r=con.execute('SELECT production_version FROM model_registry WHERE horizon=?',(h,)).fetchone()
    return r[0] if r else 'v1.0'

def set_prod(h,v):
    with sqlite3.connect(DB) as con: con.execute('INSERT INTO model_registry(horizon,production_version,updated_at_utc) VALUES(?,?,?) ON CONFLICT(horizon) DO UPDATE SET production_version=excluded.production_version,updated_at_utc=excluded.updated_at_utc',(h,v,now()))

def compare_h(h):
    rows=load_rows(h); n=len(rows); milestone=latest_milestone(n)
    if milestone is None:
        return {'status':'collecting','n':n,'next_milestone':MILESTONES[0]}
    if checkpoint_done(h,milestone):
        next_m=[m for m in MILESTONES if m>milestone]
        return {'status':'waiting_for_next_milestone','n':n,'last_evaluated':milestone,'next_milestone':(next_m[0] if next_m else None)}
    # Evaluate exactly at the latest reached checkpoint, using only data available by then.
    rows=rows[:milestone]
    if len(rows)<MIN_TRAIN+MIN_OOS:
        mark_checkpoint(h,milestone,'insufficient_oos')
        return {'status':'insufficient_oos','n':len(rows),'milestone':milestone,'required':MIN_TRAIN+MIN_OOS}
    oos_rows=rows[MIN_TRAIN:]
    production=metrics([r['y'] for r in oos_rows],[r['production'] for r in oos_rows]); save_metric(h,prod_ver(h),len(oos_rows),production,milestone)
    cand={'logreg_c0.1':lambda:LogisticRegression(C=.1,max_iter=2000),'logreg_c1':lambda:LogisticRegression(C=1,max_iter=2000),'logreg_c10':lambda:LogisticRegression(C=10,max_iter=2000),'rf_300':lambda:RandomForestClassifier(n_estimators=300,max_depth=6,min_samples_leaf=5,random_state=42,n_jobs=-1)}
    results={}
    for name,f in cand.items():
        m=walk_forward(rows,f)
        if m: results[name]=m; save_metric(h,name,len(oos_rows),m,milestone)
    eligible=[(name,m) for name,m in results.items() if better(m,production)]
    if not eligible:
        mark_checkpoint(h,milestone,'rejected')
        return {'status':'rejected','milestone':milestone,'production':production,'candidates':results,'n':len(rows)}
    winner,wmin=min(eligible,key=lambda z:(z[1]['logloss'],z[1]['brier'])); meta=train_save(rows,h,winner,cand[winner])
    if not meta:
        mark_checkpoint(h,milestone,'rejected_training')
        return {'status':'rejected_training','milestone':milestone,'winner':winner}
    version=f'v2.m{milestone}.{datetime.now(timezone.utc).strftime("%Y%m%d%H%M")}'; meta['model_version']=version; meta['evaluation_milestone']=milestone; (MODEL_DIR/f'{h}.json').write_text(json.dumps(meta,indent=2),encoding='utf-8'); set_prod(h,version); mark_checkpoint(h,milestone,'adopted')
    return {'status':'adopted','milestone':milestone,'version':version,'source':winner,'old':production,'new':wmin,'n':len(rows)}

def compare():
    init_db(); ensure_checkpoint_table(); print(json.dumps({h:compare_h(h) for h in HORIZONS},indent=2))
if __name__=='__main__': compare()
