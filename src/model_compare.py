import json, math, sqlite3, shutil
from datetime import datetime, timezone
import joblib, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss
from db import DB, init_db

HORIZONS={'5m':('actual_direction_5m','p_up_5m','p_down_5m','p_flat_5m'),'10m':('actual_direction_10m','p_up_10m','p_down_10m','p_flat_10m')}
FEATURES=['ret_1m','ret_3m','ret_5m','ret_10m','volatility_10m','volume_ratio']; CLASSES=['DOWN','FLAT','UP']
MILESTONES=(2000,5000,10000)
MIN_TRAIN=1000; MIN_OOS=500; TEST_BLOCK=25; MODEL_DIR=DB.parent/'models'
ALPHA=0.05


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


def normalize(probs):
    p=np.clip(np.asarray(probs,float),1e-6,1-1e-6)
    return p/p.sum(axis=1,keepdims=True)


def metrics(ys,probs):
    idx={c:i for i,c in enumerate(CLASSES)}; y=np.array([idx[v] for v in ys]); p=normalize(probs); pred=p.argmax(1)
    conf=p.max(1); hit=(pred==y).astype(float); ece=0.0
    for i in range(10):
        lo=i/10; hi=(i+1)/10; m=(conf>=lo)&((conf<hi) if hi<1 else (conf<=hi))
        if m.any(): ece += float(m.mean())*abs(float(hit[m].mean())-float(conf[m].mean()))
    one=np.eye(3)[y]
    return {'accuracy':float((pred==y).mean()),'logloss':float(log_loss(y,p,labels=[0,1,2])),'brier':float(np.mean(np.sum((p-one)**2,axis=1))),'calibration_error':ece}


def aligned(model,X):
    p=model.predict_proba(X); out=np.full((len(X),3),1e-6)
    for j,c in enumerate(model.classes_): out[:,CLASSES.index(c)]=p[:,j]
    return normalize(out)


def walk_forward(rows,factory):
    if len(rows)<MIN_TRAIN+MIN_OOS:return None
    preds=[]; ys=[]; ids=[]
    for end in range(MIN_TRAIN,len(rows),TEST_BLOCK):
        train=rows[:end]; test=rows[end:min(end+TEST_BLOCK,len(rows))]
        if not test: break
        model=factory(); X=np.array([r['x'] for r in train]); y=np.array([r['y'] for r in train])
        if len(set(y))<3: continue
        model.fit(X,y); pp=aligned(model,np.array([r['x'] for r in test]))
        preds.extend(pp.tolist()); ys.extend(r['y'] for r in test); ids.extend(r['id'] for r in test)
    if len(ys)<MIN_OOS:return None
    return {'metrics':metrics(ys,preds),'ys':ys,'probs':preds,'ids':ids}


def train_candidate(rows,h,name,factory,milestone):
    X=np.array([r['x'] for r in rows]); y=np.array([r['y'] for r in rows]); model=factory()
    if len(set(y))<3:return None
    model.fit(X,y)
    MODEL_DIR.mkdir(parents=True,exist_ok=True)
    candidate_path=MODEL_DIR/f'{h}.candidate.m{milestone}.{name}.joblib'
    joblib.dump(model,candidate_path)
    meta={'model_version':name,'horizon':h,'classes':list(model.classes_),'features':FEATURES,'artifact':candidate_path.name,'candidate':True,'milestone':milestone}
    (MODEL_DIR/f'{h}.candidate.m{milestone}.{name}.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    return meta


def adopt_candidate(h,meta,version):
    candidate=MODEL_DIR/meta['artifact']
    production=MODEL_DIR/f'{h}.joblib'
    if not candidate.exists(): return False
    # Production is replaced only after the complete OOS/statistical gate passes.
    shutil.copyfile(candidate,production)
    final={'model_version':version,'horizon':h,'classes':meta['classes'],'features':FEATURES,'artifact':production.name,'candidate':False,'evaluation_milestone':meta['milestone']}
    (MODEL_DIR/f'{h}.json').write_text(json.dumps(final,indent=2),encoding='utf-8')
    return True


def better(c,p):
    return c['accuracy']>=p['accuracy']-0.01 and c['logloss']<=p['logloss']-0.005 and c['brier']<=p['brier']-0.002 and c['calibration_error']<=p['calibration_error']+0.01


def loss_arrays(ys,prod,cand):
    idx={c:i for i,c in enumerate(CLASSES)}; y=np.array([idx[v] for v in ys]); one=np.eye(3)[y]
    pp=normalize(prod); cp=normalize(cand)
    prod_ll=-np.log(np.clip(pp[np.arange(len(y)),y],1e-12,1)); cand_ll=-np.log(np.clip(cp[np.arange(len(y)),y],1e-12,1))
    prod_br=np.sum((pp-one)**2,axis=1); cand_br=np.sum((cp-one)**2,axis=1)
    return {'logloss':cand_ll-prod_ll,'brier':cand_br-prod_br}


def hac_test(diff,lag):
    d=np.asarray(diff,float); n=len(d); mean=float(d.mean())
    if n<30:return {'mean_diff':mean,'stat':None,'p_value':None,'significant':False,'lag':lag}
    centered=d-mean; gamma0=float(np.mean(centered*centered)); lrv=gamma0
    for k in range(1,min(lag,n-1)+1):
        gamma=float(np.mean(centered[k:]*centered[:-k])); lrv += 2.0*(1-k/(lag+1))*gamma
    if not math.isfinite(lrv) or lrv<=0:
        return {'mean_diff':mean,'stat':None,'p_value':None,'significant':False,'lag':lag}
    stat=mean/math.sqrt(lrv/n)
    p=0.5*math.erfc(-stat/math.sqrt(2.0))
    return {'mean_diff':mean,'stat':float(stat),'p_value':float(p),'significant':bool(p<ALPHA and mean<0),'lag':lag}


def statistical_tests(ys,production,candidate,h):
    lag=1 if h=='5m' else 2
    diffs=loss_arrays(ys,production,candidate)
    tests={k:hac_test(v,lag) for k,v in diffs.items()}
    tests['both_significant']=bool(tests['logloss']['significant'] and tests['brier']['significant'])
    return tests


def save_metric(h,v,n,m,milestone):
    with sqlite3.connect(DB) as con:
        con.execute('INSERT INTO model_metrics(evaluated_at_utc,horizon,model_version,n,accuracy,logloss,brier,calibration_error) VALUES(?,?,?,?,?,?,?,?)',(now(),h,f'{v}@{milestone}',n,m['accuracy'],m['logloss'],m['brier'],m['calibration_error']))


def save_stat_test(h,model_version,milestone,n,tests):
    with sqlite3.connect(DB) as con:
        con.execute('''CREATE TABLE IF NOT EXISTS model_stat_tests (id INTEGER PRIMARY KEY AUTOINCREMENT,evaluated_at_utc TEXT NOT NULL,horizon TEXT NOT NULL,milestone INTEGER NOT NULL,model_version TEXT NOT NULL,n INTEGER NOT NULL,logloss_mean_diff REAL,logloss_stat REAL,logloss_p REAL,brier_mean_diff REAL,brier_stat REAL,brier_p REAL,alpha REAL NOT NULL,both_significant INTEGER NOT NULL)''')
        ll=tests['logloss']; br=tests['brier']
        con.execute('INSERT INTO model_stat_tests(evaluated_at_utc,horizon,milestone,model_version,n,logloss_mean_diff,logloss_stat,logloss_p,brier_mean_diff,brier_stat,brier_p,alpha,both_significant) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)',(now(),h,milestone,model_version,n,ll['mean_diff'],ll['stat'],ll['p_value'],br['mean_diff'],br['stat'],br['p_value'],ALPHA,int(tests['both_significant'])))


def ensure_checkpoint_table():
    with sqlite3.connect(DB) as con:
        con.execute('CREATE TABLE IF NOT EXISTS research_checkpoints (horizon TEXT NOT NULL, milestone INTEGER NOT NULL, evaluated_at_utc TEXT NOT NULL, status TEXT NOT NULL, PRIMARY KEY(horizon,milestone))')


def checkpoint_done(h,milestone):
    with sqlite3.connect(DB) as con:return con.execute('SELECT 1 FROM research_checkpoints WHERE horizon=? AND milestone=?',(h,milestone)).fetchone() is not None


def mark_checkpoint(h,milestone,status):
    with sqlite3.connect(DB) as con: con.execute('INSERT OR REPLACE INTO research_checkpoints(horizon,milestone,evaluated_at_utc,status) VALUES(?,?,?,?)',(h,milestone,now(),status))


def next_due_milestone(n,h):
    for m in MILESTONES:
        if n>=m and not checkpoint_done(h,m): return m
    return None


def prod_ver(h):
    with sqlite3.connect(DB) as con:r=con.execute('SELECT production_version FROM model_registry WHERE horizon=?',(h,)).fetchone()
    return r[0] if r else 'v1.0'


def set_prod(h,v):
    with sqlite3.connect(DB) as con: con.execute('INSERT INTO model_registry(horizon,production_version,updated_at_utc) VALUES(?,?,?) ON CONFLICT(horizon) DO UPDATE SET production_version=excluded.production_version,updated_at_utc=excluded.updated_at_utc',(h,v,now()))


def compare_h(h):
    rows=load_rows(h); n=len(rows); milestone=next_due_milestone(n,h)
    if milestone is None:
        future=next((m for m in MILESTONES if not checkpoint_done(h,m)),None)
        return {'status':'collecting','n':n,'next_milestone':future}
    rows=rows[:milestone]
    if len(rows)<MIN_TRAIN+MIN_OOS:
        mark_checkpoint(h,milestone,'insufficient_oos')
        return {'status':'insufficient_oos','n':len(rows),'milestone':milestone,'required':MIN_TRAIN+MIN_OOS}
    oos_rows=rows[MIN_TRAIN:]
    ys=[r['y'] for r in oos_rows]; production_probs=[r['production'] for r in oos_rows]
    production=metrics(ys,production_probs); save_metric(h,prod_ver(h),len(oos_rows),production,milestone)
    cand={'logreg_c0.1':lambda:LogisticRegression(C=.1,max_iter=2000),'logreg_c1':lambda:LogisticRegression(C=1,max_iter=2000),'logreg_c10':lambda:LogisticRegression(C=10,max_iter=2000),'rf_300':lambda:RandomForestClassifier(n_estimators=300,max_depth=6,min_samples_leaf=5,random_state=42,n_jobs=-1)}
    results={}
    for name,f in cand.items():
        wf=walk_forward(rows,f)
        if not wf: continue
        m=wf['metrics']; tests=statistical_tests(ys,production_probs,wf['probs'],h)
        results[name]={'metrics':m,'statistical_tests':tests}
        save_metric(h,name,len(oos_rows),m,milestone)
        save_stat_test(h,name,milestone,len(oos_rows),tests)
    eligible=[]
    for name,r in results.items():
        if better(r['metrics'],production) and r['statistical_tests']['both_significant']:
            eligible.append((name,r['metrics'],r['statistical_tests']))
    if not eligible:
        mark_checkpoint(h,milestone,'rejected')
        return {'status':'rejected','milestone':milestone,'production':production,'candidates':results,'n':len(rows)}
    winner,wmin,wtest=min(eligible,key=lambda z:(z[1]['logloss'],z[1]['brier']))
    meta=train_candidate(rows,h,winner,cand[winner],milestone)
    if not meta:
        mark_checkpoint(h,milestone,'rejected_training')
        return {'status':'rejected_training','milestone':milestone,'winner':winner}
    version=f'v2.m{milestone}.{datetime.now(timezone.utc).strftime("%Y%m%d%H%M")}'; meta['model_version']=version; meta['statistical_significance']=wtest
    if not adopt_candidate(h,meta,version):
        mark_checkpoint(h,milestone,'rejected_adoption')
        return {'status':'rejected_adoption','milestone':milestone,'winner':winner}
    set_prod(h,version); mark_checkpoint(h,milestone,'adopted')
    return {'status':'adopted','milestone':milestone,'version':version,'source':winner,'old':production,'new':wmin,'statistical_tests':wtest,'n':len(rows)}


def compare():
    init_db(); ensure_checkpoint_table(); print(json.dumps({h:compare_h(h) for h in HORIZONS},indent=2))

if __name__=='__main__': compare()
