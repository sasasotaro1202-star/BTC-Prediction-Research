import json, math, os, sqlite3, shutil, tempfile
from datetime import datetime, timezone
import joblib, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss
try:
    from db import DB, init_db
    from feature_schema import FEATURES
    from ensemble_model import SoftVotingEnsemble
except ModuleNotFoundError:
    from src.db import DB, init_db
    from src.feature_schema import FEATURES
    from src.ensemble_model import SoftVotingEnsemble
try:
    from lightgbm import LGBMClassifier
except ImportError:
    LGBMClassifier = None

HORIZONS={'5m':('actual_direction_5m','p_up_5m','p_down_5m','p_flat_5m'),'10m':('actual_direction_10m','p_up_10m','p_down_10m','p_flat_10m')}
CLASSES=['DOWN','FLAT','UP']; MILESTONES=(2000,5000,10000,12000,14000,16000,18000,20000,24000,30000,40000,50000); MIN_TRAIN=1000; MIN_OOS=500; TEST_BLOCK=25; MODEL_DIR=DB.parent/'models'; ALPHA=0.05
# Five-minute and ten-minute labels resolve into the future. Keep a conservative
# one-hour embargo before each test block in addition to the target-overlap purge.
PURGE_BARS={'5m':5,'10m':10}; EMBARGO_BARS={'5m':60,'10m':60}


def now(): return datetime.now(timezone.utc).isoformat()
def safe_json(t):
    try:return json.loads(t)
    except:return {}

def _parse_utc(value):
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None


def prediction_precedes_target(created_at_utc, target_at_utc):
    """Return True only when a prediction was recorded strictly before its target.

    Rows created at or after their target time are not valid point-in-time
    research observations and must be excluded fail-closed from OOS evaluation.
    """
    created = _parse_utc(created_at_utc)
    target = _parse_utc(target_at_utc)
    return bool(created is not None and target is not None and created < target)


def prediction_event_key(row):
    """Stable immutable identity for one persisted prediction event.

    Settlement fields and provenance metadata that can change during recovery
    are intentionally excluded. Model version, target timing, features, and
    emitted probabilities define the prediction event itself.
    """
    payload = {
        "created": str(row.get("created", "")),
        "target": str(row.get("target", "")),
        "model_version": str(row.get("model_version", "")),
        "x": [float(v) for v in row.get("x", [])],
        "production": [float(v) for v in row.get("production", [])],
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


def dedupe_exact_prediction_events(rows):
    """Keep one row per immutable prediction event without mutating the DB."""
    seen = set()
    out = []
    for row in rows:
        try:
            key = prediction_event_key(row)
        except (TypeError, ValueError):
            continue
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def _valid_timestamp(value):
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None


def _strict_pit_provenance_ok(scenario, created_at_utc):
    """Validate provenance needed for strict candidate OOS/promotion."""
    if not isinstance(scenario, dict):
        return False
    created = _valid_timestamp(created_at_utc)
    provenance = scenario.get("provenance")
    if created is None or not isinstance(provenance, dict):
        return False
    provenance = scenario.get("provenance")
    fallback_cutoff = provenance.get("prediction_cutoff") if isinstance(provenance, dict) else None
    decision = _valid_timestamp(scenario.get("decision_time_utc") or fallback_cutoff or created_at_utc)
    if decision is None or abs((decision - created).total_seconds()) > 60:
        return False

    required_top = ("available_at", "retrieved_at", "prediction_cutoff")
    parsed_top = {}
    for key in required_top:
        value = _valid_timestamp(provenance.get(key))
        if value is None:
            return False
        parsed_top[key] = value
    if not (parsed_top["available_at"] <= parsed_top["retrieved_at"] <= parsed_top["prediction_cutoff"] <= decision):
        return False

    sources = provenance.get("sources")
    if not isinstance(sources, dict) or not sources:
        return False

    def source_ok(name):
        info = sources.get(name)
        if not isinstance(info, dict):
            return False
        status = str(info.get("status", ""))
        if status not in {"ok", "ok_current_only"}:
            return False
        available = _valid_timestamp(info.get("available_at"))
        retrieved = _valid_timestamp(info.get("retrieved_at"))
        cutoff = _valid_timestamp(info.get("prediction_cutoff"))
        if available is None or retrieved is None or cutoff is None:
            return False
        if not (available <= retrieved <= cutoff <= decision):
            return False
        event_time = _valid_timestamp(info.get("event_time"))
        publication_time = _valid_timestamp(info.get("publication_time"))
        revision_time = _valid_timestamp(info.get("revision_time"))
        if event_time is not None and event_time > available:
            return False
        if publication_time is not None and publication_time > available:
            return False
        if revision_time is not None and publication_time is not None and revision_time < publication_time:
            return False
        return True

    mode = str(scenario.get("production_mode", ""))
    if mode == "binance_primary":
        required = ("binance_futures", "binance_depth", "binance_taker", "binance_premium")
    elif mode == "bybit_fallback":
        required = ("bybit_futures",)
    elif mode == "coinbase_fallback":
        required = ("coinbase_futures",)
    else:
        # Unknown mode is fail-closed for strict candidate OOS.
        return False

    return all(source_ok(name) for name in required)


def load_rows(h, strict_pit=False):
    ac,*_=HORIZONS[h]
    target_col = "target_5m" if h == "5m" else "target_10m"
    with sqlite3.connect(DB) as con:
        rows=con.execute(
            f"""SELECT prediction_id,created_at_utc,{target_col},feature_json,
                       {ac},p_up_{h},p_down_{h},p_flat_{h},model_version,scenario_json
                FROM predictions
                WHERE {ac} IS NOT NULL
                ORDER BY created_at_utc,prediction_id"""
        ).fetchall()
    out=[]
    for r in rows:
        # A prediction recorded at/after its target is never valid OOS data.
        if not prediction_precedes_target(r[1], r[2]):
            continue
        scenario = safe_json(r[9])
        if strict_pit and not _strict_pit_provenance_ok(scenario, r[1]):
            continue
        f=safe_json(r[3])
        if not all(k in f for k in FEATURES) or r[4] not in CLASSES: continue
        x=[float(f[k]) for k in FEATURES]
        if all(math.isfinite(v) for v in x):
            # DB storage is UP,DOWN,FLAT; research class order is DOWN,FLAT,UP.
            production=[float(r[6]),float(r[7]),float(r[5])]
            if all(math.isfinite(v) and v>=0 for v in production) and sum(production)>0:
                out.append({'id':r[0],'created':r[1],'target':r[2],'x':x,'y':r[4],'production':production,'model_version':r[8],'production_mode':str(scenario.get('production_mode',''))})
    return dedupe_exact_prediction_events(out)


def load_strict_rows(h):
    """Load strict PIT rows for generic research diagnostics.

    Fallback venue observations remain valid as separate research observations.
    Candidate-versus-Champion production comparison uses the narrower primary
    Binance cohort below.
    """
    return load_rows(h, strict_pit=True)

def load_archive_research_rows(h, max_rows=12000):
    """Build contiguous, historical-only BTC research rows from Binance Vision.
    
    This fallback is research-only. It never creates live-primary PIT evidence
    and therefore cannot satisfy Promotion Gate requirements.
    """
    steps = int(str(h).rstrip("m"))
    try:
        from binance_history import binance_archive_rows
        from bootstrap_train import make_features
        from label_policy import direction_from_return
        raw = binance_archive_rows(max(int(max_rows) + 40, 12000))
    except Exception:
        return []
    out = []
    for i in range(30, len(raw) - steps):
        try:
            x = make_features(raw[: i + 1])
            arr = np.asarray(x, dtype=float)
            if not np.isfinite(arr).all():
                continue
            future_return = float(raw[i + steps][4]) / float(raw[i][4]) - 1.0
            created = _parse_utc(datetime.fromtimestamp(int(raw[i][0]) / 1000.0, timezone.utc).isoformat())
            target = datetime.fromtimestamp(int(raw[i + steps][0]) / 1000.0, timezone.utc)
            out.append({
                "id": f"archive:{int(raw[i][0])}:{h}",
                "created": created.isoformat() if created else "",
                "target": target.isoformat(),
                "x": [float(v) for v in arr],
                "y": direction_from_return(future_return),
                "production_mode": "binance_vision_archive",
                "data_source": "binance_vision_closed_archive",
            })
        except (IndexError, ValueError, TypeError, FloatingPointError, OverflowError):
            continue
    return out[-int(max_rows):]

def load_primary_production_strict_rows(h):
    """Load only Binance-primary strict-PIT observations for Champion comparison."""
    return [
        row for row in load_rows(h, strict_pit=True)
        if str(row.get("production_mode", "")) == "binance_primary"
    ]

def normalize(probs):
    p=np.clip(np.asarray(probs,float),1e-6,1-1e-6); return p/p.sum(axis=1,keepdims=True)

def metrics(ys,probs):
    idx={c:i for i,c in enumerate(CLASSES)}; y=np.array([idx[v] for v in ys]); p=normalize(probs); pred=p.argmax(1); conf=p.max(1); hit=(pred==y).astype(float); ece=0.0
    for i in range(10):
        lo=i/10; hi=(i+1)/10; m=(conf>=lo)&((conf<hi) if hi<1 else (conf<=hi))
        if m.any(): ece += float(m.mean())*abs(float(hit[m].mean())-float(conf[m].mean()))
    one=np.eye(3)[y]
    return {'accuracy':float((pred==y).mean()),'logloss':float(log_loss(y,p,labels=[0,1,2])),'brier':float(np.mean(np.sum((p-one)**2,axis=1))),'calibration_error':ece}

def aligned(model,X):
    p=model.predict_proba(X); out=np.full((len(X),3),1e-6)
    for j,c in enumerate(model.classes_):
        if str(c) in CLASSES:
            out[:,CLASSES.index(str(c))]=p[:,j]
        elif isinstance(c,(int,np.integer)) and 0 <= int(c) < 3:
            out[:,int(c)]=p[:,j]
    return normalize(out)

def _temperature(probs,ys):
    if len(ys)<100 or len(set(ys))<3:return 1.0
    p=normalize(probs); y=np.array([CLASSES.index(v) for v in ys]); logits=np.log(np.clip(p,1e-6,1.0)); best_t=1.0; best=float('inf')
    for t in np.linspace(0.7,2.5,73):
        z=logits/t; z-=z.max(axis=1,keepdims=True); q=np.exp(z); q/=q.sum(axis=1,keepdims=True); loss=float(log_loss(y,q,labels=[0,1,2]))
        if loss<best:best=loss;best_t=float(t)
    return best_t

def apply_temperature(probs,t):
    if t==1.0:return normalize(probs)
    p=normalize(probs); z=np.log(p)/t; z-=z.max(axis=1,keepdims=True); q=np.exp(z); q/=q.sum(axis=1,keepdims=True); return q

def walk_forward(rows,factory,horizon):
    if len(rows)<MIN_TRAIN+MIN_OOS:return None
    purge=PURGE_BARS[horizon]; embargo=EMBARGO_BARS[horizon]; preds=[]; ys=[]; ids=[]
    for end in range(MIN_TRAIN,len(rows),TEST_BLOCK):
        train_end=max(0,end-purge-embargo); train=rows[:train_end]; test=rows[end:min(end+TEST_BLOCK,len(rows))]
        if len(train)<MIN_TRAIN or not test: break
        model=factory(); X=np.array([r['x'] for r in train]); y=np.array([r['y'] for r in train])
        if len(set(y))<3: continue
        # Nested time-ordered calibration: fit the candidate on the full training
        # window, calibrate temperature only on a later training holdout, then
        # apply that frozen temperature to the unseen test block.
        split=max(int(len(train)*0.75),MIN_TRAIN-100)
        cal_rows=train[split:]
        cal_model=factory(); cal_model.fit(np.array([r['x'] for r in train[:split]]),np.array([r['y'] for r in train[:split]]))
        cal_probs=aligned(cal_model,np.array([r['x'] for r in cal_rows])); t=_temperature(cal_probs,[r['y'] for r in cal_rows])
        model.fit(X,y); pp=aligned(model,np.array([r['x'] for r in test])); pp=apply_temperature(pp,t)
        preds.extend(pp.tolist()); ys.extend(r['y'] for r in test); ids.extend(r['id'] for r in test)
    if len(ys)<MIN_OOS:return None
    return {'metrics':metrics(ys,preds),'ys':ys,'probs':preds,'ids':ids}

def train_candidate(rows,h,name,factory,milestone):
    X=np.array([r['x'] for r in rows]); y=np.array([r['y'] for r in rows]); model=factory()
    if len(set(y))<3:return None
    model.fit(X,y); MODEL_DIR.mkdir(parents=True,exist_ok=True)
    candidate_path=MODEL_DIR/f'{h}.candidate.m{milestone}.{name}.joblib'; joblib.dump(model,candidate_path)
    meta={'model_version':name,'horizon':h,'classes':list(model.classes_),'features':FEATURES,'artifact':candidate_path.name,'candidate':True,'milestone':milestone}
    (MODEL_DIR/f'{h}.candidate.m{milestone}.{name}.json').write_text(json.dumps(meta,indent=2),encoding='utf-8'); return meta

def _validate_candidate(candidate,meta):
    if not candidate.exists() or candidate.stat().st_size == 0:return False
    try:
        model=joblib.load(candidate)
        classes=list(getattr(model,'classes_',[]))
        if classes != list(meta.get('classes',[])) or classes != CLASSES:return False
        if list(meta.get('features',[])) != FEATURES:return False
        probe=np.zeros((1,len(FEATURES)),dtype=float)
        probs=np.asarray(model.predict_proba(probe),dtype=float)
        if probs.shape != (1,len(CLASSES)) or not np.isfinite(probs).all():return False
        if not np.isclose(float(probs.sum()),1.0,atol=1e-6):return False
        return True
    except Exception:
        return False

def adopt_candidate(h,meta,version):
    candidate=MODEL_DIR/meta['artifact']; production=MODEL_DIR/f'{h}.joblib'; production_meta=MODEL_DIR/f'{h}.json'
    if not _validate_candidate(candidate,meta): return False
    MODEL_DIR.mkdir(parents=True,exist_ok=True)
    final={'model_version':version,'horizon':h,'classes':meta['classes'],'features':FEATURES,'artifact':production.name,'candidate':False,'evaluation_milestone':meta['milestone']}
    # Stage both artifacts, then replace as a pair. If metadata publication fails
    # after the model swap, restore the previous production pair instead of
    # leaving runtime with a model/metadata mismatch.
    model_tmp=None; meta_tmp=None; backup_model=None; backup_meta=None
    try:
        with tempfile.NamedTemporaryFile(dir=MODEL_DIR,prefix=f'.{h}.model.',suffix='.tmp',delete=False) as f:
            model_tmp=f.name
        with tempfile.NamedTemporaryFile(dir=MODEL_DIR,prefix=f'.{h}.meta.',suffix='.tmp',mode='w',encoding='utf-8',delete=False) as f:
            meta_tmp=f.name; f.write(json.dumps(final,indent=2)); f.flush(); os.fsync(f.fileno())
        shutil.copyfile(candidate,model_tmp)
        with open(model_tmp,'rb') as f: os.fsync(f.fileno())
        if production.exists():
            backup_model=production.with_suffix(production.suffix+'.bak')
            shutil.copyfile(production,backup_model)
        if production_meta.exists():
            backup_meta=production_meta.with_suffix(production_meta.suffix+'.bak')
            shutil.copyfile(production_meta,backup_meta)
        os.replace(model_tmp,production); model_tmp=None
        os.replace(meta_tmp,production_meta); meta_tmp=None
        if backup_model: backup_model.unlink(missing_ok=True)
        if backup_meta: backup_meta.unlink(missing_ok=True)
        return _validate_candidate(production,final) and json.loads(production_meta.read_text(encoding='utf-8')).get('model_version')==version
    except Exception:
        if backup_model and backup_model.exists(): os.replace(backup_model,production)
        elif production.exists() and not backup_model: production.unlink(missing_ok=True)
        if backup_meta and backup_meta.exists(): os.replace(backup_meta,production_meta)
        elif production_meta.exists() and not backup_meta: production_meta.unlink(missing_ok=True)
        return False
    finally:
        if model_tmp:
            try: os.unlink(model_tmp)
            except FileNotFoundError: pass
        if meta_tmp:
            try: os.unlink(meta_tmp)
            except FileNotFoundError: pass

def better(c,p,stability=None):
    # Aggregate gains are necessary but not sufficient: require improvement
    # across a majority of chronological test blocks.
    stable = stability is None or (
        stability.get('blocks',0) >= 8 and
        stability.get('improved_logloss_ratio',0.0) >= 0.55 and
        stability.get('improved_brier_ratio',0.0) >= 0.55
    )
    return stable and c['accuracy']>=p['accuracy']-0.01 and c['logloss']<=p['logloss']-0.005 and c['brier']<=p['brier']-0.002 and c['calibration_error']<=p['calibration_error']+0.01

def loss_arrays(ys,prod,cand):
    idx={c:i for i,c in enumerate(CLASSES)}; y=np.array([idx[v] for v in ys]); one=np.eye(3)[y]; pp=normalize(prod); cp=normalize(cand)
    prod_ll=-np.log(np.clip(pp[np.arange(len(y)),y],1e-12,1)); cand_ll=-np.log(np.clip(cp[np.arange(len(y)),y],1e-12,1))
    prod_br=np.sum((pp-one)**2,axis=1); cand_br=np.sum((cp-one)**2,axis=1)
    return {'logloss':cand_ll-prod_ll,'brier':cand_br-prod_br}

def hac_test(diff,lag,alpha=ALPHA):
    d=np.asarray(diff,float); n=len(d); mean=float(d.mean())
    if n<30:return {'mean_diff':mean,'stat':None,'p_value':None,'significant':False,'lag':lag}
    centered=d-mean; lrv=float(np.mean(centered*centered))
    for k in range(1,min(lag,n-1)+1):
        gamma=float(np.mean(centered[k:]*centered[:-k])); lrv += 2.0*(1-k/(lag+1))*gamma
    if not math.isfinite(lrv) or lrv<=0:return {'mean_diff':mean,'stat':None,'p_value':None,'significant':False,'lag':lag}
    stat=mean/math.sqrt(lrv/n); p=0.5*math.erfc(-stat/math.sqrt(2.0))
    return {'mean_diff':mean,'stat':float(stat),'p_value':float(p),'significant':bool(p<alpha and mean<0),'lag':lag}

def statistical_tests(ys,production,candidate,h,alpha=ALPHA):
    lag=PURGE_BARS[h]
    diffs=loss_arrays(ys,production,candidate)
    tests={k:hac_test(v,lag,alpha=alpha) for k,v in diffs.items()}
    tests['alpha']=float(alpha)
    tests['both_significant']=bool(tests['logloss']['significant'] and tests['brier']['significant'])
    return tests

def save_metric(h,v,n,m,milestone):
    with sqlite3.connect(DB) as con: con.execute('INSERT INTO model_metrics(evaluated_at_utc,horizon,model_version,n,accuracy,logloss,brier,calibration_error) VALUES(?,?,?,?,?,?,?,?)',(now(),h,f'{v}@{milestone}',n,m['accuracy'],m['logloss'],m['brier'],m['calibration_error']))

def _adjusted_alpha(alpha, tests_count):
    """Bonferroni family-wise alpha, with a stable lower bound on test count."""
    return float(alpha) / max(1, int(tests_count))

def save_stat_test(h,model_version,milestone,n,tests):
    with sqlite3.connect(DB) as con:
        con.execute('''CREATE TABLE IF NOT EXISTS model_stat_tests (id INTEGER PRIMARY KEY AUTOINCREMENT,evaluated_at_utc TEXT NOT NULL,horizon TEXT NOT NULL,milestone INTEGER NOT NULL,model_version TEXT NOT NULL,n INTEGER NOT NULL,logloss_mean_diff REAL,logloss_stat REAL,logloss_p REAL,brier_mean_diff REAL,brier_stat REAL,brier_p REAL,alpha REAL NOT NULL,both_significant INTEGER NOT NULL)''')
        ll=tests['logloss']; br=tests['brier']
        con.execute('INSERT INTO model_stat_tests(evaluated_at_utc,horizon,milestone,model_version,n,logloss_mean_diff,logloss_stat,logloss_p,brier_mean_diff,brier_stat,brier_p,alpha,both_significant) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)',(now(),h,milestone,model_version,n,ll['mean_diff'],ll['stat'],ll['p_value'],br['mean_diff'],br['stat'],br['p_value'],float(tests.get('alpha',ALPHA)),int(tests['both_significant'])))

def ensure_checkpoint_table():
    with sqlite3.connect(DB) as con: con.execute('CREATE TABLE IF NOT EXISTS research_checkpoints (horizon TEXT NOT NULL, milestone INTEGER NOT NULL, evaluated_at_utc TEXT NOT NULL, status TEXT NOT NULL, PRIMARY KEY(horizon,milestone))')

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
    rows=load_primary_production_strict_rows(h); n=len(rows); milestone=next_due_milestone(n,h)
    if milestone is None:
        future=next((m for m in MILESTONES if not checkpoint_done(h,m)),None); return {'status':'collecting','n':n,'next_milestone':future}
    rows=rows[:milestone]
    if len(rows)<MIN_TRAIN+MIN_OOS:
        # Do not checkpoint an unmet milestone: the dataset is still growing and
        # this exact milestone must be evaluated once enough settled OOS exists.
        return {'status':'insufficient_oos','n':len(rows),'milestone':milestone,'required':MIN_TRAIN+MIN_OOS}
    # Protect the final 20% of the available milestone data from candidate selection.
    # It is evaluated only as a descriptive production audit.
    split=max(MIN_TRAIN, int(len(rows)*0.80))
    development_rows=rows[:split]
    final_holdout_rows=rows[split:]
    oos_rows=development_rows[MIN_TRAIN:]; ys=[r['y'] for r in oos_rows]; production_probs=[r['production'] for r in oos_rows]
    production=metrics(ys,production_probs); save_metric(h,prod_ver(h),len(oos_rows),production,milestone)
    prod_by_id={r['id']:r for r in oos_rows}
    cand={
      'logreg_c0.1':lambda:Pipeline([('scale',StandardScaler()),('model',LogisticRegression(C=.1,max_iter=3000))]),
      'logreg_c1':lambda:Pipeline([('scale',StandardScaler()),('model',LogisticRegression(C=1,max_iter=3000))]),
      'logreg_c10':lambda:Pipeline([('scale',StandardScaler()),('model',LogisticRegression(C=10,max_iter=3000))]),
      'rf_500':lambda:RandomForestClassifier(n_estimators=500,max_depth=7,min_samples_leaf=8,max_features='sqrt',random_state=42,n_jobs=-1),
      # Historical BTC research repeatedly found randomized tree ensembles competitive;
      # test ExtraTrees under the same chronological/purged/calibrated gate rather than
      # assuming the research-only result transfers to production.
      'extra_trees_500':lambda:ExtraTreesClassifier(n_estimators=500,max_depth=7,min_samples_leaf=8,max_features='sqrt',random_state=42,n_jobs=-1),
      # LightGBM is a free CPU tabular challenger. It enters only the same
      # chronological/purged/calibrated/HAC gate as every other candidate.
      **({'lightgbm':lambda:LGBMClassifier(
          objective='multiclass',num_class=3,n_estimators=350,learning_rate=0.03,
          num_leaves=31,min_child_samples=30,subsample=0.9,colsample_bytree=0.9,
          reg_lambda=1.0,random_state=42,n_jobs=-1,verbosity=-1
      )} if LGBMClassifier is not None else {}),
      'hgb':lambda:HistGradientBoostingClassifier(max_iter=250,max_leaf_nodes=15,learning_rate=.04,l2_regularization=1.0,random_state=42),
      # A compact heterogeneous soft-vote candidate tests whether probability
      # averaging improves generalization over any single estimator. It remains
      # a candidate until the same chronological/HAC gate proves a real gain.
      'soft_ensemble':lambda:SoftVotingEnsemble()
    }
    results={}
    # Multiple candidate families are tested at each milestone. Apply a
    # Bonferroni family-wise correction so adding candidates cannot silently
    # inflate the false-adoption rate. The corrected alpha is used only for
    # statistical promotion; descriptive metrics remain unchanged.
    corrected_alpha=_adjusted_alpha(ALPHA,len(cand))
    for name,f in cand.items():
        wf=walk_forward(development_rows,f,h)
        if not wf: continue
        aligned_rows=[prod_by_id[i] for i in wf['ids'] if i in prod_by_id]
        if len(aligned_rows)!=len(wf['ids']): continue
        prod_aligned=[r['production'] for r in aligned_rows]
        candidate_metrics=metrics(wf['ys'],wf['probs']); production_aligned_metrics=metrics(wf['ys'],prod_aligned)
        block_diffs=[]
        for start in range(0,len(wf['ys']),TEST_BLOCK):
            by=wf['ys'][start:start+TEST_BLOCK]; bp=prod_aligned[start:start+TEST_BLOCK]; bc=wf['probs'][start:start+TEST_BLOCK]
            if len(by)<max(10,TEST_BLOCK//2): continue
            pm=metrics(by,bp); cm=metrics(by,bc)
            block_diffs.append({'logloss_delta':cm['logloss']-pm['logloss'],'brier_delta':cm['brier']-pm['brier']})
        stability={
            'blocks':len(block_diffs),
            'improved_logloss_ratio':float(np.mean([d['logloss_delta']<0 for d in block_diffs])) if block_diffs else 0.0,
            'improved_brier_ratio':float(np.mean([d['brier_delta']<0 for d in block_diffs])) if block_diffs else 0.0
        }
        tests=statistical_tests(wf['ys'],prod_aligned,wf['probs'],h,alpha=corrected_alpha)
        results[name]={'metrics':candidate_metrics,'production_aligned':production_aligned_metrics,'block_stability':stability,'statistical_tests':tests,'n':len(wf['ids'])}
        save_metric(h,name,len(wf['ids']),candidate_metrics,milestone); save_stat_test(h,name,milestone,len(wf['ids']),tests)
    eligible=[]
    for name,r in results.items():
        if better(r['metrics'],r['production_aligned'],r.get('block_stability')) and r['statistical_tests']['both_significant']: eligible.append((name,r['metrics'],r['statistical_tests']))
    if not eligible:
        mark_checkpoint(h,milestone,'rejected'); return {'status':'rejected','milestone':milestone,'production':production,'candidates':results,'n':len(rows)}
    winner,wmin,wtest=min(eligible,key=lambda z:(z[1]['logloss'],z[1]['brier'])); meta=train_candidate(development_rows,h,winner,cand[winner],milestone)
    if not meta:
        mark_checkpoint(h,milestone,'rejected_training'); return {'status':'rejected_training','milestone':milestone,'winner':winner}
    version=f'candidate.v3.m{milestone}.{datetime.now(timezone.utc).strftime("%Y%m%d%H%M")}'; meta['model_version']=version; meta['statistical_significance']=wtest
    # Candidate promotion is deliberately two-phase. This research workflow may
    # train and validate artifacts, but it must never change the Champion/production
    # registry automatically. Explicit promotion requires the independent
    # promotion gate plus a separate human-approved deployment action.
    candidate_meta_path=MODEL_DIR/f'{h}.candidate.m{milestone}.{winner}.json'
    candidate_meta_path.write_text(json.dumps(meta,indent=2,sort_keys=True),encoding='utf-8')
    holdout=metrics([r['y'] for r in final_holdout_rows],[r['production'] for r in final_holdout_rows]) if final_holdout_rows else None
    mark_checkpoint(h,milestone,'eligible_pending_explicit_promotion')
    return {'status':'eligible_pending_explicit_promotion','milestone':milestone,'version':version,'source':winner,'old':production,'new':wmin,'statistical_tests':wtest,'final_holdout_production':holdout,'n':len(rows),'production_changed':False}

def compare():
    init_db(); ensure_checkpoint_table(); print(json.dumps({h:compare_h(h) for h in HORIZONS},indent=2))

if __name__ == '__main__': compare()
