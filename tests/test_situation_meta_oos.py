import numpy as np
from datetime import datetime, timezone, timedelta
from src.situation_meta_oos import build_meta_vector, causal_train_rows, meta_feature_names, primary_live_scenario

def _row(created,target):
    return {"created":created,"target":target,"production":[0.2,0.3,0.5],"microstructure":{"book_imbalance":0.2,"taker_imbalance_5m":0.1},"situation":{"normalized_entropy":0.7,"probability_margin":0.2,"trend_state":"TREND_UP","volatility_state":"EXPANDING","orderflow_state":"BUY_PRESSURE","signal_quality":"HIGH","horizon_alignment":"AGREE"}}

def test_meta_vector_contract_is_finite_and_stable():
    v=build_meta_vector(_row("2026-09-25T00:00:00+00:00","2026-09-25T00:05:00+00:00"))
    assert v.ndim==1 and len(v)==len(meta_feature_names()) and np.isfinite(v).all() and v[0:3].tolist()==[0.2,0.3,0.5]

def test_causal_train_rows_applies_target_embargo():
    s=datetime(2026,9,25,tzinfo=timezone.utc)
    rows=[_row((s-timedelta(minutes=80)).isoformat(),(s-timedelta(minutes=65)).isoformat()),_row((s-timedelta(minutes=70)).isoformat(),(s-timedelta(minutes=30)).isoformat()),_row((s-timedelta(minutes=70)).isoformat(),(s+timedelta(minutes=5)).isoformat())]
    assert len(causal_train_rows(rows,s.isoformat(),"5m"))==1

def test_causal_train_rows_rejects_created_at_or_after_target():
    s=datetime(2026,9,25,tzinfo=timezone.utc)
    assert causal_train_rows([_row((s+timedelta(minutes=5)).isoformat(),s.isoformat())],s.isoformat(),"5m")==[]

def test_primary_live_scenario_is_fail_closed_for_fallback_modes():
    assert primary_live_scenario({"production_mode":"binance_primary"}) is True
    assert primary_live_scenario({"production_mode":"bybit_fallback"}) is False
    assert primary_live_scenario({}) is False

def test_missingness_is_explicit():
    row=_row("2026-09-25T00:00:00+00:00","2026-09-25T00:05:00+00:00"); row["microstructure"]={}
    v=build_meta_vector(row); names=meta_feature_names()
    assert v[names.index("missing=book_imbalance")]==1.0
