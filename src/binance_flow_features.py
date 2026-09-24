"""Strict PIT features from immutable Binance 5-second flow bins."""
from __future__ import annotations
import math
from typing import Any, Iterable

WINDOWS=(15_000,30_000,60_000,300_000)

def _eligible(rows:Iterable[dict[str,Any]], cutoff_ms:int)->list[dict[str,Any]]:
    out=[]
    for row in rows:
        try:
            end=int(row["end_time_ms"])
            available=int(row["available_at_ms"])
            event=int(row["last_event_time_ms"])
        except (KeyError,TypeError,ValueError):
            continue
        if end<=cutoff_ms and available<=cutoff_ms and event<=cutoff_ms:
            out.append(row)
    return sorted(out,key=lambda x:int(x["end_time_ms"]))

def derive_flow_features(rows:Iterable[dict[str,Any]], prediction_cutoff_ms:int)->dict[str,float]:
    all_rows=_eligible(rows,int(prediction_cutoff_ms))
    out={}
    for width in WINDOWS:
        suffix=[r for r in all_rows if int(prediction_cutoff_ms-width)<int(r["end_time_ms"])<=int(prediction_cutoff_ms)]
        prefix=f"flow_{width//1000}s"
        if not suffix:
            for key in ("signed_qty","signed_notional","buy_share","trade_count","avg_trade_notional","max_trade_notional","liquidation_count","liquidation_signed_notional","liquidation_notional","liquidation_to_trade_notional"):
                out[f"{prefix}_{key}"]=0.0
            out[f"{prefix}_missing"]=1.0
            continue
        buy_qty=sum(float(r.get("buy_qty",0)) for r in suffix)
        sell_qty=sum(float(r.get("sell_qty",0)) for r in suffix)
        buy_notional=sum(float(r.get("buy_notional",0)) for r in suffix)
        sell_notional=sum(float(r.get("sell_notional",0)) for r in suffix)
        trade_count=sum(max(0,int(r.get("trade_count",0))) for r in suffix)
        max_trade=max((float(r.get("max_trade_notional",0)) for r in suffix),default=0.0)
        liq_buy=sum(float(r.get("liquidation_buy_notional",0)) for r in suffix)
        liq_sell=sum(float(r.get("liquidation_sell_notional",0)) for r in suffix)
        liq_count=sum(max(0,int(r.get("liquidation_count",0))) for r in suffix)
        total_notional=buy_notional+sell_notional
        liq_notional=liq_buy+liq_sell
        out[f"{prefix}_signed_qty"]=buy_qty-sell_qty
        out[f"{prefix}_signed_notional"]=buy_notional-sell_notional
        out[f"{prefix}_buy_share"]=buy_notional/total_notional if total_notional>0 else 0.5
        out[f"{prefix}_trade_count"]=float(trade_count)
        out[f"{prefix}_avg_trade_notional"]=total_notional/trade_count if trade_count else 0.0
        out[f"{prefix}_max_trade_notional"]=max_trade
        out[f"{prefix}_liquidation_count"]=float(liq_count)
        out[f"{prefix}_liquidation_signed_notional"]=liq_buy-liq_sell
        out[f"{prefix}_liquidation_notional"]=liq_notional
        out[f"{prefix}_liquidation_to_trade_notional"]=liq_notional/total_notional if total_notional>0 else 0.0
        out[f"{prefix}_missing"]=0.0
    out["flow_30s_vs_5m_signed_notional"]=out["flow_30s_signed_notional"]-0.10*out["flow_300s_signed_notional"]
    out["flow_60s_liquidation_shock"]=out["flow_60s_liquidation_to_trade_notional"]*math.log1p(out["flow_60s_liquidation_notional"])
    out["flow_any_available"]=1.0 if all_rows else 0.0
    return out
