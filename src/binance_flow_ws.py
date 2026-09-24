"""Research-only Binance Futures aggTrade/forceOrder collector with strict PIT bins."""
from __future__ import annotations
import argparse
import asyncio
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Awaitable, Callable

import websockets

ROOT=Path(__file__).resolve().parents[1]
DEFAULT_CACHE=ROOT/"data"/"binance_flow_5s.json"
AGGTRADE_URL="wss://fstream.binance.com/market/ws/btcusdt@aggTrade"
FORCE_ORDER_URL="wss://fstream.binance.com/market/ws/btcusdt@forceOrder"
SCHEMA_VERSION=1
BIN_MS=5000
FINALIZATION_LAG_MS=10000
MAX_BINS=720


def _data(message: Any) -> dict[str, Any]:
    if not isinstance(message,dict):
        return {}
    value=message.get("data")
    return value if isinstance(value,dict) else message


def _recv_ms(value: int|None) -> int:
    return int(value if value is not None else time.time()*1000)


def _valid_time(event_ms:int, received_ms:int) -> bool:
    return event_ms>0 and received_ms>0 and event_ms<=received_ms+5000


def parse_agg_trade_message(message:Any, received_at_ms:int|None=None)->dict[str,Any]|None:
    d=_data(message)
    if d.get("e")!="aggTrade" or str(d.get("s","")).upper()!="BTCUSDT":
        return None
    try:
        event_ms=int(d["E"])
        trade_ms=int(d["T"])
        trade_id=int(d["a"])
        price=float(d["p"])
        qty=float(d["q"])
        maker=bool(d["m"])
    except (KeyError,TypeError,ValueError):
        return None
    received=_recv_ms(received_at_ms)
    if not _valid_time(event_ms,received) or trade_ms<=0 or trade_ms>received+5000:
        return None
    if trade_ms>event_ms+5000 or not all(math.isfinite(v) for v in (price,qty)):
        return None
    if price<=0 or qty<=0 or trade_id<0:
        return None
    return {
        "kind":"agg_trade",
        "event_time_ms":event_ms,
        "trade_time_ms":trade_ms,
        "retrieved_at_ms":received,
        "trade_id":trade_id,
        "price":price,
        "quantity":qty,
        "notional":price*qty,
        "aggressor_sign":-1.0 if maker else 1.0,
    }


def parse_force_order_message(message:Any, received_at_ms:int|None=None)->dict[str,Any]|None:
    d=_data(message)
    if d.get("e")!="forceOrder":
        return None
    order=d.get("o")
    if not isinstance(order,dict) or str(order.get("s","")).upper()!="BTCUSDT":
        return None
    side=str(order.get("S","")).upper()
    if side not in {"BUY","SELL"}:
        return None
    try:
        event_ms=int(d["E"])
        trade_ms=int(order.get("T",event_ms))
        order_id=int(order["i"])
        price=float(order.get("ap") or order.get("p"))
        qty=float(order["z"] or order["q"])
    except (KeyError,TypeError,ValueError):
        return None
    received=_recv_ms(received_at_ms)
    if not _valid_time(event_ms,received) or trade_ms<=0 or trade_ms>received+5000:
        return None
    if not all(math.isfinite(v) for v in (price,qty)) or price<=0 or qty<=0 or order_id<0:
        return None
    return {
        "kind":"force_order",
        "event_time_ms":event_ms,
        "trade_time_ms":trade_ms,
        "retrieved_at_ms":received,
        "order_id":order_id,
        "price":price,
        "quantity":qty,
        "notional":price*qty,
        "aggressor_sign":1.0 if side=="BUY" else -1.0,
        "side":side,
    }


def _new_bin(start_ms:int)->dict[str,Any]:
    return {
        "start_time_ms":int(start_ms),
        "end_time_ms":int(start_ms+BIN_MS),
        "available_at_ms":None,
        "last_event_time_ms":0,
        "trade_count":0,
        "buy_qty":0.0,
        "sell_qty":0.0,
        "buy_notional":0.0,
        "sell_notional":0.0,
        "max_trade_notional":0.0,
        "liquidation_count":0,
        "liquidation_buy_notional":0.0,
        "liquidation_sell_notional":0.0,
    }


def ingest_event(bins:dict[int,dict[str,Any]], event:dict[str,Any])->bool:
    event_ms=int(event["event_time_ms"])
    start=(event_ms//BIN_MS)*BIN_MS
    current=bins.get(start)
    if current is not None and current.get("available_at_ms") is not None:
        return False
    if current is None:
        current=_new_bin(start)
        bins[start]=current
    notional=float(event.get("notional",0.0))
    qty=float(event.get("quantity",0.0))
    if not math.isfinite(notional) or notional<0 or not math.isfinite(qty) or qty<0:
        return False
    sign=1.0 if float(event.get("aggressor_sign",0.0))>=0 else -1.0
    current["last_event_time_ms"]=max(int(current["last_event_time_ms"]),event_ms)
    if event.get("kind")=="agg_trade":
        current["trade_count"]+=1
        if sign>0:
            current["buy_qty"]+=qty
            current["buy_notional"]+=notional
        else:
            current["sell_qty"]+=qty
            current["sell_notional"]+=notional
        current["max_trade_notional"]=max(float(current["max_trade_notional"]),notional)
        return True
    if event.get("kind")=="force_order":
        current["liquidation_count"]+=1
        if sign>0:
            current["liquidation_buy_notional"]+=notional
        else:
            current["liquidation_sell_notional"]+=notional
        return True
    return False


def finalize_bins(bins:dict[int,dict[str,Any]], latest_event_time_ms:int, available_at_ms:int)->int:
    threshold=int(latest_event_time_ms-FINALIZATION_LAG_MS)
    count=0
    for item in bins.values():
        if item.get("available_at_ms") is not None:
            continue
        if int(item["end_time_ms"])<=threshold:
            item["available_at_ms"]=int(available_at_ms)
            count+=1
    return count


def normalize_bins(bins:dict[int,dict[str,Any]], max_bins:int=MAX_BINS)->list[dict[str,Any]]:
    out=[]
    for start in sorted(bins):
        item=bins[start]
        if not isinstance(item,dict) or item.get("available_at_ms") is None:
            continue
        try:
            available=int(item["available_at_ms"])
            last_event=int(item["last_event_time_ms"])
        except (TypeError,ValueError):
            continue
        if available<last_event:
            continue
        row={
            "start_time_ms":int(item["start_time_ms"]),
            "end_time_ms":int(item["end_time_ms"]),
            "available_at_ms":available,
            "last_event_time_ms":last_event,
            "trade_count":max(0,int(item["trade_count"])),
            "buy_qty":float(item["buy_qty"]),
            "sell_qty":float(item["sell_qty"]),
            "buy_notional":float(item["buy_notional"]),
            "sell_notional":float(item["sell_notional"]),
            "max_trade_notional":max(0.0,float(item["max_trade_notional"])),
            "liquidation_count":max(0,int(item["liquidation_count"])),
            "liquidation_buy_notional":float(item["liquidation_buy_notional"]),
            "liquidation_sell_notional":float(item["liquidation_sell_notional"]),
        }
        if not all(math.isfinite(float(row[k])) for k in (
            "buy_qty","sell_qty","buy_notional","sell_notional","max_trade_notional",
            "liquidation_buy_notional","liquidation_sell_notional")):
            continue
        out.append(row)
    return out[-int(max_bins):]


def write_cache(bins:dict[int,dict[str,Any]], path:Path=DEFAULT_CACHE)->None:
    path.parent.mkdir(parents=True,exist_ok=True)
    payload={
        "schema_version":SCHEMA_VERSION,
        "source":"Binance USD-M Futures WebSocket",
        "streams":["btcusdt@aggTrade","btcusdt@forceOrder"],
        "bin_ms":BIN_MS,
        "finalization_lag_ms":FINALIZATION_LAG_MS,
        "rows":normalize_bins(bins),
        "updated_at_utc":__import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
    }
    temp=path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temp.write_text(json.dumps(payload,sort_keys=True,separators=(",",":"))+"\n",encoding="utf-8")
    temp.replace(path)


def load_cache(path:Path=DEFAULT_CACHE)->dict[int,dict[str,Any]]:
    try:
        obj=json.loads(path.read_text(encoding="utf-8"))
    except (OSError,TypeError,ValueError):
        return {}
    if obj.get("schema_version")!=SCHEMA_VERSION or obj.get("source")!="Binance USD-M Futures WebSocket":
        return {}
    if obj.get("bin_ms")!=BIN_MS or obj.get("streams")!=["btcusdt@aggTrade","btcusdt@forceOrder"]:
        return {}
    rows=obj.get("rows")
    if not isinstance(rows,list):
        return {}
    out={}
    for row in rows:
        if not isinstance(row,dict):
            continue
        try:
            start=int(row["start_time_ms"])
            end=int(row["end_time_ms"])
            available=int(row["available_at_ms"])
            last_event=int(row["last_event_time_ms"])
        except (KeyError,TypeError,ValueError):
            continue
        if end!=start+BIN_MS or available<last_event:
            continue
        out[start]=dict(row)
    return out


async def _stream(url:str, parser, timeout_seconds:float, on_event:Callable[[dict[str,Any]],Awaitable[None]])->int:
    deadline=time.monotonic()+float(timeout_seconds)
    count=0
    while time.monotonic()<deadline:
        try:
            async with websockets.connect(url,ping_interval=20,ping_timeout=10,open_timeout=10,close_timeout=5,max_size=2_000_000) as ws:
                while time.monotonic()<deadline:
                    raw=await asyncio.wait_for(ws.recv(),timeout=min(45.0,max(0.25,deadline-time.monotonic())))
                    received=int(time.time()*1000)
                    try:
                        message=json.loads(raw)
                    except (TypeError,ValueError):
                        continue
                    event=parser(message,received)
                    if event is None:
                        continue
                    count+=1
                    await on_event(event)
        except Exception as exc:
            print(
                f"flow transport retry url={url} type={type(exc).__name__} "
                f"detail={str(exc)[:200]!r}",
                flush=True,
            )
        if time.monotonic()<deadline:
            await asyncio.sleep(0.5)
    return count


async def collect_flow_window(timeout_seconds:float, initial_bins:dict[int,dict[str,Any]]|None=None, checkpoint_seconds:float=30.0, on_checkpoint=None)->dict[int,dict[str,Any]]:
    bins=dict(initial_bins or {})
    latest_event=max((int(v.get("last_event_time_ms",0)) for v in bins.values()),default=0)
    lock=asyncio.Lock()
    last_checkpoint=time.monotonic()

    async def handle(event:dict[str,Any])->None:
        nonlocal latest_event,last_checkpoint
        async with lock:
            latest_event=max(latest_event,int(event["event_time_ms"]))
            ingest_event(bins,event)
            finalize_bins(bins,latest_event,int(event["retrieved_at_ms"]))
            snapshot=None
            if on_checkpoint is not None and time.monotonic()-last_checkpoint>=float(checkpoint_seconds):
                snapshot=dict(bins)
                last_checkpoint=time.monotonic()
        if snapshot is not None:
            result=on_checkpoint(snapshot)
            if asyncio.iscoroutine(result):
                await result

    await asyncio.gather(
        _stream(AGGTRADE_URL,parse_agg_trade_message,float(timeout_seconds),handle),
        _stream(FORCE_ORDER_URL,parse_force_order_message,float(timeout_seconds),handle),
    )
    async with lock:
        finalize_bins(bins,latest_event,int(time.time()*1000))
        final=dict(bins)
    if on_checkpoint is not None:
        result=on_checkpoint(final)
        if asyncio.iscoroutine(result):
            await result
    return final


def main()->int:
    p=argparse.ArgumentParser()
    p.add_argument("--capture-seconds",type=float,default=120.0)
    p.add_argument("--checkpoint-seconds",type=float,default=30.0)
    p.add_argument("--output",type=Path,default=DEFAULT_CACHE)
    a=p.parse_args()
    existing=load_cache(a.output)
    result=asyncio.run(collect_flow_window(a.capture_seconds,existing,a.checkpoint_seconds,lambda b:write_cache(b,a.output)))
    write_cache(result,a.output)
    rows=normalize_bins(result)
    print(json.dumps({"ok":bool(rows),"rows":len(rows),"latest_end_time_ms":rows[-1]["end_time_ms"] if rows else None,"latest_available_at_ms":rows[-1]["available_at_ms"] if rows else None},sort_keys=True))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
