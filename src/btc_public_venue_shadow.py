"""Research-only public BTC venue shadow collector with fail-closed PIT parsing."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any, Callable

import websockets

HYPERLIQUID_WS = "wss://api.hyperliquid.xyz/ws"
BITGET_WS = "wss://ws.bitget.com/v2/ws/public"


def _num(value: Any, positive: bool = True) -> bool:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(x) and (x > 0 if positive else x >= 0)


def _hash(payload: Any) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(raw).hexdigest()


def normalize_event(source: str, venue: str, event_type: str, event_time_ms: int,
                    retrieved_at_ms: int, payload: dict[str, Any]) -> dict[str, Any] | None:
    try:
        event_time_ms = int(event_time_ms)
        retrieved_at_ms = int(retrieved_at_ms)
    except (TypeError, ValueError):
        return None
    if event_time_ms <= 0 or retrieved_at_ms <= 0 or event_time_ms > retrieved_at_ms:
        return None
    payload_sha = _hash(payload)
    event_id = hashlib.sha256(
        f"{source}|{venue}|{event_type}|{event_time_ms}|{payload_sha}".encode()
    ).hexdigest()[:32]
    return {
        "schema_version": 1,
        "event_id": event_id,
        "source": source,
        "venue": venue,
        "event_type": event_type,
        "event_time_ms": event_time_ms,
        "available_at_ms": retrieved_at_ms,
        "retrieved_at_ms": retrieved_at_ms,
        "payload_sha256": payload_sha,
        "payload": payload,
    }


def parse_hyperliquid_message(message: Any, retrieved_at_ms: int,
                              coin: str = "BTC") -> list[dict[str, Any]]:
    if not isinstance(message, dict):
        return []
    channel, data = message.get("channel"), message.get("data")
    if channel == "l2Book" and isinstance(data, dict) and data.get("coin") == coin:
        levels = data.get("levels")
        if not isinstance(levels, list) or len(levels) != 2 or not isinstance(data.get("time"), int):
            return []
        clean_levels = []
        for side in levels:
            if not isinstance(side, list):
                return []
            clean_side = []
            for level in side:
                if not isinstance(level, dict) or not _num(level.get("px")) or not _num(level.get("sz"), False):
                    return []
                try:
                    n = int(level.get("n", 0))
                except (TypeError, ValueError):
                    return []
                if n < 0:
                    return []
                clean_side.append({"px": str(level["px"]), "sz": str(level["sz"]), "n": n})
            clean_levels.append(clean_side)
        event = normalize_event(
            "hyperliquid", "hyperliquid_perp", "l2_book", data["time"], retrieved_at_ms,
            {"coin": coin, "levels": clean_levels},
        )
        return [event] if event else []

    if channel == "trades" and isinstance(data, list):
        out = []
        for trade in data:
            if not isinstance(trade, dict) or trade.get("coin") != coin:
                continue
            try:
                event_time = int(trade["time"])
                px, sz = float(trade["px"]), float(trade["sz"])
            except (KeyError, TypeError, ValueError):
                continue
            if not (_num(px) and _num(sz)):
                continue
            payload = {
                "coin": coin, "side": trade.get("side"), "px": str(trade["px"]),
                "sz": str(trade["sz"]), "time": event_time,
                "hash": trade.get("hash"), "tid": trade.get("tid"),
            }
            event = normalize_event(
                "hyperliquid", "hyperliquid_perp", "trade",
                event_time, retrieved_at_ms, payload,
            )
            if event:
                out.append(event)
        return out
    return []


def parse_bitget_message(message: Any, retrieved_at_ms: int,
                         symbol: str = "BTCUSDT") -> list[dict[str, Any]]:
    if not isinstance(message, dict):
        return []
    arg, data = message.get("arg"), message.get("data")
    if not isinstance(arg, dict) or not isinstance(data, list):
        return []
    topic = arg.get("topic")
    out = []
    for item in data:
        if not isinstance(item, dict) or item.get("symbol", symbol) != symbol:
            continue
        try:
            event_time = int(item.get("ts", message.get("ts", 0)))
        except (TypeError, ValueError):
            continue
        if event_time <= 0:
            continue
        if topic in {"books5", "rpi-books5"}:
            if not isinstance(item.get("bids"), list) or not isinstance(item.get("asks"), list):
                continue
            payload = {
                "symbol": symbol, "bids": item["bids"], "asks": item["asks"],
                "action": message.get("action"), "seq": item.get("seq", message.get("seq")),
            }
            event_type = "l2_book"
        elif topic == "publicTrade":
            if not (_num(item.get("price")) and _num(item.get("size"), True)):
                continue
            payload = {
                "symbol": symbol, "side": item.get("side"),
                "price": str(item["price"]), "size": str(item["size"]),
                "trade_id": item.get("tradeId"), "ts": event_time,
            }
            event_type = "trade"
        elif topic == "liquidation":
            if not (_num(item.get("price")) and _num(item.get("amount"), True)):
                continue
            payload = {
                "symbol": symbol, "side": item.get("side"),
                "price": str(item["price"]), "amount": str(item["amount"]), "ts": event_time,
            }
            event_type = "liquidation"
        else:
            continue
        event = normalize_event(
            "bitget", "bitget_usdt_futures", event_type, event_time, retrieved_at_ms, payload
        )
        if event:
            out.append(event)
    return out


async def _app_ping(ws: Any, mode: str) -> None:
    while True:
        await asyncio.sleep(25)
        try:
            await ws.send(json.dumps({"method": "ping"}) if mode == "hyperliquid" else "ping")
        except Exception:
            return


async def capture_socket(url: str, subscription: dict[str, Any],
                         parser: Callable[[Any, int], list[dict[str, Any]]],
                         duration_seconds: float, heartbeat_mode: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    try:
        async with websockets.connect(
            url, ping_interval=20, ping_timeout=10, open_timeout=10,
            close_timeout=5, max_size=4_000_000,
        ) as ws:
            await ws.send(json.dumps(subscription, separators=(",", ":")))
            ping_task = asyncio.create_task(_app_ping(ws, heartbeat_mode))
            deadline = time.monotonic() + max(1.0, duration_seconds)
            try:
                while time.monotonic() < deadline:
                    try:
                        raw = await asyncio.wait_for(ws.recv(), timeout=min(5.0, deadline - time.monotonic()))
                    except asyncio.TimeoutError:
                        continue
                    except websockets.exceptions.ConnectionClosed:
                        break
                    try:
                        message = json.loads(raw)
                    except (TypeError, ValueError):
                        continue
                    events.extend(parser(message, int(time.time() * 1000)))
            finally:
                ping_task.cancel()
                try:
                    await ping_task
                except asyncio.CancelledError:
                    pass
    except Exception:
        return events
    return events


async def collect_shadow(duration_seconds: float = 60.0) -> list[dict[str, Any]]:
    jobs = [
        (HYPERLIQUID_WS, {"method": "subscribe", "subscription": {"type": "l2Book", "coin": "BTC"}},
         parse_hyperliquid_message, "hyperliquid"),
        (HYPERLIQUID_WS, {"method": "subscribe", "subscription": {"type": "trades", "coin": "BTC"}},
         parse_hyperliquid_message, "hyperliquid"),
        (BITGET_WS, {"op": "subscribe", "args": [
            {"instType": "usdt-futures", "topic": "books5", "symbol": "BTCUSDT"},
            {"instType": "usdt-futures", "topic": "publicTrade", "symbol": "BTCUSDT"},
            {"instType": "usdt-futures", "topic": "liquidation"},
        ]}, parse_bitget_message, "bitget"),
    ]
    groups = await asyncio.gather(*(
        capture_socket(url, subscription, parser, duration_seconds, heartbeat)
        for url, subscription, parser, heartbeat in jobs
    ))
    events = [event for group in groups for event in group]
    events.sort(key=lambda e: (e["event_time_ms"], e["event_id"]))
    return events


def validate_event_set(events: list[dict[str, Any]], cutoff_ms: int | None = None) -> dict[str, Any]:
    cutoff = int(time.time() * 1000) if cutoff_ms is None else int(cutoff_ms)
    accepted, rejected, future, duplicates = [], 0, 0, 0
    seen = set()
    for event in events:
        try:
            if event["event_time_ms"] > cutoff or event["available_at_ms"] > cutoff:
                future += 1
                continue
            if event["event_time_ms"] > event["available_at_ms"]:
                rejected += 1
                continue
            if event["event_id"] in seen:
                duplicates += 1
                continue
            seen.add(event["event_id"])
            accepted.append(event)
        except (KeyError, TypeError, ValueError):
            rejected += 1
    return {
        "accepted": len(accepted), "rejected": rejected,
        "future": future, "duplicates": duplicates,
        "sources": sorted({e["source"] for e in accepted}),
        "event_types": sorted({e["event_type"] for e in accepted}),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration-seconds", type=float, default=60.0)
    parser.add_argument("--output", type=Path, default=Path("data/shadow/btc_public_venue_events.json"))
    args = parser.parse_args()
    events = asyncio.run(collect_shadow(args.duration_seconds))
    validation = validate_event_set(events)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "schema_version": 1,
        "generated_at_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        "duration_seconds": args.duration_seconds,
        "event_count": validation["accepted"],
        "validation": validation,
        "events": events,
    }, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    return 0 if validation["accepted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())


def _window_events(events: list[dict[str, Any]], cutoff_ms: int, window_ms: int) -> list[dict[str, Any]]:
    lower = int(cutoff_ms) - int(window_ms)
    return [
        event for event in events
        if isinstance(event, dict)
        and lower <= int(event.get("event_time_ms", -1)) <= int(cutoff_ms)
        and int(event.get("available_at_ms", 2**63 - 1)) <= int(cutoff_ms)
    ]


def derive_situation_card(
    events: list[dict[str, Any]],
    cutoff_ms: int,
    window_ms: int = 60_000,
) -> dict[str, Any]:
    """Derive PIT-safe public-venue state without model-side transformations."""
    selected = _window_events(events, cutoff_ms, window_ms)
    book_mids: dict[str, float] = {}
    spread_bps: dict[str, float] = {}
    depth_imbalance: dict[str, float] = {}
    trade_buy: dict[str, float] = {}
    trade_sell: dict[str, float] = {}
    liquidation_buy: dict[str, float] = {}
    liquidation_sell: dict[str, float] = {}
    used_ids: list[str] = []

    for event in selected:
        payload = event.get("payload")
        if not isinstance(payload, dict):
            continue
        source = str(event.get("source"))
        try:
            if event["event_type"] == "l2_book":
                bids = payload.get("bids") or payload.get("levels", [[], []])[0]
                asks = payload.get("asks") or payload.get("levels", [[], []])[1]
                if not bids or not asks:
                    continue
                bid_px = float(bids[0][0] if isinstance(bids[0], (list, tuple)) else bids[0].get("px"))
                ask_px = float(asks[0][0] if isinstance(asks[0], (list, tuple)) else asks[0].get("px"))
                if not (_num(bid_px) and _num(ask_px) and bid_px <= ask_px):
                    continue
                mid = (bid_px + ask_px) / 2.0
                book_mids[source] = mid
                spread_bps[source] = (ask_px - bid_px) / mid * 10_000.0
                bid_sz = sum(float(x[1] if isinstance(x, (list, tuple)) else x.get("sz", 0)) for x in bids[:5])
                ask_sz = sum(float(x[1] if isinstance(x, (list, tuple)) else x.get("sz", 0)) for x in asks[:5])
                if bid_sz + ask_sz > 0:
                    depth_imbalance[source] = (bid_sz - ask_sz) / (bid_sz + ask_sz)
                used_ids.append(str(event["event_id"]))
            elif event["event_type"] == "trade":
                side = str(payload.get("side", "")).upper()
                size = float(payload.get("size", payload.get("sz", 0)))
                if not _num(size):
                    continue
                # Hyperliquid B/A and Bitget buy/sell are preserved; no
                # interpretation is made beyond their venue-native labels.
                if side in {"B", "BUY"}:
                    trade_buy[source] = trade_buy.get(source, 0.0) + size
                elif side in {"A", "SELL"}:
                    trade_sell[source] = trade_sell.get(source, 0.0) + size
                else:
                    continue
                used_ids.append(str(event["event_id"]))
            elif event["event_type"] == "liquidation":
                side = str(payload.get("side", "")).upper()
                amount = float(payload.get("amount", 0))
                if not _num(amount):
                    continue
                if side == "BUY":
                    liquidation_buy[source] = liquidation_buy.get(source, 0.0) + amount
                elif side == "SELL":
                    liquidation_sell[source] = liquidation_sell.get(source, 0.0) + amount
                else:
                    continue
                used_ids.append(str(event["event_id"]))
        except (TypeError, ValueError, KeyError, IndexError):
            continue

    cross_venue_gap_bps = None
    if len(book_mids) >= 2:
        values = list(book_mids.values())
        reference = sum(values) / len(values)
        if reference > 0:
            cross_venue_gap_bps = (max(values) - min(values)) / reference * 10_000.0

    def _ratio(buy: float, sell: float) -> float | None:
        total = buy + sell
        return None if total <= 0 else (buy - sell) / total

    return {
        "schema_version": 1,
        "prediction_cutoff_ms": int(cutoff_ms),
        "window_ms": int(window_ms),
        "sources": sorted({str(event.get("source")) for event in selected}),
        "event_count": len(selected),
        "used_event_ids": sorted(set(used_ids)),
        "book_mid": book_mids,
        "spread_bps": spread_bps,
        "depth_imbalance_5": depth_imbalance,
        "trade_buy_volume": trade_buy,
        "trade_sell_volume": trade_sell,
        "trade_imbalance": {
            source: _ratio(trade_buy.get(source, 0.0), trade_sell.get(source, 0.0))
            for source in sorted(set(trade_buy) | set(trade_sell))
        },
        "liquidation_buy_amount": liquidation_buy,
        "liquidation_sell_amount": liquidation_sell,
        "cross_venue_gap_bps": cross_venue_gap_bps,
        "freshness_ms": (
            int(cutoff_ms) - max(int(event["event_time_ms"]) for event in selected)
            if selected else None
        ),
    }
