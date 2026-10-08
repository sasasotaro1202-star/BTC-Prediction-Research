"""Kronos live-shadow research lane."""
from __future__ import annotations

import hashlib
import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from src.label_policy import direction_from_prices

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "data" / "external_research_runtime.json"
QUEUE = ROOT / "data" / "external_research_method_queue.json"
RESULT_DIR = ROOT / "data" / "external_research_shadow"
STATE_PATH = RESULT_DIR / "kronos_shadow.json"
MIN_SETTLED = 300
SAMPLE_PATHS = int(os.environ.get("KRONOS_SAMPLE_PATHS", "8"))
LOOKBACK_BARS = 400
MODEL_ID = "NeoQuasar/Kronos-mini"
TOKENIZER_ID = "NeoQuasar/Kronos-Tokenizer-2k"
DEVICE = "cpu"


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat()


def load_json(path: Path, default: Any = None) -> Any:
    if not path.is_file():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def active_candidate() -> str:
    runtime = load_json(RUNTIME, {})
    results = runtime.get("results", {}) if isinstance(runtime, dict) else {}
    queue = load_json(QUEUE, {})
    ordered: list[str] = []
    gate = queue.get("priority_gate", {}) if isinstance(queue, dict) else {}
    if isinstance(gate, dict):
        for group in ("immediate_local_reproduction", "predictive_method_second_wave", "research_infrastructure_next", "external_information_next", "frontier_only"):
            items = gate.get(group, [])
            if isinstance(items, list):
                for item in items:
                    if isinstance(item, dict):
                        repo = str(item.get("repository", "")).strip()
                        if repo and repo not in ordered:
                            ordered.append(repo)
    for repo in ordered:
        rec = results.get(repo)
        if isinstance(rec, dict) and rec.get("status") == "LOCAL_GATE_READY":
            return repo
    raise RuntimeError("no_external_method_local_gate_ready")


def read_contract_artifact(repository: str) -> dict[str, Any]:
    safe = repository.replace("/", "__").replace("\\", "__")
    obj = load_json(ROOT / "data" / "external_research_results" / f"{safe}.json")
    if not isinstance(obj, dict):
        raise RuntimeError("external_method_source_artifact_missing")
    return obj


def revision_from_contracts(artifact: dict[str, Any], model_id: str) -> str | None:
    checked = artifact.get("source_contract_verification", {}).get("checked", [])
    for item in checked:
        if isinstance(item, dict) and item.get("model_id") == model_id:
            value = item.get("sha")
            return str(value) if value else None
    return None


def fetch_closed_binance(limit: int = 5000) -> tuple[list[list[float]], datetime]:
    import urllib.parse
    import urllib.request

    out: list[list[float]] = []
    end_ms = int(time.time() * 1000) - 60_000
    while len(out) < limit:
        page_limit = min(1000, limit - len(out))
        q = urllib.parse.urlencode({"symbol": "BTCUSDT", "interval": "1m", "limit": page_limit, "endTime": end_ms})
        with urllib.request.urlopen("https://fapi.binance.com/fapi/v1/klines?" + q, timeout=30) as resp:
            rows = json.loads(resp.read().decode("utf-8"))
        if not isinstance(rows, list) or not rows:
            break
        parsed = []
        for row in rows:
            if not isinstance(row, list) or len(row) < 6:
                continue
            open_ms = int(row[0])
            if open_ms + 60_000 > int(time.time() * 1000):
                continue
            parsed.append([open_ms, float(row[1]), float(row[2]), float(row[3]), float(row[4]), float(row[5])])
        if not parsed:
            break
        out.extend(parsed)
        if len(parsed) < page_limit:
            break
        end_ms = min(int(row[0]) for row in parsed) - 1

    dedup = {int(r[0]): r for r in out}
    rows = [dedup[k] for k in sorted(dedup)]
    if len(rows) < limit:
        raise RuntimeError(f"insufficient_live_binance_history:{len(rows)}<{limit}")
    return rows[-limit:], utcnow()


def aggregate(rows: list[list[float]], minutes: int):
    import pandas as pd

    df = pd.DataFrame(rows, columns=["open_ms", "open", "high", "low", "close", "volume"])
    df["timestamps"] = pd.to_datetime(df["open_ms"], unit="ms", utc=True)
    parts = []
    for timestamp, group in df.groupby(df["timestamps"].dt.floor(f"{minutes}min"), sort=True):
        if len(group) != minutes:
            continue
        opens = group["open_ms"].to_numpy(dtype=np.int64)
        if not np.all(np.diff(opens) == 60_000):
            continue
        parts.append({
            "timestamps": timestamp,
            "open": float(group.iloc[0]["open"]),
            "high": float(group["high"].max()),
            "low": float(group["low"].min()),
            "close": float(group.iloc[-1]["close"]),
            "volume": float(group["volume"].sum()),
        })
    result = pd.DataFrame(parts)
    if len(result) < LOOKBACK_BARS:
        raise RuntimeError(f"insufficient_complete_{minutes}m_bars:{len(result)}<{LOOKBACK_BARS}")
    return result.tail(LOOKBACK_BARS).reset_index(drop=True)


def direction_probs(predicted_closes: list[float], base: float) -> dict[str, float]:
    counts = {"DOWN": 0, "FLAT": 0, "UP": 0}
    for close in predicted_closes:
        counts[direction_from_prices(base, float(close))] += 1
    n = float(len(predicted_closes))
    return {key: counts[key] / n for key in counts}


def ece(probs: list[dict[str, float]], actual: list[str], bins: int = 10) -> float:
    if not probs:
        return float("nan")
    confidence = np.asarray([max(p.values()) for p in probs], dtype=float)
    hits = np.asarray([1.0 if max(p, key=p.get) == label else 0.0 for p, label in zip(probs, actual)], dtype=float)
    result = 0.0
    for i in range(bins):
        lo = i / bins
        hi = (i + 1) / bins
        mask = (confidence >= lo) & ((confidence < hi) if i < bins - 1 else (confidence <= hi))
        if not np.any(mask):
            continue
        result += float(mask.mean()) * abs(float(confidence[mask].mean()) - float(hits[mask].mean()))
    return result


def metrics(records: list[dict[str, Any]], horizon: str) -> dict[str, Any]:
    settled = [r for r in records if r.get(f"actual_direction_{horizon}")]
    if not settled:
        return {"n": 0, "accuracy": None, "logloss": None, "brier": None, "ece": None}
    labels = [str(r[f"actual_direction_{horizon}"]) for r in settled]
    probabilities = [r[f"p_{horizon}"] for r in settled]
    loglosses = []
    briers = []
    hits = []
    for probability, label in zip(probabilities, labels):
        py = max(1e-6, float(probability.get(label, 0.0)))
        loglosses.append(-math.log(py))
        target = {"DOWN": 0.0, "FLAT": 0.0, "UP": 0.0}
        target[label] = 1.0
        briers.append(sum((float(probability.get(k, 0.0)) - target[k]) ** 2 for k in target) / 3.0)
        hits.append(1.0 if max(probability, key=probability.get) == label else 0.0)
    return {
        "n": len(settled),
        "accuracy": float(np.mean(hits)),
        "logloss": float(np.mean(loglosses)),
        "brier": float(np.mean(briers)),
        "ece": ece(probabilities, labels),
    }


def settle_records(records: list[dict[str, Any]]) -> bool:
    import urllib.parse
    import urllib.request

    changed = False
    now_ms = int(time.time() * 1000)
    for record in records:
        for horizon, interval in (("5m", 300_000), ("10m", 600_000)):
            if record.get(f"actual_direction_{horizon}"):
                continue
            target_start = int(record[f"target_start_ms_{horizon}"])
            target_end = target_start + interval
            if target_end > now_ms:
                continue
            query = urllib.parse.urlencode({
                "symbol": "BTCUSDT",
                "interval": "1m",
                "startTime": target_start,
                "endTime": target_end - 1,
                "limit": interval // 60_000,
            })
            with urllib.request.urlopen("https://fapi.binance.com/fapi/v1/klines?" + query, timeout=30) as resp:
                rows = json.loads(resp.read().decode("utf-8"))
            target_rows = [row for row in rows if isinstance(row, list) and int(row[0]) >= target_start and int(row[0]) < target_end]
            if len(target_rows) != interval // 60_000:
                continue
            base = float(record[f"base_close_{horizon}"])
            actual = float(target_rows[-1][4])
            record[f"actual_price_{horizon}"] = actual
            record[f"actual_direction_{horizon}"] = direction_from_prices(base, actual)
            record[f"outcome_retrieved_at_{horizon}"] = iso(utcnow())
            record[f"outcome_source_{horizon}"] = "binance_futures_rest"
            changed = True
    return changed


def predict_once() -> dict[str, Any]:
    import torch
    from model import Kronos, KronosPredictor, KronosTokenizer

    repository = active_candidate()
    if repository != "shiyu-coder/Kronos":
        raise RuntimeError(f"unsupported_shadow_adapter:{repository}")

    artifact = read_contract_artifact(repository)
    model_revision = revision_from_contracts(artifact, MODEL_ID)
    tokenizer_revision = revision_from_contracts(artifact, TOKENIZER_ID)
    if not model_revision or not tokenizer_revision:
        raise RuntimeError("kronos_weight_revision_missing")

    rows, retrieved_at = fetch_closed_binance(5000)
    prediction_cutoff = utcnow()
    frames = {minutes: aggregate(rows, minutes) for minutes in (5, 10)}

    tokenizer = KronosTokenizer.from_pretrained(TOKENIZER_ID, revision=tokenizer_revision, local_files_only=False)
    model = Kronos.from_pretrained(MODEL_ID, revision=model_revision, local_files_only=False)
    tokenizer.eval()
    model.eval()
    predictor = KronosPredictor(model, tokenizer, device=DEVICE, max_context=512)

    prediction_id = hashlib.sha256(
        f"kronos|{prediction_cutoff.isoformat()}|{model_revision}|{tokenizer_revision}".encode()
    ).hexdigest()[:24]
    record: dict[str, Any] = {
        "schema_version": 1,
        "prediction_id": prediction_id,
        "repository": repository,
        "model_id": MODEL_ID,
        "tokenizer_id": TOKENIZER_ID,
        "model_revision": model_revision,
        "tokenizer_revision": tokenizer_revision,
        "prediction_cutoff_utc": iso(prediction_cutoff),
        "source": "binance_futures_rest",
        "available_at_utc": iso(prediction_cutoff),
        "retrieved_at_utc": iso(retrieved_at),
        "research_only": True,
        "production_changed": False,
        "promotion_allowed": False,
        "external_performance_transfer_allowed": False,
        "sample_paths": SAMPLE_PATHS,
    }

    for horizon_minutes in (5, 10):
        context = frames[horizon_minutes]
        base_close = float(context.iloc[-1]["close"])
        last_start = int(context.iloc[-1]["timestamps"].timestamp() * 1000)
        target_start = last_start + horizon_minutes * 60_000
        future_ts = pd.Series([pd.Timestamp(target_start, unit="ms", tz="UTC")], name="timestamps")
        x_df = context[["open", "high", "low", "close", "volume"]].copy()
        x_df["amount"] = x_df["volume"]
        x_ts = context["timestamps"].reset_index(drop=True)
        predicted_closes = []
        for sample in range(SAMPLE_PATHS):
            torch.manual_seed(17_000 + sample + horizon_minutes * 100)
            with torch.no_grad():
                predicted = predictor.predict(
                    df=x_df.reset_index(drop=True),
                    x_timestamp=x_ts,
                    y_timestamp=future_ts,
                    pred_len=1,
                    T=1.0,
                    top_p=0.95,
                    sample_count=1,
                    verbose=False,
                )
            predicted_closes.append(float(predicted.iloc[0]["close"]))
        record[f"p_{horizon_minutes}m"] = direction_probs(predicted_closes, base_close)
        record[f"base_close_{horizon_minutes}m"] = base_close
        record[f"target_start_ms_{horizon_minutes}m"] = target_start
        record[f"target_at_utc_{horizon_minutes}m"] = iso(datetime.fromtimestamp((target_start + horizon_minutes * 60_000) / 1000, timezone.utc))
    return record


def main() -> int:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    state = load_json(
        STATE_PATH,
        {"schema_version": 1, "repository": "shiyu-coder/Kronos", "research_only": True, "production_changed": False, "records": []},
    )
    records = state.get("records", []) if isinstance(state, dict) else []
    if not isinstance(records, list):
        records = []

    settled_changed = settle_records(records)
    mature = all(len([r for r in records if r.get(f"actual_direction_{h}")]) >= MIN_SETTLED for h in ("5m", "10m"))
    if not mature:
        record = predict_once()
        if not any(r.get("prediction_id") == record["prediction_id"] for r in records):
            records.append(record)

    state.update({
        "schema_version": 1,
        "repository": "shiyu-coder/Kronos",
        "research_only": True,
        "production_changed": False,
        "updated_at_utc": iso(utcnow()),
        "records": records[-1000:],
        "metrics": {h: metrics(records, h) for h in ("5m", "10m")},
        "status": "SHADOW_MATURED" if mature else "SHADOW_COLLECTING",
        "settlement_changed": settled_changed,
    })
    save_json(STATE_PATH, state)

    runtime = load_json(RUNTIME, {})
    results = runtime.get("results", {}) if isinstance(runtime, dict) else {}
    kronos = results.get("shiyu-coder/Kronos")
    if isinstance(kronos, dict) and mature:
        kronos["status"] = "SHADOW_MATURED"
        kronos["shadow_metrics"] = state["metrics"]
        results["shiyu-coder/Kronos"] = kronos
        runtime["results"] = results
        runtime["updated_at_utc"] = iso(utcnow())
        runtime["research_only"] = True
        runtime["production_changed"] = False
        save_json(RUNTIME, runtime)

    print(json.dumps({
        "status": state["status"],
        "records": len(records),
        "settled_5m": state["metrics"]["5m"]["n"],
        "settled_10m": state["metrics"]["10m"]["n"],
        "research_only": True,
        "production_changed": False,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
