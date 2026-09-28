import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.predict import _validate_persisted_provenance


def _scenario(*, temporal_basis="retrieval_snapshot"):
    t = datetime(2026, 9, 29, 0, 0, tzinfo=timezone.utc).isoformat()
    return {
        "decision_time_utc": t,
        "provenance": {
            "available_at": t,
            "retrieved_at": t,
            "prediction_cutoff": t,
            "sources": {
                "bybit_ticker": {
                    "information_origin": "Bybit",
                    "event_time": None,
                    "available_at": t,
                    "retrieved_at": t,
                    "prediction_cutoff": t,
                    "publication_time": None,
                    "revision_time": None,
                    "status": "ok",
                    "temporal_basis": temporal_basis,
                    "fields": ["openInterest"],
                }
            },
        },
    }


def test_retrieval_snapshot_without_event_time_is_explicitly_valid():
    scenario = _scenario()
    now = datetime(2026, 9, 29, 0, 0, tzinfo=timezone.utc)
    _validate_persisted_provenance(scenario, now)


def test_missing_snapshot_basis_with_unknown_event_time_fails_closed():
    scenario = _scenario(temporal_basis=None)
    now = datetime(2026, 9, 29, 0, 0, tzinfo=timezone.utc)
    with pytest.raises(ValueError, match="prediction_source_event_time_missing_without_snapshot_basis"):
        _validate_persisted_provenance(scenario, now)
