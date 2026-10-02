import json
from datetime import datetime, timedelta, timezone

import numpy as np

from src.experience_meta_target_permutation_oos import (
    SEEDS,
    _eligible_train,
    _metrics,
)


def test_eligible_train_requires_strict_prediction_and_settlement_boundary():
    cutoff = datetime(2026, 10, 2, 1, 0, tzinfo=timezone.utc)
    rows = [
        {
            "created_at_utc": (cutoff - timedelta(minutes=30)).isoformat(),
            "settled_at_utc": (cutoff - timedelta(minutes=20)).isoformat(),
        },
        {
            "created_at_utc": (cutoff - timedelta(minutes=20)).isoformat(),
            "settled_at_utc": cutoff.isoformat(),
        },
        {
            "created_at_utc": cutoff.isoformat(),
            "settled_at_utc": (cutoff + timedelta(minutes=10)).isoformat(),
        },
    ]
    selected = _eligible_train(rows, cutoff)
    assert len(selected) == 1


def test_metrics_is_bounded_and_reports_auc_for_two_classes():
    y = np.asarray([0, 1, 0, 1], dtype=int)
    p = np.asarray([0.1, 0.9, 0.2, 0.8], dtype=float)
    out = _metrics(y, p)
    assert out["n"] == 4
    assert 0.0 <= out["logloss"]
    assert 0.0 <= out["brier"] <= 1.0
    assert 0.0 <= out["accuracy"] <= 1.0
    assert 0.0 <= out["auc"] <= 1.0


def test_metrics_json_contract_and_seed_count():
    out = _metrics(np.asarray([0, 0, 0]), np.asarray([0.2, 0.3, 0.4]))
    assert out["auc"] is None
    assert len(SEEDS) >= 3
    json.dumps(out)
