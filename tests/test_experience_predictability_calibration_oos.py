import json
from datetime import datetime, timedelta, timezone

import numpy as np

from src.experience_predictability_calibration_oos import (
    _fit_calibrator,
    _paired_bootstrap,
    _risk_metrics,
)


def test_calibrator_excludes_current_and_unmatured_history():
    current = datetime(2026, 10, 2, 1, 0, tzinfo=timezone.utc)
    history = [
        (
            current - timedelta(minutes=30),
            current - timedelta(minutes=20),
            0.9,
            1,
        ),
        (
            current - timedelta(minutes=20),
            current + timedelta(minutes=5),
            0.1,
            0,
        ),
        (
            current,
            current + timedelta(minutes=20),
            0.95,
            1,
        ),
    ]
    value, method, count = _fit_calibrator(history, current, 0.4)
    assert 0.0 <= value <= 1.0
    assert method == "FALLBACK_RAW"
    assert count == 1


def test_calibrator_fits_from_prior_matured_cases_only():
    current = datetime(2026, 10, 2, 2, 0, tzinfo=timezone.utc)
    history = []
    for i in range(40):
        created = current - timedelta(minutes=200 - i)
        settled = created + timedelta(minutes=1)
        risk = 0.2 + 0.015 * i
        error = int(risk > 0.5)
        history.append((created, settled, risk, error))
    value, method, count = _fit_calibrator(history, current, 0.25)
    assert method == "FITTED_PRIOR_ONLY_LOGISTIC"
    assert count == 40
    assert 0.01 <= value <= 0.99
    assert value > 0.25


def test_risk_metrics_are_bounded_and_auc_optional():
    y = np.asarray([0, 1, 0, 1], dtype=int)
    p = np.asarray([0.1, 0.8, 0.2, 0.7], dtype=float)
    report = _risk_metrics(y, p)
    assert report["n"] == 4
    assert 0.0 <= report["brier"] <= 1.0
    assert 0.0 <= report["ece"] <= 1.0
    assert 0.0 <= report["auc"] <= 1.0


def test_paired_bootstrap_is_finite():
    values = np.asarray([-0.2, 0.0, 0.1, -0.05] * 5, dtype=float)
    report = _paired_bootstrap(values)
    assert all(np.isfinite(report[k]) for k in ("mean", "low", "high"))
    assert report["low"] <= report["mean"] <= report["high"]


def test_payload_helpers_are_json_serializable():
    current = datetime(2026, 10, 2, tzinfo=timezone.utc)
    history = [
        (
            current - timedelta(minutes=20),
            current - timedelta(minutes=10),
            0.3,
            0,
        )
    ]
    report = _fit_calibrator(history, current, 0.3)
    json.dumps({"fit": report})
