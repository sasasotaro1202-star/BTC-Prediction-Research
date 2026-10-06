import numpy as np

from src.recency_challenger import CLASSES, bootstrap_ci_accuracy, frequency_baseline, metrics


def test_metrics_three_class_known_case():
    y = np.array(["DOWN", "FLAT", "UP"], dtype=object)
    p = np.eye(3)
    out = metrics(y, p)
    assert out["accuracy"] > 0.999999
    assert out["brier"] < 1e-12


def test_bootstrap_accuracy_is_paired():
    y = np.array(["DOWN", "FLAT", "UP"] * 2000, dtype=object)
    p = np.tile(np.eye(3), (2000, 1))
    q = np.roll(p, 1, axis=1)
    out = bootstrap_ci_accuracy(y, p, q)
    assert out["ci95_low"] > 0.0


def test_frequency_baseline_uses_pre_holdout_reference_distribution():
    reference = np.array(["DOWN", "FLAT", "FLAT", "UP"], dtype=object)
    out = frequency_baseline(reference, 3)
    expected = np.tile(np.array([0.25, 0.50, 0.25]), (3, 1))
    assert np.allclose(out, expected)


def test_frequency_baseline_rejects_empty_reference():
    try:
        frequency_baseline(np.array([], dtype=object), 2)
    except ValueError as exc:
        assert str(exc) == "baseline_reference_must_not_be_empty"
    else:
        raise AssertionError("expected ValueError")
