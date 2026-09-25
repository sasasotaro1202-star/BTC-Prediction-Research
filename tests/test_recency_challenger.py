import numpy as np

from src.recency_challenger import CLASSES, bootstrap_ci_accuracy, metrics


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
