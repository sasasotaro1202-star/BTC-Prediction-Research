from pathlib import Path

import numpy as np

from src.recency_challenger import CLASSES, bootstrap_ci_accuracy, frequency_baseline, metrics, select_validation_model


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


def test_recency_report_has_explicit_promotion_firewall():
    s=Path("src/recency_challenger.py").read_text(encoding="utf-8")
    assert '"promotion_evidence_eligible": False' in s
    assert '"promotion_allowed": False' in s
    assert '"pit_status": "UNVERIFIABLE_HISTORICAL_AVAILABILITY"' in s
    assert '"analysis_git_sha": os.getenv("GITHUB_SHA") or "LOCAL_UNPINNED"' in s


def test_recency_workflow_has_pre_expensive_main_lineage_guard():
    s=Path(".github/workflows/btc_recency_challenger.yml").read_text(encoding="utf-8")
    assert "Validate main lineage before expensive research" in s
    assert 'UNSAFE_MAIN_RUN expected=${remote_sha} actual=${GITHUB_SHA}' in s


def test_validation_selection_balances_accuracy_and_proper_scores():
    results = [
        {"model": "accuracy_leader", "accuracy": 0.458, "logloss": 1.050, "brier": 0.635},
        {"model": "stable_leader", "accuracy": 0.452, "logloss": 1.037, "brier": 0.625},
        {"model": "weak", "accuracy": 0.420, "logloss": 1.060, "brier": 0.640},
    ]
    selected, ranked = select_validation_model(results)
    assert selected == "stable_leader"
    assert ranked[0]["validation_mean_rank"] <= ranked[1]["validation_mean_rank"]


def test_validation_selection_is_deterministic_on_ties():
    results = [
        {"model": "b", "accuracy": 0.45, "logloss": 1.0, "brier": 0.6},
        {"model": "a", "accuracy": 0.45, "logloss": 1.0, "brier": 0.6},
    ]
    selected, _ = select_validation_model(results)
    assert selected == "a"


def test_report_schema_has_top_level_safety_firewall():
    s=Path("src/recency_challenger.py").read_text(encoding="utf-8")
    assert '"pit_status": "UNVERIFIABLE_HISTORICAL_AVAILABILITY"' in s
    assert '"promotion_evidence_eligible": False' in s
    assert '"promotion_allowed": False' in s


def test_report_schema_has_top_level_validation_selection_policy():
    s = Path("src/recency_challenger.py").read_text(encoding="utf-8")
    assert '"validation_selection_policy": "equal_rank_accuracy_logloss_brier"' in s
