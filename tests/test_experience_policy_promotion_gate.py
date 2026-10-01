from datetime import datetime, timedelta, timezone

from src import experience_policy_promotion_gate as mod


def _research(ll5=-0.02, br5=-0.01, ll10=-0.01, br10=-0.005):
    return {
        "horizons": {
            "5m": {
                "status":"OK","prequential_test_rows":350,
                "baseline_error_probability":{"logloss":0.67,"brier":0.24},
                "hierarchical_experience_memory":{"logloss":0.67+ll5,"brier":0.24+br5},
            },
            "10m": {
                "status":"OK","prequential_test_rows":350,
                "baseline_error_probability":{"logloss":0.69,"brier":0.25},
                "hierarchical_experience_memory":{"logloss":0.69+ll10,"brier":0.25+br10},
            },
        }
    }


def _stability(pass_value=True):
    latest = {"non_worse": bool(pass_value)}
    horizon = {
        "status": "OK",
        "block_count": 5,
        "non_worse_fraction": 0.80 if pass_value else 0.40,
        "latest_block": latest,
    }
    return {"horizons": {"5m": dict(horizon), "10m": dict(horizon)}}


def _fresh_pit(verified=False):
    return {
        "generated_at_utc": (datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat(),
        "pit_verified": verified,
    }


def test_gate_holds_when_strict_pit_is_not_verified():
    report = mod.evaluate(
        _research(ll5=-0.05, br5=-0.02, ll10=-0.04, br10=-0.01),
        _fresh_pit(verified=False),
        _stability(True),
    )
    assert report["decision"] == "HOLD"
    assert "strict_pit_not_verified" in report["reasons"]
    assert "independent_holdout_evidence_required" in report["reasons"]


def test_gate_rejects_when_candidate_misses_metric_gate():
    report = mod.evaluate(
        _research(ll5=0.01, br5=0.001, ll10=0.0, br10=0.0),
        _fresh_pit(verified=True),
        config={"require_independent_holdout": False},
        stability=_stability(True),
    )
    assert report["decision"] == "REJECTED"
    assert any("candidate_does_not_meet_metric_gate" in r for r in report["reasons"])


def test_gate_requires_independent_holdout_even_when_metrics_pass():
    report = mod.evaluate(
        _research(ll5=-0.03, br5=-0.01, ll10=-0.03, br10=-0.01),
        _fresh_pit(verified=True),
        stability=_stability(True),
    )
    assert report["decision"] == "HOLD"
    assert "independent_holdout_evidence_required" in report["reasons"]


def test_gate_can_produce_promotion_candidate_only_with_all_requirements():
    report = mod.evaluate(
        _research(ll5=-0.04, br5=-0.02, ll10=-0.04, br10=-0.02),
        _fresh_pit(verified=True),
        config={"require_independent_holdout": False},
        stability=_stability(True),
    )
    assert report["decision"] == "PROMOTION_CANDIDATE"
    assert report["promotion_evidence_eligible"] is False
    assert report["production_changed"] is False


def test_gate_holds_stale_pit():
    stale={
        "generated_at_utc": (datetime.now(timezone.utc)-timedelta(hours=2)).isoformat(),
        "pit_verified": True,
    }
    report=mod.evaluate(
        _research(ll5=-0.04,br5=-0.02,ll10=-0.04,br10=-0.02),
        stale,
        config={"require_independent_holdout":False},
        stability=_stability(True),
    )
    assert report["decision"]=="HOLD"
    assert "pit_audit_not_fresh" in report["reasons"]


def test_gate_blocks_failed_stability():
    report = mod.evaluate(
        _research(ll5=-0.04, br5=-0.02, ll10=-0.04, br10=-0.02),
        _fresh_pit(verified=True),
        config={"require_independent_holdout": False},
        stability=_stability(False),
    )
    assert report["decision"] == "REJECTED"
    assert "5m:stability_gate_failed" in report["reasons"]
    assert "10m:stability_gate_failed" in report["reasons"]


def test_gate_holds_when_stability_is_missing():
    report = mod.evaluate(
        _research(ll5=-0.04, br5=-0.02, ll10=-0.04, br10=-0.02),
        _fresh_pit(verified=True),
        config={"require_independent_holdout": False},
        stability=None,
    )
    assert report["decision"] == "HOLD"
    assert "experience_stability_oos_missing" in report["reasons"]
