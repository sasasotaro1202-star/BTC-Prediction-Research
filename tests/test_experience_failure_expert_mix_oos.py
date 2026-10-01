import json

import numpy as np

from src.experience_failure_expert_mix_oos import (
    CANDIDATES,
    ETA,
    MIN_TRAIN,
    VALIDATION_SIZE,
    WEIGHT_FLOOR,
    _binary_logloss,
    _eligible_prior,
    _mixed_risk,
    _sequential_validation_risks,
    _update_weights,
)


def _row(
    *,
    experience_id=1,
    created="2026-10-01T00:00:00+00:00",
    settled="2026-10-01T00:05:00+00:00",
    correct=1,
    regime="TREND",
    direction="UP",
    mode="binance_primary",
    info="CLEAN",
):
    p = {"DOWN": 0.1, "FLAT": 0.1, "UP": 0.8}
    return {
        "experience_id": experience_id,
        "horizon": "5m",
        "created_at_utc": created,
        "settled_at_utc": settled,
        "correct": correct,
        "regime": regime,
        "predicted_direction": direction,
        "production_mode": mode,
        "information_state": info,
        "actual_direction": "UP" if correct else "DOWN",
        "probability_json": json.dumps(p),
    }


def test_update_weights_rewards_lower_loss_expert():
    weights = {name: 1.0 / 3.0 for name in CANDIDATES}
    losses = {"global": 0.80, "case_memory": 0.60, "meta": 0.75}
    updated = _update_weights(weights, losses)
    assert np.isclose(sum(updated.values()), 1.0)
    assert updated["case_memory"] > updated["global"]
    assert updated["case_memory"] > updated["meta"]
    assert all(v > 0.0 for v in updated.values())


def test_update_weights_keeps_experts_alive():
    weights = {name: 1.0 / 3.0 for name in CANDIDATES}
    losses = {"global": 0.0, "case_memory": 10.0, "meta": 10.0}
    updated = _update_weights(weights, losses)
    assert np.isclose(sum(updated.values()), 1.0)
    assert min(updated.values()) >= (WEIGHT_FLOOR / len(CANDIDATES)) - 1e-12


def test_mixed_risk_is_bounded_and_weighted():
    risks = {"global": 0.9, "case_memory": 0.3, "meta": 0.6}
    weights = {"global": 0.2, "case_memory": 0.5, "meta": 0.3}
    mixed = _mixed_risk(risks, weights)
    assert 0.0 <= mixed <= 1.0
    assert np.isclose(mixed, 0.2 * 0.9 + 0.5 * 0.3 + 0.3 * 0.6)


def test_eligible_prior_requires_both_prediction_and_settlement_before_current():
    current = _row(
        experience_id=999,
        created="2026-10-01T01:00:00+00:00",
        settled="2026-10-01T01:05:00+00:00",
    )
    usable = _row(
        experience_id=1,
        created="2026-10-01T00:30:00+00:00",
        settled="2026-10-01T00:35:00+00:00",
    )
    unsettled = _row(
        experience_id=2,
        created="2026-10-01T00:30:00+00:00",
        settled="2026-10-01T01:10:00+00:00",
    )
    assert [r["experience_id"] for r in _eligible_prior([usable, unsettled], current)] == [1]


def test_sequential_validation_risks_are_finite_and_bounded():
    train = [
        _row(
            experience_id=i,
            created=f"2026-09-20T00:{i:02d}:00+00:00",
            settled=f"2026-09-20T00:{i:02d}:30+00:00",
            correct=(i % 2 == 0),
            direction="UP" if i % 2 == 0 else "DOWN",
        )
        for i in range(1, MIN_TRAIN + 5)
    ]
    validation = [
        _row(
            experience_id=1000 + i,
            created=f"2026-10-01T00:{i:02d}:00+00:00",
            settled=f"2026-10-01T00:{i:02d}:30+00:00",
            correct=(i % 2 == 0),
            direction="UP" if i % 2 == 0 else "DOWN",
        )
        for i in range(1, 6)
    ]
    values = _sequential_validation_risks(train, validation)
    assert set(values) == {"global", "case_memory"}
    for value in values.values():
        assert value.shape == (5,)
        assert np.isfinite(value).all()
        assert np.all((value >= 0.0) & (value <= 1.0))


def test_binary_logloss_contract():
    y = np.asarray([0, 1, 1], dtype=int)
    p = np.asarray([0.2, 0.7, 0.8], dtype=float)
    assert np.isfinite(_binary_logloss(y, p))
    assert ETA > 0.0
    assert VALIDATION_SIZE >= 20
