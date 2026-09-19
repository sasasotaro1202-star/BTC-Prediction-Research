import math

from selective_gate import decide


def test_abstains_below_confidence():
    d = decide({"DOWN": 0.45, "FLAT": 0.10, "UP": 0.45})
    assert not d.eligible
    assert d.reason == "confidence_below_gate"


def test_requires_margin():
    d = decide({"DOWN": 0.44, "FLAT": 0.12, "UP": 0.44}, min_confidence=0.40)
    assert not d.eligible
    assert d.reason == "class_margin_below_gate"


def test_requires_structural_agreement():
    d = decide(
        {"DOWN": 0.10, "FLAT": 0.10, "UP": 0.80},
        structural_class="DOWN",
    )
    assert not d.eligible
    assert d.reason == "model_structure_disagreement"


def test_rejects_cross_venue_divergence():
    d = decide(
        {"DOWN": 0.10, "FLAT": 0.10, "UP": 0.80},
        structural_class="UP",
        cross_exchange_gap=0.001,
    )
    assert not d.eligible
    assert d.reason == "cross_exchange_divergence"


def test_eligible_when_all_gates_pass():
    d = decide(
        {"DOWN": 0.05, "FLAT": 0.10, "UP": 0.85},
        structural_class="UP",
        secondary_class="UP",
        cross_exchange_gap=0.0001,
    )
    assert d.eligible
    assert d.predicted_class == "UP"
    assert math.isclose(d.confidence, 0.85)
