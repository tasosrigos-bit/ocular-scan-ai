"""Table-driven regression tests for the CDS decision logic. This is the
safety net: any change to configs/cds/rules_v1.yaml or cds/rules.py that
moves an urgency assignment must update this table deliberately."""

import pytest

from oct_cds.cds.rules import CDSRuleEngine, load_rules
from oct_cds.cds.schema import CaseInput, ModelResult, Urgency


def _mr(top_key: str, top_p: float, ood: float = 0.1) -> ModelResult:
    others = {
        k: (1 - top_p) / 7
        for k in ["AMD", "CNV", "CSR", "DME", "DR", "Drusen", "Macular Hole", "Normal"]
        if k != top_key
    }
    return ModelResult(probs={top_key: top_p, **others}, ood_score=ood,
                       model_version="test")


ENGINE = CDSRuleEngine()
CASE = CaseInput(image_path="x.png")


@pytest.mark.parametrize(
    "top_key,top_p,expected_urgency,abstain",
    [
        ("CNV", 0.92, Urgency.urgent, False),
        ("DME", 0.90, Urgency.urgent, False),
        ("Macular Hole", 0.88, Urgency.urgent, False),
        ("AMD", 0.85, Urgency.soon, False),
        ("CSR", 0.80, Urgency.soon, False),
        ("DR", 0.78, Urgency.soon, False),
        ("Drusen", 0.83, Urgency.routine, False),
        ("Normal", 0.97, Urgency.none, False),
        ("CNV", 0.55, Urgency.soon, True),        # below min_confidence -> abstain
    ],
)
def test_urgency_table(top_key, top_p, expected_urgency, abstain):
    rec = ENGINE.evaluate(_mr(top_key, top_p), CASE)
    assert rec.abstained is abstain
    assert rec.urgency == expected_urgency
    if not abstain:
        assert rec.predicted_class == top_key


def test_low_margin_abstains_even_when_confident_enough():
    mr = ModelResult(
        probs={"CNV": 0.46, "DME": 0.44, "AMD": 0.02, "CSR": 0.02, "DR": 0.02,
               "Drusen": 0.02, "Macular Hole": 0.01, "Normal": 0.01},
        ood_score=0.1, model_version="test",
    )
    rec = ENGINE.evaluate(mr, CASE)
    assert rec.abstained and not rec.ood_rejected


def test_ood_rejection_precedes_classification():
    rec = ENGINE.evaluate(_mr("CNV", 0.99, ood=0.95), CASE)
    assert rec.ood_rejected and rec.predicted_class is None


def test_context_escalation_for_acute_symptoms():
    base = ENGINE.evaluate(_mr("DME", 0.9), CaseInput(image_path="x.png"))
    escalated = ENGINE.evaluate(
        _mr("DME", 0.9),
        CaseInput(image_path="x.png", symptoms=["sudden vision loss"]),
    )
    order = [Urgency.none, Urgency.routine, Urgency.soon, Urgency.urgent, Urgency.emergent]
    assert order.index(escalated.urgency) >= order.index(base.urgency)


def test_rules_version_is_stamped():
    rec = ENGINE.evaluate(_mr("Normal", 0.99), CASE)
    assert rec.rules_version == load_rules()["version"]
