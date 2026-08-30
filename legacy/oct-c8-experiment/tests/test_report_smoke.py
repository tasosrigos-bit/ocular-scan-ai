from oct_cds.cds.report import build_report
from oct_cds.cds.rules import CDSRuleEngine
from oct_cds.cds.schema import CaseInput, ModelResult


def _mr(top="CNV", p=0.9):
    rest = {k: (1 - p) / 7 for k in
            ["AMD", "CNV", "CSR", "DME", "DR", "Drusen", "Macular Hole", "Normal"]
            if k != top}
    return ModelResult(probs={top: p, **rest}, ood_score=0.1, model_version="m1",
                       calibrator_version="t1", temperature=1.3)


def test_report_has_required_sections():
    mr = _mr()
    case = CaseInput(image_path="scan.png", eye="OD", acquisition_device="optopol_revo")
    rec = CDSRuleEngine().evaluate(mr, case)
    rep = build_report(rec, mr, case)

    for key in ("case", "impression", "differential", "triage", "provenance", "disclaimer"):
        assert key in rep
    assert rep["provenance"]["rules_version"] == rec.rules_version
    assert rep["provenance"]["temperature"] == 1.3
    assert rep["triage"]["urgency"] == "urgent"
    assert rep["disclaimer"]                       # non-empty medical disclaimer
    assert "AI CLINICAL DECISION SUPPORT" in rep["narrative"]


def test_abstain_report_has_no_predicted_class():
    mr = _mr("CNV", 0.5)
    case = CaseInput(image_path="scan.png")
    rec = CDSRuleEngine().evaluate(mr, case)
    rep = build_report(rec, mr, case)
    assert rep["impression"]["predicted_class"] is None
    assert rep["impression"]["abstained"] is True
