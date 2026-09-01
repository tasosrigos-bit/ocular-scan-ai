from oct_cds.cds.rules import CDSRuleEngine
from oct_cds.cds.schema import CaseInput, ModelResult
from oct_cds.rag.verify import verify_narrative

ENGINE = CDSRuleEngine()
CASE = CaseInput(image_path="x.png")
RETRIEVED = {"cnv#overview", "cnv#referral", "cnv#model-behavior-note"}


def _rec(top, p=0.92):
    rest = {k: (1 - p) / 7 for k in
            ["AMD", "CNV", "CSR", "DME", "DR", "Drusen", "Macular Hole", "Normal"] if k != top}
    return ENGINE.evaluate(ModelResult(probs={top: p, **rest}, ood_score=0.1,
                                       model_version="t"), CASE)


def test_clean_narrative_passes():
    rec = _rec("CNV")  # urgency: urgent
    txt = ("The OCT features are consistent with CNV [cnv#overview]. Neovascular "
           "membranes leak fluid and can cause rapid central vision loss "
           "[cnv#overview].")
    r = verify_narrative(txt, rec, RETRIEVED)
    assert r.ok and not r.flags


def test_unresolved_citation_hard_fails():
    rec = _rec("CNV")
    r = verify_narrative("Consistent with CNV [made#up].", rec, RETRIEVED)
    assert not r.ok and any("not in retrieved set" in f for f in r.flags)


def test_no_citation_hard_fails():
    rec = _rec("CNV")
    r = verify_narrative("Consistent with CNV, a neovascular process.", rec, RETRIEVED)
    assert not r.ok and "no citations" in r.flags


def test_asserting_a_different_class_hard_fails():
    rec = _rec("CNV")
    r = verify_narrative("The findings show DME [cnv#overview].", rec, RETRIEVED)
    assert not r.ok and any("different finding (DME)" in f for f in r.flags)


def test_triage_downgrade_flagged_when_decision_is_urgent():
    rec = _rec("CNV")  # urgent
    txt = "Consistent with CNV [cnv#overview]. No referral is needed for this patient."
    strict = verify_narrative(txt, rec, RETRIEVED, strict=True)
    assert not strict.ok
    lax = verify_narrative(txt, rec, RETRIEVED, strict=False)
    assert lax.ok and any("downgrades triage" in f for f in lax.flags)


def test_general_referral_language_from_kb_is_ok():
    rec = _rec("CNV")
    txt = ("Consistent with CNV [cnv#referral]. CNV generally warrants urgent "
           "rather than routine referral [cnv#referral].")
    assert verify_narrative(txt, rec, RETRIEVED).ok
