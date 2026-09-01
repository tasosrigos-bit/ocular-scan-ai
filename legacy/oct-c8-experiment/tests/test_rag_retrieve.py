"""Decision-aware retrieval: the predicted class's entry is ALWAYS retrieved,
deterministically, with no vector index in the loop."""

from oct_cds.cds.rules import CDSRuleEngine
from oct_cds.cds.schema import CaseInput, ModelResult
from oct_cds.rag.ingest import load_knowledge_base
from oct_cds.rag.retrieve import retrieve_for

KB = load_knowledge_base()
ENGINE = CDSRuleEngine()
CASE = CaseInput(image_path="x.png")


def _rec(top, p=0.9):
    rest = {k: (1 - p) / 7 for k in
            ["AMD", "CNV", "CSR", "DME", "DR", "Drusen", "Macular Hole", "Normal"] if k != top}
    return ENGINE.evaluate(ModelResult(probs={top: p, **rest}, ood_score=0.1,
                                       model_version="t"), CASE)


def test_predicted_class_entry_always_retrieved():
    for cls, entry in [("Drusen", "amd"), ("CNV", "cnv"), ("DME", "dme"),
                       ("Macular Hole", "macular_hole"), ("CSR", "csr")]:
        rps = retrieve_for(_rec(cls), KB)
        ids = {rp.passage.id for rp in rps}
        assert any(i.startswith(f"{entry}#") for i in ids), f"{cls} -> {entry} missing"
        assert f"{entry}#overview" in ids


def test_deterministic_order_and_repeatable():
    a = [rp.passage.id for rp in retrieve_for(_rec("Drusen"), KB)]
    b = [rp.passage.id for rp in retrieve_for(_rec("Drusen"), KB)]
    assert a == b
    # predicted-class entry passages come before any 'semantic' ones
    whys = [rp.why for rp in retrieve_for(_rec("Drusen"), KB)]
    assert whys == sorted(whys, key=lambda w: {"predicted-class entry": 0,
                                               "model-behavior note": 1, "semantic": 2}[w])


def test_model_behavior_note_included_for_differential():
    # confident DME, Drusen as 2nd -> the Drusen (amd) entry's model-behavior note rides along
    mr = ModelResult(
        probs={"DME": 0.74, "Drusen": 0.18, "AMD": 0.02, "CNV": 0.02, "CSR": 0.01,
               "DR": 0.01, "Macular Hole": 0.01, "Normal": 0.01},
        ood_score=0.1, model_version="t",
    )
    rec = ENGINE.evaluate(mr, CASE)
    assert not rec.abstained and rec.predicted_class == "DME"
    ids = {rp.passage.id for rp in retrieve_for(rec, KB)}
    assert "dme#model-behavior-note" in ids
    assert "amd#model-behavior-note" in ids   # Drusen -> amd entry, via the differential
