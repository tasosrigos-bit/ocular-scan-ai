"""End-to-end narrator with the StubBackend — no model download, no index."""

from oct_cds.cds.report import build_report
from oct_cds.cds.rules import CDSRuleEngine
from oct_cds.cds.schema import CaseInput, ModelResult
from oct_cds.rag.cache import NarrativeCache
from oct_cds.rag.llm import StubBackend
from oct_cds.rag.narrator import Narrator

ENGINE = CDSRuleEngine()
CASE = CaseInput(image_path="scan.png", eye="OD")


def _rec(top, p=0.9):
    rest = {k: (1 - p) / 7 for k in
            ["AMD", "CNV", "CSR", "DME", "DR", "Drusen", "Macular Hole", "Normal"] if k != top}
    return ENGINE.evaluate(ModelResult(probs={top: p, **rest}, ood_score=0.1,
                                       model_version="m1"), CASE)


def _narrator(tmp_path):
    return Narrator(backend=StubBackend(),
                    cache=NarrativeCache(tmp_path / "cache", enabled=True))


def test_verified_narrative_with_citation(tmp_path):
    res = _narrator(tmp_path).narrate(_rec("CNV"), CASE)
    assert res.rag_used
    gn = res.narrative
    assert gn.verified and not gn.fallback_used
    assert gn.citations and gn.citations[0]["id"] in gn.retrieved_ids
    assert gn.retrieved_ids  # decision-aware retrieval ran
    assert "[" in gn.text and "]" in gn.text


def test_normal_skips_rag(tmp_path):
    res = _narrator(tmp_path).narrate(_rec("Normal", p=0.99), CASE)
    assert not res.rag_used and "Normal" in res.reason


def test_abstain_skips_rag(tmp_path):
    mr = ModelResult(probs={"CNV": 0.45, "DME": 0.43, "AMD": 0.03, "CSR": 0.03,
                            "DR": 0.02, "Drusen": 0.02, "Macular Hole": 0.01, "Normal": 0.01},
                     ood_score=0.1, model_version="m1")
    rec = ENGINE.evaluate(mr, CASE)
    assert rec.abstained
    res = _narrator(tmp_path).narrate(rec, CASE)
    assert not res.rag_used


def test_cache_makes_it_reproducible(tmp_path):
    n = _narrator(tmp_path)
    a = n.narrate(_rec("DME"), CASE).narrative
    b = n.narrate(_rec("DME"), CASE).narrative
    assert a.text == b.text and a.retrieved_ids == b.retrieved_ids


class _Flakey:
    """Uncited on the first call, cited on the second (mimics a small model that
    needs the retry nudge)."""
    name = "flakey"

    def __init__(self):
        self.calls = 0

    def generate(self, system, user, *, max_tokens, temperature):
        self.calls += 1
        import re
        pid = re.findall(r"\[([a-z0-9][a-z0-9#\-]*)\]", user)[0]
        if self.calls == 1:
            return "The OCT is consistent with the predicted finding, a chronic process."
        return f"The OCT is consistent with the predicted finding [{pid}]."


def test_retry_recovers_a_missing_citation(tmp_path):
    be = _Flakey()
    n = Narrator(backend=be, cache=NarrativeCache(tmp_path / "c"), retry_uncited=1)
    res = n.narrate(_rec("CNV"), CASE)
    assert be.calls == 2
    assert res.narrative.verified and not res.narrative.fallback_used
    assert "took 2 attempts" in res.narrative.flags


def test_no_retry_when_disabled(tmp_path):
    be = _Flakey()
    n = Narrator(backend=be, cache=NarrativeCache(tmp_path / "c"), retry_uncited=0)
    res = n.narrate(_rec("CNV"), CASE)
    assert be.calls == 1 and res.narrative.fallback_used


def test_build_report_attaches_rag_fields(tmp_path):
    probs = {k: (0.9 if k == "Macular Hole" else 0.1 / 7) for k in
             ["AMD", "CNV", "CSR", "DME", "DR", "Drusen", "Macular Hole", "Normal"]}
    mr = ModelResult(probs=probs, ood_score=0.1, model_version="m1")
    rec = ENGINE.evaluate(mr, CASE)

    rep = build_report(rec, mr, CASE, narrator=_narrator(tmp_path))

    assert rep["impression"]["predicted_class"] == "Macular Hole"      # verbatim from rec
    assert rep["triage"]["urgency"] == rec.urgency.value               # verbatim from rec
    assert rep["narrator_meta"]["rag_used"] and rep["narrator_meta"]["verified"]
    assert "narrative_rag" in rep and rep["citations"]
    # Part 1 templated narrative is still present and unchanged
    assert rep["narrative"].startswith("AI CLINICAL DECISION SUPPORT")


def test_build_report_without_narrator_unchanged(tmp_path):
    probs = {k: (0.95 if k == "Normal" else 0.05 / 7) for k in
             ["AMD", "CNV", "CSR", "DME", "DR", "Drusen", "Macular Hole", "Normal"]}
    mr = ModelResult(probs=probs, ood_score=0.1, model_version="m1")
    rec = ENGINE.evaluate(mr, CASE)
    rep = build_report(rec, mr, CASE)          # no narrator
    assert "narrator_meta" not in rep and "narrative_rag" not in rep
