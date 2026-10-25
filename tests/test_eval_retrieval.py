"""Retrieval metrics against a question's known source passage."""
from __future__ import annotations

import math

import pytest

from ocular.rag import eval_retrieval as er
from ocular.rag.chunk import Chunk
from ocular.rag.retrieve import Hit

EVIDENCE = ("Drusen are extracellular deposits between the retinal pigment epithelium and "
            "Bruch membrane. Large soft drusen increase the risk of progression to late "
            "age-related macular degeneration. Optical coherence tomography shows them as "
            "dome-shaped elevations of the retinal pigment epithelium band.")


def _hit(doc, text, rank):
    return Hit(Chunk(doc, "Title", None, "CC BY", "Results", text, 0), 0.0, rank)


def test_a_chunk_inside_the_evidence_and_one_containing_it_both_count():
    inner = " ".join(EVIDENCE.split()[:25])
    outer = "Background on the macula. " + EVIDENCE + " Further unrelated discussion of cataract surgery."
    assert er.is_evidence(inner, EVIDENCE)
    assert er.is_evidence(outer, EVIDENCE)
    assert not er.is_evidence("Glaucoma damages the optic nerve through raised pressure.", EVIDENCE)


def test_article_and_evidence_levels_differ():
    hits = [
        _hit("PMC9", "Unrelated passage about keratoconus and corneal crosslinking.", 1),
        _hit("PMC1", "Another section of the source article, on methods and statistics used.", 2),
        _hit("PMC1", EVIDENCE, 3),
    ]
    s = er.score(hits, "PMC1", EVIDENCE, k=5)
    assert s["article_hit_at_1"] == 0.0 and s["article_hit_at_5"] == 1.0
    assert s["article_mrr"] == pytest.approx(1 / 2)
    assert s["evidence_mrr"] == pytest.approx(1 / 3)
    # Two source passages at ranks 2 and 3, against an ideal of ranks 1 and 2.
    ideal = 1 + 1 / math.log2(3)
    assert s["article_ndcg_at_5"] == pytest.approx((1 / math.log2(3) + 1 / math.log2(4)) / ideal)


def test_a_miss_scores_zero_and_only_the_top_k_count():
    hits = [_hit("PMC9", "Unrelated.", i) for i in range(1, 6)] + [_hit("PMC1", EVIDENCE, 6)]
    s = er.score(hits, "PMC1", EVIDENCE, k=5)
    assert all(v == 0.0 for v in s.values())


def test_evidence_from_another_article_does_not_count():
    s = er.score([_hit("PMC9", EVIDENCE, 1)], "PMC1", EVIDENCE, k=5)
    assert s["evidence_hit_at_5"] == 0.0 and s["article_hit_at_5"] == 0.0


def test_the_answer_level_counts_only_gold_chunks_and_is_absent_without_them():
    hits = [_hit("PMC1", "Other text from the source.", 1), _hit("PMC1", EVIDENCE, 2)]
    scored = er.score(hits, "PMC1", EVIDENCE, k=5, gold={("PMC1", EVIDENCE)})
    assert scored["answer_hit_at_1"] == 0.0
    assert scored["answer_hit_at_5"] == 1.0
    assert scored["answer_mrr"] == 0.5
    assert not any(m.startswith("answer_") for m in er.score(hits, "PMC1", EVIDENCE, k=5))
