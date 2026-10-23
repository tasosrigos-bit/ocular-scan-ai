"""Retrieval: tokenising, ranking, and the reciprocal rank fusion behind hybrid search."""
from __future__ import annotations

import numpy as np
import pytest

from ocular.rag import retrieve
from ocular.rag.index import tokenize


def test_tokenize_lowercases_and_drops_punctuation():
    assert tokenize("Soft DRUSEN, (RPE)-level!") == ["soft", "drusen", "rpe", "level"]


def test_top_indices_returns_the_best_first_and_handles_edge_sizes():
    scores = np.array([0.1, 0.9, 0.5, 0.7])
    assert retrieve._top_indices(scores, 3) == [1, 3, 2]
    assert retrieve._top_indices(scores, 10) == [1, 3, 2, 0]
    assert retrieve._top_indices(scores, 0) == []


def test_vector_search_ranks_by_cosine(tiny_index):
    hits = retrieve.vector_search(tiny_index, "drusen", k=3)
    assert [h.chunk.title for h in hits] == ["Article A", "Article B", "Article C"]
    assert [h.rank for h in hits] == [1, 2, 3]
    assert hits[0].score == pytest.approx(1.0)


def test_bm25_search_puts_the_matching_chunks_first(tiny_index):
    hits = retrieve.bm25_search(tiny_index, "drusen deposits", k=2)
    assert [h.chunk.title for h in hits] == ["Article A", "Article B"]
    assert hits[0].score > hits[1].score > 0


def test_hybrid_search_fuses_the_two_rankings_by_reciprocal_rank(tiny_index):
    hits = retrieve.hybrid_search(tiny_index, "drusen deposits", k=3, rrf_k=60)
    # A is first and B second in both rankings, so their fused scores are exact.
    assert [h.chunk.title for h in hits[:2]] == ["Article A", "Article B"]
    assert hits[0].score == pytest.approx(2 / 61)
    assert hits[1].score == pytest.approx(2 / 62)
    assert len(hits) == 3
    assert all(a.score >= b.score for a, b in zip(hits, hits[1:]))


def test_retrieve_dispatches_by_mode_and_rejects_unknown_ones(tiny_index):
    assert retrieve.retrieve(tiny_index, "drusen deposits", k=1, mode="bm25")[0].chunk.title == "Article A"
    assert retrieve.retrieve(tiny_index, "drusen", k=1, mode="vector")[0].chunk.title == "Article A"
    with pytest.raises(ValueError, match="unknown mode"):
        retrieve.retrieve(tiny_index, "drusen", mode="dense")
