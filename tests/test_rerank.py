"""Reranking orders by the cross-encoder's raw logits and reports 0-1 relevance."""
from __future__ import annotations

import math

import numpy as np
import pytest

from ocular.rag import rerank
from ocular.rag.chunk import Chunk
from ocular.rag.retrieve import Hit


def _hits(n):
    return [Hit(Chunk(f"PMC{i}", "t", None, "CC BY", "", f"passage {i}", i), 0.0, i + 1) for i in range(n)]


def test_large_logits_still_rank_in_order(monkeypatch):
    # Logits as large as MedCPT's: a sigmoid rounds all three to 1.0.
    monkeypatch.setattr(rerank, "load_reranker", lambda *a, **k: lambda q, texts: np.array([30.0, 50.0, 40.0], np.float32))
    out = rerank.rerank("q", _hits(3), model_name="any")
    assert [h.chunk.doc_id for h in out] == ["PMC1", "PMC2", "PMC0"]
    assert [h.rank for h in out] == [1, 2, 3]


def test_scores_are_the_sigmoid_of_the_logits_and_k_is_respected(monkeypatch):
    monkeypatch.setattr(rerank, "load_reranker", lambda *a, **k: lambda q, texts: np.array([0.0, 2.0, -1.0], np.float32))
    out = rerank.rerank("q", _hits(3), model_name="any", k=2)
    assert [h.chunk.doc_id for h in out] == ["PMC1", "PMC0"]
    assert out[0].score == pytest.approx(1 / (1 + math.exp(-2)))
    assert out[1].score == pytest.approx(0.5)


def test_no_hits_needs_no_model(monkeypatch):
    monkeypatch.setattr(rerank, "load_reranker", lambda *a, **k: pytest.fail("should not load"))
    assert rerank.rerank("q", []) == []
