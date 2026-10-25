"""The search-only agent's loop, with the searches and the model stubbed."""
from __future__ import annotations

import pytest

from ocular.rag import pipeline
from ocular.rag.chunk import Chunk
from ocular.rag.retrieve import Hit


def hit(name: str) -> Hit:
    return Hit(Chunk(doc_id=name, title=name, doi="", license="", section="", text=f"text of {name}", index=0), 0.0, 1)


@pytest.fixture
def stubs(monkeypatch):
    searched: list[str] = []
    results = {"q": ["a", "b"], "second": ["b", "c"], "third": ["d"]}

    def search_once(store, query, **kw):
        searched.append(query)
        return [hit(n) for n in results.get(query, ["z"])]

    def rerank(query, hits, model_name=None, k=None):
        # Reverse alphabetical, so the final order is visibly the reranker's.
        return sorted(hits, key=lambda h: h.chunk.doc_id, reverse=True)[:k]

    monkeypatch.setattr(pipeline, "search_once", search_once)
    monkeypatch.setattr(pipeline.rerank, "rerank", rerank)
    return searched


def decisions(monkeypatch, *steps):
    replies = iter(pipeline.SearchDecision(reason="r", sufficient=s, next_query=n) for s, n in steps)
    monkeypatch.setattr(pipeline.llm, "generate_structured", lambda *a, **kw: next(replies))


def test_the_first_search_is_the_question_and_the_pool_is_deduplicated(stubs, monkeypatch):
    decisions(monkeypatch, (False, "second"), (True, ""))
    out = pipeline.agentic_search("q", store=None, k=2)
    assert stubs == ["q", "second"] and out.queries == ["q", "second"]
    assert [h.chunk.doc_id for h in out.pool] == ["a", "b", "c"]
    assert [h.chunk.doc_id for h in out.top] == ["c", "b"]
    assert len(out.reasons) == 2


def test_the_agent_stops_when_the_passages_suffice(stubs, monkeypatch):
    decisions(monkeypatch, (True, "ignored"))
    out = pipeline.agentic_search("q", store=None)
    assert out.queries == ["q"]


def test_a_repeated_query_ends_the_search(stubs, monkeypatch):
    decisions(monkeypatch, (False, "second"), (False, " Q "))
    out = pipeline.agentic_search("q", store=None)
    assert out.queries == ["q", "second"]


def test_searches_are_capped(stubs, monkeypatch):
    decisions(monkeypatch, *[(False, f"query {i}") for i in range(10)])
    out = pipeline.agentic_search("q", store=None, max_searches=5)
    assert len(out.queries) == 5 and len(out.reasons) == 4
