"""The literature search, the one tool the assistant calls."""
from __future__ import annotations

from ocular.rag import tools


def test_search_corpus_numbers_the_passages_and_collects_them(tiny_index):
    collected = []
    search = tools.make_search_corpus(tiny_index, k=2, rerank_model=None, collector=collected)
    out = search.invoke({"query": "drusen deposits"})
    assert out.startswith("[1] (Article A - Results)\ndrusen deposits drusen deposits")
    assert "\n\n[2] (Article B - Results)\n" in out
    assert [h.chunk.title for h in collected] == ["Article A", "Article B"]


def test_format_says_so_when_nothing_was_found():
    assert tools.format_passages([]) == "No relevant passages found."
