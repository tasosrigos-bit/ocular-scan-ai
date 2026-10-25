"""The Qdrant store, run against Qdrant's in-process mode so no server is needed."""
from __future__ import annotations

import pytest
from qdrant_client import QdrantClient

from ocular.rag import index, qdrant_store, retrieve
from tests.conftest import QUERY_VECTOR


@pytest.fixture
def store(tiny_index, monkeypatch):
    client = QdrantClient(":memory:")
    qdrant_store.build_collection(tiny_index, client, "tiny", batch=4)
    store = qdrant_store.QdrantStore(client, "tiny")
    monkeypatch.setattr(store, "embed_query", lambda text: QUERY_VECTOR)
    return store


def test_doc_vector_holds_the_bm25_term_frequency_weights():
    vec = qdrant_store.doc_vector(["drusen", "drusen", "rpe"], avgdl=3.0, k1=1.5, b=0.75)
    weights = dict(zip(vec.indices, vec.values))
    # At average length the weight is tf * (k1 + 1) / (tf + k1).
    assert weights[qdrant_store.term_id("drusen")] == pytest.approx(2 * 2.5 / 3.5)
    assert weights[qdrant_store.term_id("rpe")] == pytest.approx(1.0)


def test_query_vector_counts_repeated_terms():
    vec = qdrant_store.query_vector(["oct", "drusen", "oct"])
    assert dict(zip(vec.indices, vec.values))[qdrant_store.term_id("oct")] == 2.0


def test_the_collection_keeps_every_chunk_and_the_model_name(store, tiny_index):
    assert store.client.count("tiny").count == len(tiny_index.chunks)
    assert store.model_name == "fake"


def test_chunks_can_be_written_without_an_index_on_disk(tiny_index, monkeypatch):
    # The build script's --qdrant route: chunks and fresh embeddings, no files.
    client = QdrantClient(":memory:")
    meta = index.make_manifest("fake", 2, 6, {"chunk_strategy": "fixed"})
    n = qdrant_store.write_collection(client, "direct", tiny_index.chunks, tiny_index.embeddings, meta)
    store = qdrant_store.QdrantStore(client, "direct")
    monkeypatch.setattr(store, "embed_query", lambda text: QUERY_VECTOR)
    assert n == 6
    assert store.meta["chunk_strategy"] == "fixed" and store.meta["n_chunks"] == 6
    assert store.search("drusen deposits", k=1)[0].chunk.title == "Article A"


@pytest.mark.parametrize("mode", ["vector", "bm25", "hybrid"])
def test_each_mode_agrees_with_the_in_memory_index_on_the_top_hits(store, tiny_index, mode):
    query = "drusen deposits"
    in_memory = retrieve.retrieve(tiny_index, query, k=2, mode=mode)
    in_qdrant = retrieve.retrieve(store, query, k=2, mode=mode)  # dispatches to store.search
    assert [h.chunk for h in in_qdrant] == [h.chunk for h in in_memory]
    assert [h.rank for h in in_qdrant] == [1, 2]


def test_hybrid_scores_are_the_shared_rrf_fusion(store):
    hits = store.search("drusen deposits", k=2, mode="hybrid")
    assert hits[0].score == pytest.approx(2 / 61)
    assert hits[1].score == pytest.approx(2 / 62)


def test_unknown_mode_and_missing_collection_are_reported(store):
    with pytest.raises(ValueError, match="unknown mode"):
        store.search("drusen", mode="dense")
    with pytest.raises(FileNotFoundError, match="build_qdrant.py"):
        qdrant_store.QdrantStore(QdrantClient(":memory:"), "missing")


def test_streaming_resumes_after_an_interruption(tiny_index, monkeypatch):
    client = QdrantClient(":memory:")
    by_text = {c.text: tiny_index.embeddings[i] for i, c in enumerate(tiny_index.chunks)}
    meta = index.make_manifest("fake", 2, 6)
    calls = []

    def flaky(texts):
        calls.append(len(texts))
        if len(calls) == 2:
            raise RuntimeError("MPS stalled")
        return [by_text[t] for t in texts]

    with pytest.raises(RuntimeError, match="stalled"):
        qdrant_store.stream_collection(client, "full", tiny_index.chunks, flaky, meta, batch=2)
    assert client.count("full").count == 2  # the first batch survived

    seen = []
    n = qdrant_store.stream_collection(
        client, "full", tiny_index.chunks, lambda t: [by_text[x] for x in t], meta,
        batch=2, progress=seen.append,
    )
    assert n == 6 and seen == [2, 4, 6]  # resumed at chunk 2, not from the start
    store = qdrant_store.QdrantStore(client, "full")
    monkeypatch.setattr(store, "embed_query", lambda text: QUERY_VECTOR)
    assert [h.chunk.title for h in store.search("drusen deposits", k=2)] == ["Article A", "Article B"]


def test_streaming_refuses_to_resume_a_different_index(tiny_index):
    client = QdrantClient(":memory:")
    embed = lambda texts: tiny_index.embeddings[: len(texts)]  # noqa: E731
    qdrant_store.stream_collection(client, "full", tiny_index.chunks[:2], embed, index.make_manifest("fake", 2, 2))
    with pytest.raises(ValueError, match="different index"):
        qdrant_store.stream_collection(client, "full", tiny_index.chunks, embed, index.make_manifest("other", 2, 6))
