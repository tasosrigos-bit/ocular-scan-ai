"""EmbeddingGemma's question and passage labels, and when they are applied."""
from __future__ import annotations

import numpy as np
import pytest
from qdrant_client import QdrantClient

from ocular.rag import index, qdrant_store
from ocular.rag.chunk import Chunk

GEMMA = "google/embeddinggemma-300m"
MEDEMBED = "abhinand/MedEmbed-base-v0.1"
CHUNK = Chunk("PMC1", "Anti-VEGF in DME", None, "CC BY", "Results", "Injections improved acuity.", 0)


def test_only_gemma_gets_labels():
    assert index.passage_text(GEMMA, CHUNK) == "title: Anti-VEGF in DME | text: Injections improved acuity."
    assert index.passage_text(MEDEMBED, CHUNK) == "Injections improved acuity."
    assert index.query_text(GEMMA, "How is DME treated?", labelled=True) == (
        "task: search result | query: How is DME treated?"
    )
    assert index.query_text(MEDEMBED, "How is DME treated?", labelled=True) == "How is DME treated?"


def test_an_untitled_passage_uses_none():
    untitled = Chunk("PMC2", "", None, "CC BY", "", "Text.", 0)
    assert index.passage_text(GEMMA, untitled) == "title: none | text: Text."


@pytest.fixture
def seen(monkeypatch):
    texts = []

    def load_embedder(model_name, device=None):
        return lambda batch: texts.extend(batch) or np.ones((len(batch), 2), np.float32)

    monkeypatch.setattr(index, "load_embedder", load_embedder)
    monkeypatch.setattr(qdrant_store, "load_embedder", load_embedder)
    return texts


@pytest.mark.parametrize("labelled, expected", [
    (True, "task: search result | query: drusen"),
    (False, "drusen"),  # an index built before labels: queried exactly as it was built
])
def test_queries_are_labelled_only_when_the_index_was(seen, labelled, expected):
    idx = index.Index([CHUNK], np.ones((1, 2), np.float32), None, GEMMA, {"input_labels": labelled})
    idx.embed_query("drusen")
    assert seen == [expected]


def test_streaming_embeds_labelled_text_but_stores_the_plain_chunk(seen):
    client = QdrantClient(":memory:")
    meta = index.make_manifest(GEMMA, 2, 1, {"input_labels": True})
    embed = index.load_embedder(GEMMA)
    qdrant_store.stream_collection(
        client, "g", [CHUNK], embed, meta, as_text=lambda c: index.passage_text(GEMMA, c)
    )
    assert seen == ["title: Anti-VEGF in DME | text: Injections improved acuity."]
    point = client.retrieve("g", [0], with_payload=True)[0]
    assert point.payload["text"] == "Injections improved acuity."

    seen.clear()
    qdrant_store.QdrantStore(client, "g").embed_query("drusen")
    assert seen == ["task: search result | query: drusen"]
