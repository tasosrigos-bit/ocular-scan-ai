"""Chunking: the three strategies cut articles into citable passages."""
from __future__ import annotations

import pytest

from ocular.rag.chunk import _split_sentences, chunk_document
from ocular.rag.corpus import Document


def _doc(sections):
    return Document("PMC1", "A title", "10.1/x", "CC BY", "A journal", sections)


def _sentences(n, words=10, tag="S"):
    # Each sentence opens with a capital tag such as "S3", since the splitter only
    # breaks where a capital follows the full stop.
    return " ".join(f"{tag}{i} " + " ".join(["word"] * (words - 2)) + " end." for i in range(n))


def test_split_sentences_breaks_on_end_punctuation_before_a_capital():
    text = "Drusen are deposits. They lie under the RPE! Are they common? (Yes.) e.g. often."
    assert _split_sentences(text) == [
        "Drusen are deposits.",
        "They lie under the RPE!",
        "Are they common?",
        "(Yes.) e.g. often.",
    ]


def test_fixed_chunks_respect_the_size_and_carry_an_overlap():
    doc = _doc([("Intro", _sentences(12))])
    chunks = chunk_document(doc, "fixed", size=40, overlap=10)
    assert len(chunks) > 1
    assert all(len(c.text.split()) <= 40 for c in chunks)
    # The last sentence of one chunk opens the next, so context is not cut off.
    for prev, nxt in zip(chunks, chunks[1:]):
        last_sentence = prev.text.split(". ")[-1]
        assert nxt.text.startswith(last_sentence.rstrip("."))


def test_section_chunks_never_span_two_sections():
    doc = _doc([("Methods", _sentences(3, tag="M")), ("Results", _sentences(3, tag="R"))])
    chunks = chunk_document(doc, "section", max_size=300)
    assert [c.section for c in chunks] == ["Methods", "Results"]
    assert "R0" not in chunks[0].text and "M0" not in chunks[1].text


def test_section_chunks_split_an_oversized_section():
    doc = _doc([("Discussion", _sentences(10))])
    chunks = chunk_document(doc, "section", max_size=30)
    assert len(chunks) > 1
    assert all(c.section == "Discussion" for c in chunks)
    assert all(len(c.text.split()) <= 30 for c in chunks)


def test_chunks_carry_the_source_metadata_in_order():
    doc = _doc([("Intro", _sentences(6))])
    chunks = chunk_document(doc, "fixed", size=20, overlap=0)
    assert [c.index for c in chunks] == list(range(len(chunks)))
    assert {(c.doc_id, c.title, c.doi, c.license) for c in chunks} == {
        ("PMC1", "A title", "10.1/x", "CC BY")
    }


def test_unknown_strategy_and_semantic_without_embedder_are_rejected():
    doc = _doc([("Intro", "One sentence.")])
    with pytest.raises(ValueError, match="unknown strategy"):
        chunk_document(doc, "paragraph")
    with pytest.raises(ValueError, match="embed_fn"):
        chunk_document(doc, "semantic")
