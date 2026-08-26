"""Split parsed articles into retrieval chunks.

Retrieval works over passages, not whole articles, so each :class:`~ocular.rag.corpus.Document`
is broken into overlapping-or-not text chunks that carry their source metadata
(id, title, DOI, licence, section) for citation. *How* the split is made is one
of the experiment knobs (Phase A), so three strategies share one interface:

- ``"fixed"`` - pack sentences up to a target size with overlap, **ignoring**
  section headings. The simple baseline.
- ``"section"`` - respect the JATS section structure: never cross a heading, and
  split a section further only when it exceeds the size cap. Likely the strongest
  default here, since the corpus is already cleanly sectioned.
- ``"semantic"`` - split where the topic shifts, detected from a drop in cosine
  similarity between consecutive sentences (a percentile-based breakpoint). Needs
  an embedding function, supplied by the caller so this module stays decoupled
  from any particular embedder.

All three work over **sentences** as the atomic unit, so no chunk ever cuts a
sentence in half. Size is measured in words (a cheap proxy for tokens).
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from itertools import groupby
from pathlib import Path
from typing import Callable

import numpy as np

from ocular.rag.corpus import Document

# The strategies exposed to experiments/run_rag_eval.py as a knob.
STRATEGIES = ("fixed", "section", "semantic")

# A function that maps a list of texts to an (n, d) array of embeddings. Only the
# semantic strategy needs one; it is injected rather than imported so this module
# depends on no embedding library.
EmbedFn = Callable[[list[str]], np.ndarray]

# Rough sentence boundary: end punctuation followed by whitespace and an opening
# capital or bracket. Good enough for prose; a linguistic splitter (spaCy/nltk)
# could be dropped in later without touching the strategies.
_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\"'])")


@dataclass
class Chunk:
    """One retrieval passage with the metadata needed to cite it.

    Attributes
    ----------
    doc_id : str
        Source article's PubMed Central id.
    title : str
        Source article title.
    doi : str or None
        Source DOI, for the citation.
    license : str
        Source licence label.
    section : str
        Heading path the chunk came from (`` > ``-joined); empty for a ``fixed``
        chunk that spans headings, otherwise the first section it covers.
    text : str
        The passage text.
    index : int
        Position of the chunk within its document, from zero.
    """

    doc_id: str
    title: str
    doi: str | None
    license: str
    section: str
    text: str
    index: int


def _split_sentences(text: str) -> list[str]:
    """Split a paragraph block into trimmed, non-empty sentences."""
    return [s.strip() for s in _SENTENCE.split(text) if s.strip()]


def _wc(text: str) -> int:
    """Word count, the size proxy used throughout."""
    return len(text.split())


def _units(doc: Document) -> list[tuple[str, str]]:
    """Flatten a document into ``(section, sentence)`` units in reading order."""
    return [
        (section, sentence)
        for section, text in doc.sections
        for sentence in _split_sentences(text)
    ]


def _fixed(
    units: list[tuple[str, str]], size: int, overlap: int
) -> list[tuple[str, str]]:
    """Pack sentences to a target word size with overlap, ignoring sections.

    When a chunk is emitted, its trailing sentences (up to ``overlap`` words) are
    carried into the next chunk, so context straddling a boundary is not lost.
    """
    out: list[tuple[str, str]] = []
    cur: list[tuple[str, str]] = []
    words = 0
    for unit in units:
        w = _wc(unit[1])
        if cur and words + w > size:
            out.append((cur[0][0], " ".join(u[1] for u in cur)))
            tail: list[tuple[str, str]] = []
            tw = 0
            for prev in reversed(cur):  # keep the last `overlap` words as context
                if tw >= overlap:
                    break
                tail.insert(0, prev)
                tw += _wc(prev[1])
            cur = tail
            words = sum(_wc(u[1]) for u in cur)
        cur.append(unit)
        words += w
    if cur:
        out.append((cur[0][0], " ".join(u[1] for u in cur)))
    return out


def _section(
    units: list[tuple[str, str]], max_size: int
) -> list[tuple[str, str]]:
    """Chunk within section boundaries, splitting only oversized sections."""
    out: list[tuple[str, str]] = []
    for section, grp in groupby(units, key=lambda u: u[0]):
        cur: list[str] = []
        words = 0
        for _, sentence in grp:
            w = _wc(sentence)
            if cur and words + w > max_size:
                out.append((section, " ".join(cur)))
                cur, words = [], 0
            cur.append(sentence)
            words += w
        if cur:
            out.append((section, " ".join(cur)))
    return out


def _semantic(
    units: list[tuple[str, str]],
    embed_fn: EmbedFn,
    threshold: float,
    max_size: int,
) -> list[tuple[str, str]]:
    """Split at topic shifts: percentile breakpoints in consecutive-sentence distance.

    Each sentence is embedded; the cosine distance between neighbours is a
    topic-shift signal. A new chunk starts wherever that distance reaches the
    ``threshold``-th percentile of all distances in the document, or the running
    chunk would exceed ``max_size`` words. The section label is the first section
    a chunk covers.
    """
    if len(units) <= 1:
        return [(units[0][0], units[0][1])] if units else []

    emb = np.asarray(embed_fn([u[1] for u in units]), dtype=np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8
    distances = 1.0 - (emb[:-1] * emb[1:]).sum(axis=1)  # between i and i+1
    cut = np.percentile(distances, threshold)

    out: list[tuple[str, str]] = []
    cur = [units[0]]
    words = _wc(units[0][1])
    for i in range(1, len(units)):
        w = _wc(units[i][1])
        if cur and (distances[i - 1] >= cut or words + w > max_size):
            out.append((cur[0][0], " ".join(u[1] for u in cur)))
            cur, words = [], 0
        cur.append(units[i])
        words += w
    if cur:
        out.append((cur[0][0], " ".join(u[1] for u in cur)))
    return out


def chunk_document(
    doc: Document,
    strategy: str = "section",
    embed_fn: EmbedFn | None = None,
    size: int = 200,
    overlap: int = 40,
    max_size: int = 300,
    threshold: float = 90.0,
) -> list[Chunk]:
    """Chunk one document with the chosen strategy.

    Parameters
    ----------
    doc : Document
        The parsed article.
    strategy : {"fixed", "section", "semantic"}, optional
        Which splitter to use. Defaults to ``"section"``.
    embed_fn : callable, optional
        Required for ``"semantic"``; maps texts to an ``(n, d)`` embedding array.
    size, overlap : int, optional
        Target and overlap word counts for ``"fixed"``.
    max_size : int, optional
        Word cap per chunk for ``"section"`` and ``"semantic"``.
    threshold : float, optional
        Breakpoint percentile for ``"semantic"``.

    Returns
    -------
    list of Chunk
        The document's chunks, each carrying its source metadata.
    """
    units = _units(doc)
    if strategy == "fixed":
        pieces = _fixed(units, size, overlap)
    elif strategy == "section":
        pieces = _section(units, max_size)
    elif strategy == "semantic":
        if embed_fn is None:
            raise ValueError("semantic chunking requires an embed_fn")
        pieces = _semantic(units, embed_fn, threshold, max_size)
    else:
        raise ValueError(f"unknown strategy {strategy!r}; choose from {STRATEGIES}")

    return [
        Chunk(doc.pmcid, doc.title, doc.doi, doc.license, section, text, i)
        for i, (section, text) in enumerate(pieces)
    ]


def chunk_corpus(
    docs: list[Document], strategy: str = "section", embed_fn: EmbedFn | None = None, **params
) -> list[Chunk]:
    """Chunk a whole corpus, concatenating every document's chunks.

    Parameters
    ----------
    docs : list of Document
        The parsed corpus.
    strategy : {"fixed", "section", "semantic"}, optional
        The splitter to use.
    embed_fn : callable, optional
        Required for ``"semantic"``.
    **params
        Strategy parameters forwarded to :func:`chunk_document` (``size``,
        ``overlap``, ``max_size``, ``threshold``).

    Returns
    -------
    list of Chunk
        Every chunk from every document.
    """
    return [
        chunk
        for doc in docs
        for chunk in chunk_document(doc, strategy, embed_fn, **params)
    ]


def save_chunks(chunks: list[Chunk], path: Path) -> None:
    """Write chunks to a JSON Lines cache, one per line.

    The cache lets several embedders reuse the same chunks without re-chunking,
    which matters for the semantic strategy since its boundaries are independent
    of the retrieval embedder yet costly to recompute.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for chunk in chunks:
            fh.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")


def load_chunks(path: Path) -> list[Chunk]:
    """Read chunks written by :func:`save_chunks`."""
    with path.open(encoding="utf-8") as fh:
        return [Chunk(**json.loads(line)) for line in fh]
