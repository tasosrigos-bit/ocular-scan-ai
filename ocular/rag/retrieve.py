"""Retrieve the chunks most relevant to a query.

Given an :class:`~ocular.rag.index.Index`, this module ranks the corpus chunks
for a query in one of three ways, which is the retrieval knob compared in the
Phase-A experiments:

- **vector** - cosine similarity between the query embedding and each chunk
  embedding. Captures meaning, so a query and a chunk can match even when they
  share no words.
- **bm25** - a lexical score that rewards exact term overlap. Catches acronyms
  and precise wording (``CNV``, ``OCT``) that embeddings tend to blur.
- **hybrid** - the two rankings fused with Reciprocal Rank Fusion (RRF), which
  combines their complementary strengths.

RRF fuses on **ranks** rather than scores, so it needs no calibration between the
bounded cosine scale and the unbounded BM25 scale. For a chunk ``d`` appearing in
either ranked list,

    score(d) = sum over retrievers of  1 / (rrf_k + rank(d))

with ``rank`` counted from one and ``rrf_k`` a smoothing constant (60 by
convention). The fusion is computed over chunk row indices, since both retrievers
rank the same aligned ``index.chunks`` array.

Every function returns :class:`Hit` objects that carry the chunk and its metadata,
so a downstream reranker or answer step can cite the source. ``retrieve`` returns
a wide list (large ``k``) when a reranker follows, or a narrow one when the chunks
go straight to the language model.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ocular.rag.chunk import Chunk
from ocular.rag.index import Index, tokenize

# The retrieval modes exposed to experiments/run_rag_eval.py as a knob.
MODES = ("vector", "bm25", "hybrid")


@dataclass
class Hit:
    """One retrieved chunk with its ranking.

    Attributes
    ----------
    chunk : Chunk
        The retrieved chunk, with the metadata needed to cite it.
    score : float
        The ranking score. Its meaning depends on the mode: cosine similarity for
        ``vector``, a BM25 score for ``bm25``, and a fused RRF score for ``hybrid``.
    rank : int
        The chunk's one-based position in the returned list.
    """

    chunk: Chunk
    score: float
    rank: int


def _top_indices(scores: np.ndarray, n: int) -> list[int]:
    """Return the row indices of the ``n`` highest scores, best first."""
    n = min(n, len(scores))
    if n == 0:
        return []
    top = np.argpartition(-scores, n - 1)[:n]  # n highest, unordered
    return [int(i) for i in top[np.argsort(-scores[top])]]  # then sort those


def _vector_scores(index: Index, query: str) -> np.ndarray:
    """Cosine similarity of the query against every chunk embedding."""
    # Embeddings are stored L2-normalised, so a dot product is the cosine.
    return index.embeddings @ index.embed_query(query)


def _bm25_scores(index: Index, query: str) -> np.ndarray:
    """BM25 lexical score of the query against every chunk."""
    return np.asarray(index.bm25.get_scores(tokenize(query)), dtype=np.float32)


def _hits(index: Index, indices: list[int], scores: np.ndarray) -> list[Hit]:
    """Wrap ranked row indices as :class:`Hit` objects."""
    return [Hit(index.chunks[i], float(scores[i]), rank) for rank, i in enumerate(indices, 1)]


def vector_search(index: Index, query: str, k: int = 5) -> list[Hit]:
    """Rank chunks by embedding cosine similarity.

    Parameters
    ----------
    index : Index
        The searchable index.
    query : str
        The query text.
    k : int, optional
        Number of hits to return. Defaults to 5.

    Returns
    -------
    list of Hit
        The ``k`` most similar chunks, best first.
    """
    scores = _vector_scores(index, query)
    return _hits(index, _top_indices(scores, k), scores)


def bm25_search(index: Index, query: str, k: int = 5) -> list[Hit]:
    """Rank chunks by BM25 lexical score.

    Parameters
    ----------
    index : Index
        The searchable index.
    query : str
        The query text.
    k : int, optional
        Number of hits to return. Defaults to 5.

    Returns
    -------
    list of Hit
        The ``k`` highest-scoring chunks, best first.
    """
    scores = _bm25_scores(index, query)
    return _hits(index, _top_indices(scores, k), scores)


def hybrid_search(
    index: Index, query: str, k: int = 5, candidates: int = 50, rrf_k: int = 60
) -> list[Hit]:
    """Rank chunks by Reciprocal Rank Fusion of the vector and BM25 rankings.

    Each retriever contributes its top ``candidates`` chunks; a chunk's fused score
    is the sum of ``1 / (rrf_k + rank)`` over the retrievers that rank it. Working
    on ranks makes the fusion insensitive to the two retrievers' different score
    scales.

    Parameters
    ----------
    index : Index
        The searchable index.
    query : str
        The query text.
    k : int, optional
        Number of hits to return. Defaults to 5.
    candidates : int, optional
        How many chunks each retriever contributes before fusion. Defaults to 50.
    rrf_k : int, optional
        The RRF smoothing constant. Defaults to 60.

    Returns
    -------
    list of Hit
        The ``k`` chunks with the highest fused score, best first. Each hit's
        ``score`` is the RRF score.
    """
    vector_ranked = _top_indices(_vector_scores(index, query), candidates)
    bm25_ranked = _top_indices(_bm25_scores(index, query), candidates)

    fused: dict[int, float] = {}
    for ranked in (vector_ranked, bm25_ranked):
        for rank, i in enumerate(ranked, 1):
            fused[i] = fused.get(i, 0.0) + 1.0 / (rrf_k + rank)

    order = sorted(fused, key=fused.get, reverse=True)[:k]
    return [Hit(index.chunks[i], fused[i], rank) for rank, i in enumerate(order, 1)]


def retrieve(
    index: Index, query: str, k: int = 5, mode: str = "hybrid", candidates: int = 50
) -> list[Hit]:
    """Retrieve the top chunks for a query with the chosen mode.

    Parameters
    ----------
    index : Index
        The searchable index.
    query : str
        The query text.
    k : int, optional
        Number of hits to return. Use a wide ``k`` when a reranker follows, a
        narrow one when the chunks go straight to the model. Defaults to 5.
    mode : {"vector", "bm25", "hybrid"}, optional
        Which retriever to use. Defaults to ``"hybrid"``.
    candidates : int, optional
        Per-retriever pool size for ``"hybrid"``. Ignored otherwise. Defaults to 50.

    Returns
    -------
    list of Hit
        The retrieved chunks, best first.
    """
    if mode == "vector":
        return vector_search(index, query, k)
    if mode == "bm25":
        return bm25_search(index, query, k)
    if mode == "hybrid":
        return hybrid_search(index, query, k, candidates)
    raise ValueError(f"unknown mode {mode!r}; choose from {MODES}")
