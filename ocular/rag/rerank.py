"""Rerank retrieved candidates with a cross-encoder.

First-stage retrieval (see :mod:`ocular.rag.retrieve`) is fast because it compares
independently computed embeddings, but that independence limits its precision. A
**cross-encoder** reads the query and a candidate *together* and scores their
relevance directly, which is far more accurate but far too slow to run over the
whole corpus. The standard remedy, applied here, is to rerank only the top-N
candidates the retriever already found:

    retrieve top-N (wide)  ->  cross-encoder rescores the N  ->  keep top-k

Because only a handful of pairs are scored per query, the reranker is cheap even
on CPU, and it runs at query time rather than at index-build time, so it needs no
precomputed artifact. Whether to rerank, and with which model, is one of the
Phase-A experiment knobs.

The reranker is deliberately independent of the embedding model. A single strong
cross-encoder is applied to the candidates from any retriever, which isolates the
effect of reranking rather than entangling it with the choice of embedder.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from ocular.rag.retrieve import Hit

# A function that scores a query against a list of candidate texts.
RerankFn = Callable[[str, list[str]], np.ndarray]

# Loaded rerankers, keyed by (model name, device), instantiated once per process.
_RERANKERS: dict[tuple[str, str | None], RerankFn] = {}

# The default cross-encoder. A small (149M) ModernBERT reranker that, in the
# Phase-A sweep, lifted context precision by ~+0.06 over the ms-marco-MiniLM
# baseline at a modest latency cost (~+4 s/query), and far more cheaply than the
# heavy ``BAAI/bge-reranker-v2-m3`` (+22 s/query for a smaller gain). Plug-and-play
# via CrossEncoder. Swap for the retrieval knob when experimenting.
DEFAULT_RERANKER = "Alibaba-NLP/gte-reranker-modernbert-base"


def load_reranker(model_name: str = DEFAULT_RERANKER, device: str | None = None) -> RerankFn:
    """Load a cross-encoder and return a query-candidate scoring function.

    Parameters
    ----------
    model_name : str, optional
        A `sentence-transformers` CrossEncoder id. Defaults to
        :data:`DEFAULT_RERANKER`.
    device : str, optional
        Device to run on (``"cpu"``, ``"mps"``). ``None`` lets the library choose.
        Reranking scores only a few dozen pairs per query, so the device rarely
        matters.

    Returns
    -------
    RerankFn
        A callable ``(query, texts) -> np.ndarray`` of relevance scores.
    """
    key = (model_name, device)
    if key not in _RERANKERS:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is not installed; add it with "
                "uv add sentence-transformers."
            ) from exc

        model = CrossEncoder(model_name, device=device)

        def score(query: str, texts: list[str]) -> np.ndarray:
            pairs = [[query, text] for text in texts]
            return np.asarray(model.predict(pairs), dtype=np.float32)

        _RERANKERS[key] = score
    return _RERANKERS[key]


def rerank(
    query: str,
    hits: list[Hit],
    model_name: str = DEFAULT_RERANKER,
    k: int | None = None,
    device: str | None = None,
) -> list[Hit]:
    """Rerank retrieved hits by cross-encoder relevance, best first.

    Parameters
    ----------
    query : str
        The query the hits were retrieved for.
    hits : list of Hit
        The first-stage candidates, typically a wide ``retrieve(..., k=N)``.
    model_name : str, optional
        The cross-encoder to use. Defaults to :data:`DEFAULT_RERANKER`.
    k : int, optional
        Number of hits to keep. ``None`` keeps all, reordered.
    device : str, optional
        Device for the cross-encoder.

    Returns
    -------
    list of Hit
        The hits reordered by the cross-encoder, each with its ``score`` set to
        the cross-encoder score and its ``rank`` renumbered from one.
    """
    if not hits:
        return []
    scores = load_reranker(model_name, device)(query, [h.chunk.text for h in hits])
    order = np.argsort(-scores)
    if k is not None:
        order = order[:k]
    return [Hit(hits[i].chunk, float(scores[i]), rank) for rank, i in enumerate(order, 1)]
