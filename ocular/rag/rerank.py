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
retrieval experiment knobs.

The reranker is deliberately independent of the embedding model. A single strong
cross-encoder is applied to the candidates from any retriever, which isolates the
effect of reranking rather than entangling it with the choice of embedder.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from ocular.rag.retrieve import Hit

# A function that scores a query against a list of candidate texts, as raw logits.
RerankFn = Callable[[str, list[str]], np.ndarray]

# Loaded rerankers, keyed by (model name, device), instantiated once per process.
_RERANKERS: dict[tuple[str, str | None], RerankFn] = {}

# The default cross-encoder. A small (149M) ModernBERT reranker that, in the
# retrieval experiments on the 508-article corpus, lifted context precision by ~+0.06 over the ms-marco-MiniLM
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

        import torch

        model = CrossEncoder(model_name, device=device)
        # Some cross-encoders, ncbi/MedCPT-Cross-Encoder among them, set no input limit
        # in their tokenizer, so long passages would overrun the 512-token BERT
        # encoder. Those are capped at 512, while a model that states its own limit,
        # such as the 8,192-token gte reranker, keeps it.
        if not model.max_length or model.max_length > 100_000:
            model.max_length = 512

        def score(query: str, texts: list[str]) -> np.ndarray:
            pairs = [[query, text] for text in texts]
            # Raw logits, so candidates that a sigmoid would round to the same 0 or 1
            # still rank in order. :func:`rerank` converts them to 0-1 afterwards.
            return np.asarray(model.predict(pairs, activation_fn=torch.nn.Identity()), dtype=np.float32)

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
        the cross-encoder's relevance in 0-1 (the sigmoid of its logit) and its
        ``rank`` renumbered from one.

    Notes
    -----
    The hits are ordered by the raw logits and only then mapped to 0-1. A
    cross-encoder with large logits, such as MedCPT (about +7 for a relevant
    passage and -12 for an irrelevant one), would otherwise have several strong
    candidates rounded to exactly 1.0 and ranked arbitrarily among themselves.
    For a model with modest logits, such as the gte reranker, the order and the
    scores are the same as ranking on the 0-1 values directly.
    """
    if not hits:
        return []
    logits = load_reranker(model_name, device)(query, [h.chunk.text for h in hits])
    order = np.argsort(-logits, kind="stable")
    if k is not None:
        order = order[:k]
    relevance = 1 / (1 + np.exp(-logits.astype(np.float64)))
    return [Hit(hits[i].chunk, float(relevance[i]), rank) for rank, i in enumerate(order, 1)]
