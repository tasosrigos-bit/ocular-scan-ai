"""Qdrant as the vector store, holding the dense and the lexical side of the index.

The in-memory :class:`~ocular.rag.index.Index` keeps the embeddings in a NumPy
matrix and rebuilds BM25 on every start. This module moves both into one Qdrant
collection, so the index lives in a database server that the application queries
rather than loads.

Each chunk becomes one point with two named vectors:

- ``dense`` - the chunk's normalised embedding, searched by cosine similarity.
- ``bm25`` - a sparse vector of the chunk's terms, weighted by the BM25 term
  frequency part ``tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / avgdl))``. The
  collection applies the IDF part itself (``Modifier.IDF``), so a query scored
  against it gets the full BM25 sum over the query terms.

Hybrid search asks Qdrant for both rankings and fuses them with the same
:func:`~ocular.rag.retrieve.rrf` the in-memory index uses, so the two stores differ
only in where the rankings come from. Qdrant's IDF is ``ln(1 + (N - n + 0.5) / (n +
0.5))``, which stays positive where the ``rank_bm25`` variant can go negative, so
the lexical ranking is close to the in-memory one rather than identical.
``experiments/compare_vector_stores.py`` measures how close.

Terms map to sparse indices by CRC32 hashing, so no vocabulary has to be stored and
a query is encoded without a lookup. With a vocabulary of tens of thousands of terms
in a 32-bit space, a collision is rare and costs at most a slightly merged weight.
"""
from __future__ import annotations

import zlib
from collections import Counter
from dataclasses import asdict

import numpy as np

from ocular.rag.chunk import Chunk
from ocular.rag.index import Index, load_embedder, query_text, tokenize
from ocular.rag.retrieve import MODES, Hit, rrf

# The BM25Okapi defaults of rank_bm25, which the in-memory index uses.
K1 = 1.5
B = 0.75


def term_id(token: str) -> int:
    """Map a term to its sparse vector index."""
    return zlib.crc32(token.encode("utf-8"))


def doc_vector(tokens: list[str], avgdl: float, k1: float = K1, b: float = B):
    """The BM25 term-frequency weights of one chunk, as a Qdrant sparse vector."""
    from qdrant_client import models

    weights: dict[int, float] = {}
    dl = len(tokens)
    for token, tf in Counter(tokens).items():
        weight = tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / avgdl))
        idx = term_id(token)
        weights[idx] = weights.get(idx, 0.0) + weight
    return models.SparseVector(indices=list(weights), values=list(weights.values()))


def query_vector(tokens: list[str]):
    """A query as a sparse vector, each term weighted by how often it occurs.

    A repeated query term counts once per occurrence, as in ``rank_bm25``.
    """
    from qdrant_client import models

    weights: dict[int, float] = {}
    for token, count in Counter(tokens).items():
        idx = term_id(token)
        weights[idx] = weights.get(idx, 0.0) + float(count)
    return models.SparseVector(indices=list(weights), values=list(weights.values()))


def _create(client, collection: str, dim: int, meta: dict) -> None:
    """Create a collection with the dense and the BM25 vectors and the manifest.

    The dense vectors and the payloads are kept on disk and memory-mapped, so
    several collections of a few hundred thousand passages fit alongside the
    embedding model on a 16 GB machine; the operating system caches what is read.
    """
    from qdrant_client import models

    client.create_collection(
        collection,
        vectors_config={
            "dense": models.VectorParams(size=dim, distance=models.Distance.COSINE, on_disk=True)
        },
        sparse_vectors_config={"bm25": models.SparseVectorParams(modifier=models.Modifier.IDF)},
        on_disk_payload=True,
        metadata={**meta, "bm25": {"k1": K1, "b": B}},
    )


def _avgdl(chunks: list[Chunk]) -> float:
    """The mean passage length in tokens, the one corpus statistic BM25 needs up front.

    Counted passage by passage rather than by keeping every token list, which for a
    few hundred thousand passages would hold gigabytes of small strings in memory.
    """
    return sum(len(tokenize(c.text)) for c in chunks) / len(chunks)


def _upsert(client, collection: str, rows, chunks, embeddings, avgdl: float) -> None:
    """Write one batch of points, ``embeddings`` aligned with ``rows``."""
    from qdrant_client import models

    client.upsert(
        collection,
        points=[
            models.PointStruct(
                id=i,
                vector={"dense": vec.tolist(), "bm25": doc_vector(tokenize(chunks[i].text), avgdl)},
                payload=asdict(chunks[i]),
            )
            for i, vec in zip(rows, embeddings)
        ],
        wait=True,
    )


def write_collection(
    client,
    collection: str,
    chunks: list[Chunk],
    embeddings: np.ndarray,
    meta: dict,
    batch: int = 256,
) -> int:
    """Write chunks and their embeddings into a fresh Qdrant collection.

    The point id of each chunk is its position in ``chunks``, and its payload is the
    chunk itself, so a hit carries everything needed to cite it. The manifest is
    stored as the collection's metadata, so a reader knows which model embeds the
    queries.

    Parameters
    ----------
    client : qdrant_client.QdrantClient
        A connected client.
    collection : str
        Collection name. An existing collection of that name is replaced.
    chunks : list of Chunk
        The passages, row-aligned with ``embeddings``.
    embeddings : numpy.ndarray
        Normalised chunk embeddings, shape ``(n_chunks, dim)``.
    meta : dict
        The index manifest. It must name the embedding model as ``model_name``.
    batch : int, optional
        Points per upsert request.

    Returns
    -------
    int
        The number of points written.
    """
    if client.collection_exists(collection):
        client.delete_collection(collection)
    _create(client, collection, int(embeddings.shape[1]), meta)
    avgdl = _avgdl(chunks)
    for start in range(0, len(chunks), batch):
        rows = range(start, min(start + batch, len(chunks)))
        _upsert(client, collection, rows, chunks, embeddings[start:start + batch], avgdl)
    return len(chunks)


def stream_collection(
    client,
    collection: str,
    chunks: list[Chunk],
    embed,
    meta: dict,
    batch: int = 256,
    progress=None,
    as_text=None,
) -> int:
    """Embed chunks batch by batch straight into Qdrant, resuming a partial build.

    For a corpus too large to embed in one sitting. Each batch is embedded and
    written before the next, so nothing is held in memory and nothing is written to
    disk, and every finished batch survives an interruption. Points are written in
    id order, so when the collection already exists for the same chunks and model,
    the build continues from the number of points it holds.

    Parameters
    ----------
    client : qdrant_client.QdrantClient
        A connected client.
    collection : str
        Collection name.
    chunks : list of Chunk
        The passages, in a fixed order that becomes their point ids.
    embed : callable
        Maps a list of texts to normalised embeddings, e.g. from
        :func:`~ocular.rag.index.load_embedder`.
    meta : dict
        The index manifest, naming the model as ``model_name`` and the dimensions as
        ``dim``. An existing collection whose manifest disagrees on the model or the
        number of chunks is not resumed but refused.
    batch : int, optional
        Chunks embedded and written per step.
    progress : callable, optional
        Called with the number of chunks already in the collection before the
        first batch, then with the number finished after each batch.
    as_text : callable, optional
        Maps a chunk to the text that is embedded, e.g.
        :func:`~ocular.rag.index.passage_text` for a model trained with passage
        labels. Defaults to the chunk's own text. The payload always keeps the
        plain chunk.

    Returns
    -------
    int
        The number of points in the collection when done.
    """
    start = 0
    if client.collection_exists(collection):
        have = client.get_collection(collection).config.metadata or {}
        if (have.get("model_name"), have.get("n_chunks")) != (meta["model_name"], len(chunks)):
            raise ValueError(
                f"collection {collection!r} holds a different index "
                f"({have.get('model_name')}, {have.get('n_chunks')} chunks); "
                "delete it or choose another name"
            )
        start = client.count(collection, exact=True).count
        print(f"resuming {collection!r} at chunk {start:,} of {len(chunks):,}", flush=True)
    else:
        _create(client, collection, meta["dim"], {**meta, "n_chunks": len(chunks)})

    avgdl = _avgdl(chunks)
    if progress:
        progress(start)
    for at in range(start, len(chunks), batch):
        rows = range(at, min(at + batch, len(chunks)))
        vectors = embed([(as_text or (lambda c: c.text))(chunks[i]) for i in rows])
        _upsert(client, collection, rows, chunks, vectors, avgdl)
        if progress:
            progress(rows[-1] + 1)
    return client.count(collection, exact=True).count


def build_collection(index: Index, client, collection: str, batch: int = 256) -> int:
    """Copy a built in-memory index into a Qdrant collection, without re-embedding."""
    meta = {**index.meta, "model_name": index.model_name}
    return write_collection(client, collection, index.chunks, index.embeddings, meta, batch)


class QdrantStore:
    """A searchable index backed by a Qdrant collection.

    It offers the same ``vector``, ``bm25`` and ``hybrid`` modes as the in-memory
    index through :meth:`search`, which :func:`ocular.rag.retrieve.retrieve` calls
    when it is handed a store rather than an :class:`~ocular.rag.index.Index`.

    Parameters
    ----------
    client : qdrant_client.QdrantClient
        A connected client.
    collection : str
        The collection written by :func:`build_collection`.
    """

    def __init__(self, client, collection: str):
        if not client.collection_exists(collection):
            raise FileNotFoundError(
                f"no Qdrant collection {collection!r}. Load the index into Qdrant with\n\n"
                f"    uv run python scripts/build_qdrant.py --index {collection}\n"
            )
        self.client = client
        self.collection = collection
        self.meta = client.get_collection(collection).config.metadata or {}
        self.model_name = self.meta["model_name"]

    @classmethod
    def connect(cls, url: str, collection: str) -> QdrantStore:
        """Connect to a Qdrant server and open one collection."""
        from qdrant_client import QdrantClient

        return cls(QdrantClient(url=url), collection)

    def embed_query(self, text: str) -> np.ndarray:
        """Encode a query with the model that embedded the collection."""
        text = query_text(self.model_name, text, self.meta.get("input_labels", False))
        return load_embedder(self.model_name)([text])[0]

    def _ranked(self, query, using: str, limit: int):
        return self.client.query_points(
            self.collection, query=query, using=using, limit=limit, with_payload=True
        ).points

    def search(
        self, query: str, k: int = 5, mode: str = "hybrid", candidates: int = 50, rrf_k: int = 60
    ) -> list[Hit]:
        """Rank chunks for a query.

        Parameters
        ----------
        query : str
            The query text.
        k : int, optional
            Number of hits to return.
        mode : {"vector", "bm25", "hybrid"}, optional
            Which ranking to use. Defaults to ``"hybrid"``.
        candidates : int, optional
            How many chunks each side contributes before fusion in ``"hybrid"``.
        rrf_k : int, optional
            The RRF smoothing constant.

        Returns
        -------
        list of Hit
            The retrieved chunks, best first. As in the in-memory index, a hit's
            score is the cosine, the BM25 score or the fused RRF score by mode.
        """
        if mode not in MODES:
            raise ValueError(f"unknown mode {mode!r}; choose from {MODES}")
        if mode == "vector":
            points = self._ranked(self.embed_query(query).tolist(), "dense", k)
        elif mode == "bm25":
            points = self._ranked(query_vector(tokenize(query)), "bm25", k)
        if mode != "hybrid":
            return [Hit(Chunk(**p.payload), float(p.score), r) for r, p in enumerate(points, 1)]

        dense = self._ranked(self.embed_query(query).tolist(), "dense", candidates)
        sparse = self._ranked(query_vector(tokenize(query)), "bm25", candidates)
        payloads = {p.id: p.payload for p in (*dense, *sparse)}
        fused = rrf([[p.id for p in dense], [p.id for p in sparse]], k, rrf_k)
        return [Hit(Chunk(**payloads[i]), score, r) for r, (i, score) in enumerate(fused, 1)]
