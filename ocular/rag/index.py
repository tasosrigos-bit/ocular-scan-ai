"""Embed and index chunks for retrieval.

An index turns a list of :class:`~ocular.rag.chunk.Chunk` into something a query
can search. It holds two representations, so ``retrieve.py`` can run dense,
sparse, or hybrid retrieval over the same corpus:

- **dense** - a normalised embedding per chunk, for cosine (dot-product) search.
- **sparse** - a BM25 lexical index, which catches exact terms and acronyms
  (``CNV``, ``DME``) that embeddings can blur.

The embedding model is **not fixed here**. Generic vs biomedical embeddings is one
of the retrieval experiments, so ``model_name`` is a required argument with no
default and any `sentence-transformers` model works. Its choice - and therefore
the `sentence-transformers` dependency - is deliberately deferred until the
experiment decides; importing this module needs neither, because the embedding
and BM25 backends are imported lazily, only when a function that needs them runs.

Each index writes a ``manifest.json`` recording the model and the config that
produced it, so a sweep's artifacts stay self-describing and reproducible.

The embedder returned by :func:`load_embedder` is also exactly the ``embed_fn``
the semantic chunker expects, so one loaded model serves both chunking and
indexing.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from ocular import config
from ocular.rag.chunk import Chunk, EmbedFn

# Root of the index tree; one subdirectory per experiment config lives beneath it.
INDEX_DIR = config.DATA_DIR / "index"

# Loaded embedders, keyed by (model name, device), so a model is instantiated
# once per process and reused across chunking, indexing and query encoding.
_EMBEDDERS: dict[tuple[str, str | None], EmbedFn] = {}

# Models whose custom architecture must be trusted to load from the Hub. The Jina
# encoders ship their own modelling code; the MiniLM and MedEmbed encoders are
# plain BERT and need no such trust.
_TRUST_REMOTE_CODE = {"jinaai/jina-embeddings-v5-text-nano"}


def load_embedder(model_name: str, device: str | None = None) -> EmbedFn:
    """Load a sentence-embedding model and return it as an ``embed_fn``.

    The returned callable maps a list of texts to an ``(n, d)`` array of
    L2-normalised float32 embeddings, so dense similarity is a plain dot product.
    Results are cached per model name for the life of the process.

    Parameters
    ----------
    model_name : str
        Any `sentence-transformers` model id, e.g. ``"all-MiniLM-L6-v2"`` or a
        biomedical model. The choice is an experiment knob, not fixed here.
    device : str, optional
        The device to run on (``"cpu"``, ``"mps"``, ``"cuda"``). ``None`` lets
        sentence-transformers pick. Forcing ``"cpu"`` is the reliable fallback for
        larger models, whose full-corpus encode can stall on Apple's MPS backend.

    Returns
    -------
    EmbedFn
        A ``list[str] -> np.ndarray`` embedding function.
    """
    key = (model_name, device)
    if key not in _EMBEDDERS:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # dep intentionally not added until a model is chosen
            raise ImportError(
                "sentence-transformers is not installed; add it once the "
                "embedding model is chosen (uv add --group rag sentence-transformers)."
            ) from exc

        model = SentenceTransformer(
            model_name, device=device, trust_remote_code=model_name in _TRUST_REMOTE_CODE
        )

        def embed(texts: list[str]) -> np.ndarray:
            vectors = model.encode(
                texts, normalize_embeddings=True, convert_to_numpy=True
            )
            if model.device.type == "mps":
                # Hand cached GPU buffers back after each call: over a long build the
                # MPS allocator otherwise holds gigabytes and pushes macOS into swap.
                import torch

                torch.mps.empty_cache()
            return np.asarray(vectors, dtype=np.float32)

        _EMBEDDERS[key] = embed
    return _EMBEDDERS[key]


# Models trained with labels in front of their input. EmbeddingGemma marks a question
# "task: search result | query: " and a passage "title: {title} | text: ", per its
# model card; the other embedders define no labels. An index records whether it was
# built with them ("input_labels" in its manifest), and only then are queries
# labelled, so an index built before labels were added is still queried as it was
# built.
_LABELLED = {"google/embeddinggemma-300m"}


def uses_labels(model_name: str) -> bool:
    """Whether the model was trained with question and passage labels."""
    return model_name in _LABELLED


def passage_text(model_name: str, chunk: Chunk) -> str:
    """The text embedded for a passage: labelled with its article title if the model wants it."""
    if uses_labels(model_name):
        return f"title: {chunk.title or 'none'} | text: {chunk.text}"
    return chunk.text


def query_text(model_name: str, text: str, labelled: bool) -> str:
    """The text embedded for a question, labelled when its index was built with labels."""
    if labelled and uses_labels(model_name):
        return f"task: search result | query: {text}"
    return text


def tokenize(text: str) -> list[str]:
    """Lower-case word tokens for BM25.

    Retrieval must tokenise the query with this same function, so that queries and
    chunks are compared on identical terms; it is public for that reason.
    """
    return re.findall(r"\w+", text.lower())


def _build_bm25(chunks: list[Chunk]):
    """Build a BM25 index over the chunk texts."""
    try:
        from rank_bm25 import BM25Okapi
    except ImportError as exc:
        raise ImportError(
            "rank-bm25 is not installed; add it with "
            "uv add --group rag rank-bm25."
        ) from exc
    return BM25Okapi([tokenize(c.text) for c in chunks])


@dataclass
class Index:
    """A searchable index over one chunked, embedded corpus.

    Attributes
    ----------
    chunks : list of Chunk
        The chunks, aligned row-for-row with ``embeddings``.
    embeddings : numpy.ndarray
        Normalised chunk embeddings, shape ``(n_chunks, dim)``.
    bm25 : rank_bm25.BM25Okapi
        The lexical index over the same chunks.
    model_name : str
        The embedding model, so a query can be encoded the same way at search time.
    meta : dict
        The manifest that produced this index (model, config, counts, timestamp).
    """

    chunks: list[Chunk]
    embeddings: np.ndarray
    bm25: Any
    model_name: str
    meta: dict

    def embed_query(self, text: str) -> np.ndarray:
        """Encode a query with the index's own embedding model."""
        text = query_text(self.model_name, text, self.meta.get("input_labels", False))
        return load_embedder(self.model_name)([text])[0]


def build_index(
    chunks: list[Chunk],
    model_name: str,
    out_dir: Path | None = None,
    config_meta: dict | None = None,
    device: str | None = None,
) -> Path:
    """Embed chunks and write the index artifact to disk.

    Writes three files under ``out_dir``: ``embeddings.npy`` (the dense matrix),
    ``chunks.jsonl`` (the chunks, row-aligned with the matrix) and
    ``manifest.json`` (model, dimensions, counts, timestamp and any config passed
    in). BM25 is not persisted; it is cheap and deterministic, so it is rebuilt
    from the chunk texts on load.

    Parameters
    ----------
    chunks : list of Chunk
        The corpus chunks to index.
    model_name : str
        The `sentence-transformers` model to embed with.
    out_dir : pathlib.Path, optional
        Destination directory. Defaults to :data:`INDEX_DIR`.
    config_meta : dict, optional
        Extra fields to record in the manifest, e.g. the chunk strategy and its
        parameters, so the artifact fully describes the config that made it.
    device : str, optional
        Device passed to :func:`load_embedder` (e.g. ``"cpu"`` to avoid an MPS
        stall on larger models). ``None`` lets sentence-transformers choose.

    Returns
    -------
    pathlib.Path
        The directory the index was written to.
    """
    out_dir = out_dir or INDEX_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    embed = load_embedder(model_name, device)
    embeddings = embed([passage_text(model_name, c) for c in chunks])

    np.save(out_dir / "embeddings.npy", embeddings)
    with (out_dir / "chunks.jsonl").open("w", encoding="utf-8") as fh:
        for chunk in chunks:
            fh.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")

    manifest = make_manifest(
        model_name, int(embeddings.shape[1]), len(chunks),
        {"input_labels": uses_labels(model_name), **(config_meta or {})},
    )
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return out_dir


def make_manifest(model_name: str, dim: int, n_chunks: int, config_meta: dict | None = None) -> dict:
    """Describe an index: its model, dimensions, size, build time and config."""
    return {
        "model_name": model_name,
        "dim": dim,
        "n_chunks": n_chunks,
        "created_at": datetime.now(timezone.utc).isoformat(),
        **(config_meta or {}),
    }


def load_index(index_dir: Path | None = None) -> Index:
    """Load an index artifact and rebuild its BM25 side.

    Parameters
    ----------
    index_dir : pathlib.Path, optional
        The directory written by :func:`build_index`. Defaults to :data:`INDEX_DIR`.

    Returns
    -------
    Index
        The loaded index, ready for dense, sparse or hybrid retrieval.

    Raises
    ------
    FileNotFoundError
        If the index has not been built or downloaded, with both ways to obtain it.
    """
    index_dir = index_dir or INDEX_DIR
    if not (index_dir / "manifest.json").exists():
        raise FileNotFoundError(
            f"no index at {index_dir}.\n"
            "Indexes are build artifacts and are not tracked by git. Download the "
            "prebuilt ones from the repository release with\n\n"
            "    gh release download v0.1.0 --pattern rag-artifacts.tar.gz\n"
            "    tar xzf rag-artifacts.tar.gz\n\n"
            "run from the repository root, or rebuild them yourself with "
            "scripts/build_all_indexes.py."
        )
    manifest = json.loads((index_dir / "manifest.json").read_text())
    embeddings = np.load(index_dir / "embeddings.npy")
    with (index_dir / "chunks.jsonl").open(encoding="utf-8") as fh:
        chunks = [Chunk(**json.loads(line)) for line in fh]
    return Index(chunks, embeddings, _build_bm25(chunks), manifest["model_name"], manifest)
