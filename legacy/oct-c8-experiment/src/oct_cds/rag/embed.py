"""Sentence embeddings for retrieval. Default: BAAI/bge-small-en-v1.5 (384-d,
CPU-fine). Query and passage encodings follow the bge instruction convention."""

from __future__ import annotations

import numpy as np

# bge-* retrieval: prepend this to *queries* only; passages get no prefix.
_BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class Embedder:
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", device: str | None = None):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self._model = SentenceTransformer(model_name, device=device)
        self.dim = int(self._model.get_sentence_embedding_dimension())

    def _encode(self, texts: list[str]) -> np.ndarray:
        vecs = self._model.encode(
            texts, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
        )
        return np.asarray(vecs, dtype=np.float32)

    def encode_passages(self, texts: list[str]) -> np.ndarray:
        return self._encode(list(texts))

    def encode_query(self, text: str) -> np.ndarray:
        prefix = _BGE_QUERY_PREFIX if "bge" in self.model_name.lower() else ""
        return self._encode([prefix + text])[0]
