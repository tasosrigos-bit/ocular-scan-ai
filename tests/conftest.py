"""Shared fixtures: a tiny in-memory index and a synthetic B-scan.

The real index needs the embedding model from Hugging Face and the corpus build, and
the real classifier needs the delivered checkpoint, none of which a CI runner has.
These fixtures stand in for them, small enough that the expected rankings can be
worked out by hand.
"""
from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from ocular.rag.chunk import Chunk
from ocular.rag.index import Index, _build_bm25

# Six chunks. Only the first two mention the query terms, so BM25 ranks A above B
# and scores the rest zero. The 2-d embeddings are set so that the cosine with the
# query vector [1, 0] falls from A to F, putting A and B first in both rankings.
TEXTS = {
    "A": "drusen deposits drusen deposits",
    "B": "drusen deposits seen",
    "C": "retinal fluid in the macula",
    "D": "a full thickness macular hole",
    "E": "choroidal neovascular membrane",
    "F": "a normal foveal contour",
}
EMBEDDINGS = {
    "A": [1.0, 0.0],
    "B": [0.9, 0.436],
    "C": [0.5, 0.866],
    "D": [0.0, 1.0],
    "E": [-0.5, 0.866],
    "F": [-1.0, 0.0],
}
QUERY_VECTOR = np.array([1.0, 0.0], dtype=np.float32)


class FakeIndex(Index):
    """An :class:`Index` whose query embedding is fixed rather than computed."""

    def embed_query(self, text: str) -> np.ndarray:
        return QUERY_VECTOR


@pytest.fixture
def tiny_index() -> FakeIndex:
    chunks = [
        Chunk(f"PMC{i}", f"Article {name}", None, "CC BY", "Results", text, 0)
        for i, (name, text) in enumerate(TEXTS.items())
    ]
    emb = np.array(list(EMBEDDINGS.values()), dtype=np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    return FakeIndex(chunks, emb, _build_bm25(chunks), "fake", {})


@pytest.fixture
def synthetic_scan(tmp_path):
    """A 496 x 768 grey image with a bright curved band on a dark background.

    It is shaped like a macular B-scan closely enough for the retina crop and the
    curvature correction to find the band, which is all the pipeline tests need.
    """
    h, w = 496, 768
    rng = np.random.default_rng(0)
    img = rng.normal(20, 4, (h, w))
    x = np.arange(w)
    centre = 240 + 40 * ((x - w / 2) / (w / 2)) ** 2
    rows = np.arange(h)[:, None]
    img[np.abs(rows - centre[None, :]) < 35] += 160
    path = tmp_path / "scan.png"
    Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).save(path)
    return path
