"""FAISS flat inner-product index over the knowledge-base passages.

The corpus is tiny (~30 passages), so this is a brute-force exact index. It is
cached to `knowledge_base/.index/` and rebuilt when the KB content or the embed
model changes.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from oct_cds.common.logging import get_logger
from oct_cds.rag.embed import Embedder
from oct_cds.rag.ingest import KnowledgeBase, load_knowledge_base

log = get_logger(__name__)


class VectorIndex:
    def __init__(self, dim: int):
        import faiss

        self.dim = dim
        self._index = faiss.IndexFlatIP(dim)
        self.ids: list[str] = []

    def add(self, ids: list[str], vecs: np.ndarray) -> None:
        self._index.add(np.ascontiguousarray(vecs, dtype=np.float32))
        self.ids.extend(ids)

    def search(self, query_vec: np.ndarray, k: int) -> list[tuple[str, float]]:
        k = min(k, len(self.ids))
        if k == 0:
            return []
        scores, idx = self._index.search(query_vec.reshape(1, -1).astype(np.float32), k)
        return [(self.ids[i], float(s)) for s, i in zip(scores[0], idx[0]) if i >= 0]

    # -- persistence ------------------------------------------------
    def save(self, dir_: Path, meta: dict) -> None:
        import faiss

        dir_.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(dir_ / "passages.faiss"))
        (dir_ / "ids.json").write_text(json.dumps(self.ids), encoding="utf-8")
        (dir_ / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, dir_: Path) -> "VectorIndex":
        import faiss

        idx = faiss.read_index(str(dir_ / "passages.faiss"))
        obj = cls.__new__(cls)
        obj._index = idx
        obj.dim = idx.d
        obj.ids = json.loads((dir_ / "ids.json").read_text(encoding="utf-8"))
        return obj


def _index_dir(kb: KnowledgeBase) -> Path:
    return kb.kb_dir / ".index"


def _fingerprint(kb: KnowledgeBase, model_name: str) -> dict:
    return {"kb_content_hash": kb.content_hash, "embed_model": model_name,
            "n_passages": len(kb.passages)}


def build_or_load_index(
    kb: KnowledgeBase | None = None,
    embedder: Embedder | None = None,
    *,
    model_name: str = "BAAI/bge-small-en-v1.5",
    rebuild: bool = False,
) -> tuple[VectorIndex, Embedder, KnowledgeBase]:
    kb = kb or load_knowledge_base()
    idir = _index_dir(kb)
    fp = _fingerprint(kb, model_name) if embedder is None else _fingerprint(kb, embedder.model_name)

    if not rebuild and (idir / "meta.json").exists():
        try:
            if json.loads((idir / "meta.json").read_text()) == fp:
                log.info("using cached passage index (%s)", idir)
                return VectorIndex.load(idir), embedder or Embedder(model_name), kb
        except Exception:  # noqa: BLE001
            pass

    embedder = embedder or Embedder(model_name)
    vecs = embedder.encode_passages([p.text for p in kb.passages])
    index = VectorIndex(embedder.dim)
    index.add([p.id for p in kb.passages], vecs)
    index.save(idir, _fingerprint(kb, embedder.model_name))
    log.info("built passage index: %d passages -> %s", len(kb.passages), idir)
    return index, embedder, kb
