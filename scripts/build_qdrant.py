"""Load a built index into Qdrant.

Reads the in-memory index artifact from ``data/index/<name>`` (the chunks, their
embeddings and the manifest) and writes it into a Qdrant collection of the same
name, with the dense embeddings and the BM25 sparse vectors side by side. Nothing
is re-embedded, so this takes seconds rather than the hour the embedding took.

Start Qdrant first, for example with ``docker compose up -d qdrant``, then::

    uv run python scripts/build_qdrant.py
    uv run python scripts/build_qdrant.py --index section__MedEmbed-base-v0.1
"""
from __future__ import annotations

import argparse
import os
import time

from qdrant_client import QdrantClient

from ocular import service
from ocular.rag import index, qdrant_store


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--index", default=service.INDEX, help="index directory name under data/index")
    parser.add_argument("--url", default=os.getenv("QDRANT_URL", "http://localhost:6333"))
    args = parser.parse_args()

    start = time.perf_counter()
    idx = index.load_index(index.INDEX_DIR / args.index)
    n = qdrant_store.build_collection(idx, QdrantClient(url=args.url), args.index)
    print(f"wrote {n} chunks to {args.url} collection {args.index!r} "
          f"in {time.perf_counter() - start:.1f}s")


if __name__ == "__main__":
    main()
