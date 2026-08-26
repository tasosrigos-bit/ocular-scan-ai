"""Build a retrieval index over the corpus: chunk, embed, and persist.

This is the second stage of the retrieval pipeline. It chunks the saved corpus
with a chosen strategy, embeds the chunks with a chosen model, and writes a
self-describing index artifact under ``data/index/``. Both the chunk strategy and
the embedding model are experiment knobs, passed on the command line.

Chunking is cached per strategy. The chunk boundaries depend only on the strategy
(and, for the semantic strategy, on a fixed small embedder used to detect topic
shifts), not on the retrieval embedding model, so several embedders reuse the same
chunks without paying the chunking cost again. This matters most for the semantic
strategy, whose boundary detection embeds every sentence in the corpus.

Usage
-----
    python scripts/build_index.py \\
        --chunk-strategy section \\
        --embed-model sentence-transformers/all-MiniLM-L6-v2
"""
from __future__ import annotations

import argparse

from ocular.rag import chunk, corpus, index

# Fixed embedder for semantic chunk-boundary detection. It is a constant of the
# pipeline rather than a knob, so it stays independent of the retrieval embedder.
CHUNK_EMBEDDER = "sentence-transformers/all-MiniLM-L6-v2"


def _slug(model_name: str) -> str:
    """The last path component of a model id, for naming the index directory."""
    return model_name.split("/")[-1]


def load_or_build_chunks(docs: list[corpus.Document], strategy: str) -> list[chunk.Chunk]:
    """Return the corpus chunked with ``strategy``, using a per-strategy cache."""
    cache = index.INDEX_DIR / "_chunks" / f"chunks_{strategy}.jsonl"
    if cache.exists():
        print(f"chunks: reusing cache {cache.name}")
        return chunk.load_chunks(cache)

    embed_fn = index.load_embedder(CHUNK_EMBEDDER) if strategy == "semantic" else None
    print(f"chunks: building ({strategy}) ...")
    chunks = chunk.chunk_corpus(docs, strategy=strategy, embed_fn=embed_fn)
    chunk.save_chunks(chunks, cache)
    return chunks


def main() -> None:
    """Parse arguments, chunk and embed the corpus, and write the index."""
    parser = argparse.ArgumentParser(description="Build a retrieval index over the corpus.")
    parser.add_argument(
        "--chunk-strategy", default="section", choices=chunk.STRATEGIES,
        help="How to split articles into chunks.",
    )
    parser.add_argument(
        "--embed-model", required=True,
        help="sentence-transformers model id used to embed the chunks.",
    )
    parser.add_argument(
        "--device", default=None,
        help="Embedding device: cpu, mps or cuda. Default lets the library choose; "
             "use cpu for larger models that stall on Apple MPS.",
    )
    args = parser.parse_args()

    docs = corpus.load_corpus()
    print(f"corpus: {len(docs)} documents")

    chunks = load_or_build_chunks(docs, args.chunk_strategy)
    print(f"chunks: {len(chunks):,}")

    out_dir = index.INDEX_DIR / f"{args.chunk_strategy}__{_slug(args.embed_model)}"
    print(f"embedding {len(chunks):,} chunks with {args.embed_model} ...")
    index.build_index(
        chunks,
        args.embed_model,
        out_dir,
        config_meta={"chunk_strategy": args.chunk_strategy},
        device=args.device,
    )
    print(f"index written to {out_dir}")


if __name__ == "__main__":
    main()
