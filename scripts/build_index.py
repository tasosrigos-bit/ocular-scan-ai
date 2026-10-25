"""Build a retrieval index over the corpus: chunk, embed, and persist.

This is the second stage of the retrieval pipeline. It chunks the saved corpus
with a chosen strategy, embeds the chunks with a chosen model, and writes a
self-describing index artifact under ``data/index/``. Both the chunk strategy and
the embedding model are experiment knobs, passed on the command line.

With ``--qdrant`` the chunks are embedded batch by batch straight into a Qdrant
collection named like the index directory would be, and no index files are written.
A build that stops part way resumes from the last finished batch when run again.
That is the route for the application, which searches Qdrant, and for a corpus too
large to embed in one sitting; the files remain the route for the notebooks and the
evaluation scripts, which load an index into memory.

``--corpus-dir`` indexes a corpus other than ``data/corpus``. Its chunks are then
cached beside it rather than under ``data/index/_chunks``, and its name is the
corpus folder's, e.g. ``fixed__MedEmbed-base-v0.1__corpus_full``.

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

    # straight into Qdrant, at $QDRANT_URL or http://localhost:6333
    python scripts/build_index.py --chunk-strategy fixed \\
        --embed-model abhinand/MedEmbed-base-v0.1 --device cpu --qdrant

    # the full corpus, on the Apple GPU
    python scripts/build_index.py --corpus-dir data/corpus_full --chunk-strategy fixed \\
        --embed-model abhinand/MedEmbed-base-v0.1 --device mps --qdrant
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from ocular.rag import chunk, corpus, index, qdrant_store

# Fixed embedder for semantic chunk-boundary detection. It is a constant of the
# pipeline rather than a knob, so it stays independent of the retrieval embedder.
CHUNK_EMBEDDER = "sentence-transformers/all-MiniLM-L6-v2"


def _slug(model_name: str) -> str:
    """The last path component of a model id, for naming the index directory."""
    return model_name.split("/")[-1]


def load_or_build_chunks(
    docs: list[corpus.Document], strategy: str, cache_dir: Path | None = None
) -> list[chunk.Chunk]:
    """Return the corpus chunked with ``strategy``, using a per-strategy cache."""
    cache = (cache_dir or index.INDEX_DIR / "_chunks") / f"chunks_{strategy}.jsonl"
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
    parser.add_argument(
        "--qdrant", nargs="?", metavar="URL", default=None,
        const=os.getenv("QDRANT_URL", "http://localhost:6333"),
        help="Write the index into the Qdrant server at URL instead of data/index. "
             "Without a URL, $QDRANT_URL or http://localhost:6333.",
    )
    parser.add_argument(
        "--corpus-dir", type=Path, default=None,
        help="Corpus folder to index. Defaults to data/corpus.",
    )
    args = parser.parse_args()

    docs = corpus.load_corpus(args.corpus_dir)
    print(f"corpus: {len(docs)} documents")

    other_corpus = args.corpus_dir is not None and args.corpus_dir.resolve() != corpus.CORPUS_DIR
    chunks = load_or_build_chunks(
        docs, args.chunk_strategy, args.corpus_dir if other_corpus else None
    )
    print(f"chunks: {len(chunks):,}")

    name = f"{args.chunk_strategy}__{_slug(args.embed_model)}"
    if other_corpus:
        name += f"__{args.corpus_dir.resolve().name}"
    print(f"embedding {len(chunks):,} chunks with {args.embed_model} ...", flush=True)
    if args.qdrant:
        from qdrant_client import QdrantClient
        from tqdm import tqdm

        embed = index.load_embedder(args.embed_model, args.device)
        meta = index.make_manifest(
            args.embed_model, int(embed(["dimension probe"]).shape[1]), len(chunks),
            {
                "chunk_strategy": args.chunk_strategy,
                "corpus_documents": len(docs),
                "input_labels": index.uses_labels(args.embed_model),
            },
        )
        bar = None

        def progress(done: int) -> None:
            # The bar opens at the resume point, so a resumed build's rate and ETA
            # count only the chunks embedded in this run.
            nonlocal bar
            if bar is None:
                bar = tqdm(total=len(chunks), initial=done, unit="chunk")
            else:
                bar.update(done - bar.n)

        n = qdrant_store.stream_collection(
            QdrantClient(url=args.qdrant, timeout=120), name, chunks, embed, meta,
            progress=progress, as_text=lambda c: index.passage_text(args.embed_model, c),
        )
        if bar is not None:
            bar.close()
        print(f"{n:,} chunks in {args.qdrant} collection {name!r}")
        return

    out_dir = index.INDEX_DIR / name
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
