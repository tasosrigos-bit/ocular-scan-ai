"""Run the RAG evaluation sweep and write experiments/results/rag_eval.csv.

This is the one real experiment of the RAG layer. It walks a grid of retrieval
configurations - chunk strategy x embedding model x retrieval mode x reranking -
and, for each, scores the system on the synthetic question set with the
reference-free metrics in :mod:`ocular.rag.eval_rag`. One row per configuration is
written to a CSV that the ``05`` notebook later reads.

Two design choices keep each run manageable:

- The grid is built from command-line flags and defaults small, so a run is a
  deliberate handful of configurations rather than a large cross-product.
- Results are written incrementally and the run is **resumable**: a configuration
  already present in the CSV is skipped, so an interrupted sweep continues where
  it left off.

Use ``--limit`` to score only the first few questions for a quick pass before
committing to the full set.

Usage
-----
    # compare the three embedders (section chunks, hybrid, reranked), 10 questions
    python experiments/run_rag_eval.py --limit 10

    # a fuller grid on all questions
    python experiments/run_rag_eval.py --modes vector bm25 hybrid --rerank both
"""
from __future__ import annotations

import argparse
import csv
import itertools
from pathlib import Path

from ocular.rag import eval_rag, index, questions

RESULTS = Path(__file__).resolve().parent / "results" / "rag_eval.csv"
RERANKER = "Alibaba-NLP/gte-reranker-modernbert-base"

# Embedder index-directory slugs, matching the names build_index.py wrote.
EMBEDDERS = ["all-MiniLM-L6-v2", "MedEmbed-base-v0.1", "embeddinggemma-300m"]

_FIELDS = [
    "chunk_strategy", "embed_model", "mode", "rerank",
    "n", "faithfulness", "answer_relevance", "context_precision", "latency_s",
]


def _done_configs() -> set[tuple[str, str, str, str]]:
    """Configurations already recorded in the CSV, for resuming."""
    if not RESULTS.exists():
        return set()
    with RESULTS.open(encoding="utf-8") as fh:
        return {
            (r["chunk_strategy"], r["embed_model"], r["mode"], r["rerank"])
            for r in csv.DictReader(fh)
        }


def _append(row: dict) -> None:
    """Append one result row, writing the header if the file is new."""
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    new = not RESULTS.exists()
    with RESULTS.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_FIELDS)
        if new:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    """Build the configuration grid and evaluate each, resuming and appending."""
    parser = argparse.ArgumentParser(description="Run the RAG evaluation sweep.")
    parser.add_argument("--strategies", nargs="+", default=["section"], help="Chunk strategies.")
    parser.add_argument("--embedders", nargs="+", default=EMBEDDERS, help="Embedder index slugs.")
    parser.add_argument("--modes", nargs="+", default=["hybrid"], help="Retrieval modes.")
    parser.add_argument(
        "--rerank", choices=["on", "off", "both"], default="on",
        help="Whether to rerank: on, off, or both (adds a config per option).",
    )
    parser.add_argument(
        "--reranker", default=RERANKER,
        help="Cross-encoder model to rerank with when --rerank is on/both.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only the first N questions.")
    args = parser.parse_args()

    qs = questions.load_questions()
    rerank_opts = {"on": [args.reranker], "off": [None], "both": [None, args.reranker]}[args.rerank]

    grid = list(itertools.product(args.strategies, args.embedders, args.modes, rerank_opts))
    done = _done_configs()
    n_q = args.limit or len(qs)
    print(f"{len(grid)} configs x {n_q} questions; results -> {RESULTS}")

    for strategy, embed, mode, rerank in grid:
        rerank_label = "none" if rerank is None else rerank.split("/")[-1]
        key = (strategy, embed, mode, rerank_label)
        if key in done:
            print(f"[skip] {strategy} | {embed} | {mode} | rerank={rerank_label} (done)")
            continue

        index_dir = index.INDEX_DIR / f"{strategy}__{embed}"
        if not (index_dir / "manifest.json").exists():
            print(f"[skip] {index_dir.name} - index not built yet")
            continue

        print(f"[run ] {strategy} | {embed} | {mode} | rerank={rerank_label} ...")
        result = eval_rag.evaluate(
            qs, index.load_index(index_dir), mode=mode, rerank_model=rerank, limit=args.limit
        )
        _append({
            "chunk_strategy": strategy, "embed_model": embed, "mode": mode,
            "rerank": rerank_label, **result,
        })
        print(f"       faithfulness={result['faithfulness']} "
              f"relevance={result['answer_relevance']} "
              f"context_precision={result['context_precision']}")

    print("done")


if __name__ == "__main__":
    main()
