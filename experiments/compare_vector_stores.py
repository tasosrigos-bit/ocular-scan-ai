"""Compare retrieval from Qdrant against the in-memory index it was loaded from.

Moving the index into Qdrant changes where the rankings come from, not the
retrieval configuration: the dense side is the same embeddings searched by cosine,
and the hybrid fusion is the same RRF code. The lexical side differs slightly,
because Qdrant computes the BM25 IDF with a formula that stays positive (see
:mod:`ocular.rag.qdrant_store`). This script measures what that changes on the 50
evaluation questions, for the retrieval the application actually runs: a hybrid
search for 50 candidates, then the cross-encoder keeping the top 5.

For each question it records, from both stores, whether the article the question
was written from is among the final five passages (hit@5), how many of the five
passages the two stores share, and the retrieval latency. One row per question is
written to ``experiments/results/vector_store_parity.csv`` and a summary is printed.

Qdrant must be running with the index loaded::

    docker compose up -d qdrant
    uv run python scripts/build_qdrant.py
    uv run python experiments/compare_vector_stores.py
"""
from __future__ import annotations

import argparse
import csv
import os
import time
from pathlib import Path

from ocular import service
from ocular.rag import index, questions, rerank, retrieve
from ocular.rag.qdrant_store import QdrantStore

RESULTS = Path(__file__).resolve().parent / "results" / "vector_store_parity.csv"


def _final_five(store, query: str) -> tuple[list, float]:
    """The application's retrieval: 50 hybrid candidates, reranked to five."""
    start = time.perf_counter()
    hits = retrieve.retrieve(store, query, k=50, mode="hybrid", candidates=50)
    seconds = time.perf_counter() - start
    return rerank.rerank(query, hits, model_name=rerank.DEFAULT_RERANKER, k=5), seconds


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", default=os.getenv("QDRANT_URL", "http://localhost:6333"))
    parser.add_argument("--limit", type=int, default=None, help="score only the first N questions")
    args = parser.parse_args()

    memory = index.load_index(index.INDEX_DIR / service.INDEX)
    qdrant = QdrantStore.connect(args.url, service.INDEX)
    qs = questions.load_questions()[: args.limit]
    _final_five(memory, "warm up the embedder and the reranker")

    rows = []
    for q in qs:
        mem, mem_s = _final_five(memory, q.question)
        qd, qd_s = _final_five(qdrant, q.question)
        mem_ids = [(h.chunk.doc_id, h.chunk.index) for h in mem]
        qd_ids = [(h.chunk.doc_id, h.chunk.index) for h in qd]
        rows.append({
            "id": q.id,
            "memory_hit5": int(q.source_doc_id in {d for d, _ in mem_ids}),
            "qdrant_hit5": int(q.source_doc_id in {d for d, _ in qd_ids}),
            "shared_of_5": len(set(mem_ids) & set(qd_ids)),
            "same_top1": int(mem_ids[:1] == qd_ids[:1]),
            "memory_retrieval_ms": round(mem_s * 1000, 1),
            "qdrant_retrieval_ms": round(qd_s * 1000, 1),
        })

    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    n = len(rows)
    mean = lambda key: sum(r[key] for r in rows) / n  # noqa: E731
    print(f"{n} questions, written to {RESULTS.relative_to(Path.cwd())}")
    print(f"hit@5           memory {mean('memory_hit5'):.2f}   qdrant {mean('qdrant_hit5'):.2f}")
    print(f"shared passages {mean('shared_of_5'):.2f} of 5 on average")
    print(f"same top hit    {mean('same_top1'):.2f}")
    print(f"retrieval       memory {mean('memory_retrieval_ms'):.0f} ms   "
          f"qdrant {mean('qdrant_retrieval_ms'):.0f} ms per query (before reranking)")


if __name__ == "__main__":
    main()
