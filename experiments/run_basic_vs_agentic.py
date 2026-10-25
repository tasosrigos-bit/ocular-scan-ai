"""Basic against agentic RAG on the chosen retrieval configuration, scored on the passages each answers from.

The retrieval experiments chose section chunks, EmbeddingGemma, hybrid retrieval
and the gte reranker. Both pipelines search that collection the same way and hand
five passages to the same answering step; only the searching differs:

- **basic** - one search for the question: 50 hybrid candidates, reranked by gte,
  top five.
- **agentic** - a search-only agent (:func:`ocular.rag.pipeline.agentic_search`).
  Its first search is the basic one; it may then reformulate and search again, up
  to five searches in all. Everything it finds is pooled, reranked against the
  original question, and the top five are handed on.

The five passages are scored with :mod:`ocular.rag.eval_retrieval` against the
question's gold chunks, as in the retrieval experiments, so the comparison needs no
judge: did the agent's extra searches put the answer in front of the model more
often? For the agent the pool is scored too (was a gold chunk found at all, even if
cut from the top five), with the number of searches, its queries and the latency.
Both pipelines then write the answer, logged for inspection.

The agent is designed on the validation split and reported once on the test split.
Each pipeline is an MLflow child run in the ``ocular-rag-pipelines`` experiment
under one parent per split; the parent holds the paired comparison: the
per-question difference with a bootstrap interval, and the questions the agent
gained or lost. A trial (``--limit``) logs nothing to MLflow. Usage::

    uv run --group eval python experiments/run_basic_vs_agentic.py --split validation
    uv run --group eval python experiments/run_basic_vs_agentic.py --split test
    uv run --group eval python experiments/run_basic_vs_agentic.py --limit 5
"""
from __future__ import annotations

import argparse
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor
from statistics import mean

from run_retrieval_experiments import chunk_lookup, load_dataset

from ocular import config
from ocular.rag import eval_retrieval, generate, llm, pipeline, rerank
from ocular.rag.qdrant_store import QdrantStore

EXPERIMENT = "ocular-rag-pipelines"
# The configuration chosen in the retrieval experiments: best answer MRR on the
# validation split among the three tied on answer Hit@5.
COLLECTION = "section__embeddinggemma-300m__corpus_full"
STRATEGY = "section"
MODE = "hybrid"
RERANKER = "Alibaba-NLP/gte-reranker-modernbert-base"
COMPARED = ("answer_hit_at_5", "answer_mrr", "answer_ndcg_at_5", "evidence_hit_at_5", "article_hit_at_5")


def ids_of(hits, ids: dict) -> str:
    return " ".join(str(ids.get(hash((h.chunk.doc_id, h.chunk.text)), "?")) for h in hits)


def run_question(q, store, lookup, k: int, candidates: int) -> tuple[dict, dict]:
    """Both pipelines on one question: the scored passages, the agent's trace and both answers."""
    ids, gold = lookup
    g = gold.get(q.id) if q.gold_chunks else None
    base = {"id": q.id, "question": q.question, "split": q.split, "category": q.category}

    start = time.perf_counter()
    hits = pipeline.search_once(store, q.question, mode=MODE, k=k, candidates=candidates, rerank_model=RERANKER)
    search_s = time.perf_counter() - start
    answer = generate.answer_from(q.question, hits).answer
    basic = base | eval_retrieval.score(hits, q.source_doc_id, q.evidence, k=k, gold=g) | {
        "searches": 1, "search_s": round(search_s, 2), "total_s": round(time.perf_counter() - start, 2),
        "retrieved_ids": ids_of(hits, ids), "answer": answer,
    }

    start = time.perf_counter()
    found = pipeline.agentic_search(q.question, store, mode=MODE, k=k, candidates=candidates, rerank_model=RERANKER)
    search_s = time.perf_counter() - start
    answer = generate.answer_from(q.question, found.top).answer
    pool_hit = float(bool(g) and any((h.chunk.doc_id, h.chunk.text) in g for h in found.pool))
    agentic = base | eval_retrieval.score(found.top, q.source_doc_id, q.evidence, k=k, gold=g) | {
        "searches": len(found.queries), "pool_size": len(found.pool), "pool_answer_hit": pool_hit,
        "search_s": round(search_s, 2), "total_s": round(time.perf_counter() - start, 2),
        "queries": " || ".join(found.queries), "reasons": " || ".join(found.reasons),
        "retrieved_ids": ids_of(found.top, ids), "answer": answer,
    }
    return basic, agentic


def paired(basic: list[dict], agentic: list[dict], metric: str, n: int = 2000) -> dict:
    """The agent's per-question gain over basic on one metric, with a bootstrap interval."""
    diffs = [a[metric] - b[metric] for b, a in zip(basic, agentic)]
    rng = random.Random(config.SEED)
    means = sorted(mean(rng.choices(diffs, k=len(diffs))) for _ in range(n))
    return {
        "diff": mean(diffs), "ci_low": means[int(0.025 * n)], "ci_high": means[int(0.975 * n) - 1],
        "gained": sum(d > 0 for d in diffs), "lost": sum(d < 0 for d in diffs),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--qdrant", default=os.getenv("QDRANT_URL", "http://localhost:6333"))
    parser.add_argument("--tracking-uri", default=os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001"))
    parser.add_argument("--split", default="validation", choices=("train", "validation", "test"))
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--candidates", type=int, default=50)
    parser.add_argument("--workers", type=int, default=4, help="Questions run at once.")
    parser.add_argument("--limit", type=int, default=None,
                        help="A trial on the first N questions: nothing is logged to MLflow.")
    args = parser.parse_args()
    trial = args.limit is not None

    import mlflow
    import pandas as pd
    from qdrant_client import QdrantClient

    mlflow.set_tracking_uri(args.tracking_uri)
    qs, dataset = load_dataset()
    qs = [q for q in qs if q.split == args.split][: args.limit]
    store = QdrantStore(QdrantClient(url=args.qdrant, timeout=120), COLLECTION)
    lookup = chunk_lookup(STRATEGY, qs)
    print(f"basic vs agentic on {len(qs)} {args.split} questions{' (trial, not logged)' if trial else ''}", flush=True)

    # Load the embedder and reranker once before the threads start.
    pipeline.search_once(store, qs[0].question, mode=MODE, k=args.k, candidates=args.candidates, rerank_model=RERANKER)
    start = time.perf_counter()
    with ThreadPoolExecutor(args.workers) as pool:
        results = list(pool.map(lambda q: run_question(q, store, lookup, args.k, args.candidates), qs))
    basic, agentic = [r[0] for r in results], [r[1] for r in results]
    print(f"ran in {time.perf_counter() - start:.0f}s", flush=True)

    summaries = {}
    for name, rows in (("basic", basic), ("agentic", agentic)):
        numeric = [m for m in rows[0] if isinstance(rows[0][m], (int, float))]
        summaries[name] = {m: mean(r[m] for r in rows) for m in numeric}
        s = summaries[name]
        print(f"  {name:8} answer hit@5 {s['answer_hit_at_5']:.3f}  answer MRR {s['answer_mrr']:.3f}  "
              f"evidence hit@5 {s['evidence_hit_at_5']:.3f}  article hit@5 {s['article_hit_at_5']:.3f}  "
              f"searches {s['searches']:.2f}  total {s['total_s']:.1f}s"
              + (f"  pool answer hit {s['pool_answer_hit']:.3f}" if name == "agentic" else ""))
    comparison = {m: paired(basic, agentic, m) for m in COMPARED}
    for m, c in comparison.items():
        print(f"  agentic - basic {m:20} {c['diff']:+.3f} [{c['ci_low']:+.3f}, {c['ci_high']:+.3f}]  "
              f"gained {c['gained']}  lost {c['lost']}")
    if trial:
        for r in agentic:
            print(f"\n{r['id']} ({r['searches']} searches): {r['queries']}\n  reasons: {r['reasons']}")
        return

    mlflow.set_experiment(EXPERIMENT)
    params = {"collection": COLLECTION, "mode": MODE, "reranker": RERANKER, "k": args.k,
              "candidates": args.candidates, "split": args.split, "n_questions": len(qs),
              "answer_model": llm.MODEL}
    with mlflow.start_run(run_name=f"basic vs agentic ({args.split})"):
        mlflow.set_tags({"stage": "pipelines-parent", "split": args.split, "dataset_digest": dataset.digest})
        mlflow.log_input(dataset, context="evaluation")
        mlflow.log_params(params | {"max_searches": pipeline.MAX_SEARCHES})
        mlflow.log_metrics({f"{m}__{k}": v for m, c in comparison.items() for k, v in c.items()})
        mlflow.log_table(pd.DataFrame([{"metric": m, **c} for m, c in comparison.items()]), artifact_file="comparison.json")
        for name, rows in (("basic", basic), ("agentic", agentic)):
            with mlflow.start_run(run_name=f"{name} ({args.split})", nested=True):
                mlflow.set_tags({"stage": "pipelines", "pipeline": name, "split": args.split,
                                 "dataset_digest": dataset.digest})
                mlflow.log_input(dataset, context="evaluation")
                mlflow.log_params(params | ({"max_searches": pipeline.MAX_SEARCHES} if name == "agentic" else {}))
                mlflow.log_metrics(summaries[name])
                mlflow.log_table(pd.DataFrame(rows), artifact_file="per_question.json")


if __name__ == "__main__":
    main()
