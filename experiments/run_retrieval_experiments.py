"""Retrieval experiments on the full corpus: compare retrieval configurations in MLflow, no LLM.

Every combination of chunking strategy, embedding model, retrieval mode and
reranker (none, the general gte reranker, or the biomedical MedCPT cross-encoder,
which NCBI trained on PubMed search logs) is run over the 75-question full-corpus set and scored against each
question's known source article, its evidence passage and the gold chunks that hold
its answer sentences, with
:mod:`ocular.rag.eval_retrieval`. No language model is involved, so the scores are
deterministic and the whole grid costs nothing beyond compute.

Each configuration is an MLflow child run, named like ``semantic | MedEmbed |
hybrid | gte``, under one parent run per invocation. A child run records the
configuration as parameters, the metrics averaged over the questions (and by
split and subspecialty as tags-free metrics), and a per-question table, so the UI
can chart, for example, evidence Hit@5 by embedding model and chunking strategy,
and drill into any question. BM25 does not depend on the embedding model, so it is
run once per chunking strategy. A configuration already logged in the experiment
is skipped, so the grid can be extended as new collections finish.

Qdrant must hold the ``*__corpus_full`` collections and an MLflow server must be
running (``docker compose up -d qdrant mlflow``). Usage::

    uv run --group eval python experiments/run_retrieval_experiments.py
    uv run --group eval python experiments/run_retrieval_experiments.py --limit 5 --collections fixed__all-MiniLM-L6-v2__corpus_full
"""
from __future__ import annotations

import argparse
import os
import time
from collections import defaultdict
from statistics import mean

from ocular import config
from ocular.rag import chunk, eval_retrieval, questions, rerank, retrieve
from ocular.rag.qdrant_store import QdrantStore

EXPERIMENT = "ocular-retrieval"
from ocular.rag.eval_dataset import NAME as DATASET
SUFFIX = "__corpus_full"
MODES = ("vector", "bm25", "hybrid")
EMBEDDERS = {"all-MiniLM-L6-v2": "MiniLM", "MedEmbed-base-v0.1": "MedEmbed", "embeddinggemma-300m": "Gemma"}
# The rerankers compared, with the short names used in run names. MedCPT is a US
# Government work in the public domain (cite Jin et al., Bioinformatics 2023).
RERANKERS = {
    "Alibaba-NLP/gte-reranker-modernbert-base": "gte",
    "ncbi/MedCPT-Cross-Encoder": "MedCPT",
}


def load_dataset():
    """The question set from the MLflow evaluation dataset, and the dataset itself.

    The questions are read from the registered dataset rather than the local file,
    so every run is tied to the exact records it was evaluated on.
    """
    from mlflow.genai import datasets

    ds = datasets.get_dataset(name=DATASET)
    qs = [
        questions.Question(
            id=r.tags["id"], question=r.inputs["question"],
            ground_truth=r.expectations["expected_response"],
            source_doc_id=r.expectations["source_doc_id"], source_section="",
            evidence=r.expectations["evidence"], split=r.tags["split"],
            subspecialty=r.tags["subspecialty"], category=r.tags["category"],
            answer_sentences=r.expectations.get("answer_sentences") or [],
            gold_chunks=r.expectations.get("gold_chunks") or {},
        )
        for r in ds.to_df().itertuples()
    ]
    return sorted(qs, key=lambda q: q.id), ds


def chunk_lookup(strategy: str, qs) -> tuple[dict[int, int], dict[str, set]]:
    """A strategy's chunk ids by content, and each question's gold chunks by content.

    The first maps ``hash((doc_id, text))`` to the chunk's position, which is its
    Qdrant point id, so the ids of retrieved chunks can be logged. The second holds,
    per question, the ``(doc_id, text)`` of its gold chunks in this strategy.
    """
    chunks = chunk.load_chunks(config.DATA_DIR / "corpus_full" / f"chunks_{strategy}.jsonl")
    ids = {hash((c.doc_id, c.text)): i for i, c in enumerate(chunks)}
    gold = {q.id: {(chunks[i].doc_id, chunks[i].text) for i in q.gold_chunks.get(strategy, [])} for q in qs}
    return ids, gold


def complete_collections(client) -> list[str]:
    """The full-corpus collections whose every passage has been written."""
    names = []
    for c in client.get_collections().collections:
        if not c.name.endswith(SUFFIX):
            continue
        info = client.get_collection(c.name)
        if info.points_count == (info.config.metadata or {}).get("n_chunks"):
            names.append(c.name)
    return sorted(names)


def grid(collections: list[str], rerankers: list[str]) -> list[dict]:
    """Every configuration to run, with BM25 once per chunking strategy."""
    configs, bm25_done = [], set()
    for name in collections:
        strategy, embedder = name.removesuffix(SUFFIX).split("__")
        for mode in MODES:
            if mode == "bm25":
                if strategy in bm25_done:
                    continue
                bm25_done.add(strategy)
            for rr in (None, *rerankers):
                configs.append({
                    "collection": name,
                    "chunk_strategy": strategy,
                    "embedder": "none" if mode == "bm25" else EMBEDDERS.get(embedder, embedder),
                    "mode": mode,
                    "reranker": "none" if rr is None else rr.split("/")[-1],
                    "_reranker_id": rr,
                })
    return configs


def run_name(cfg: dict) -> str:
    rr = RERANKERS.get(cfg["_reranker_id"], cfg["reranker"]) if cfg["_reranker_id"] else "none"
    return f"{cfg['chunk_strategy']} | {cfg['embedder']} | {cfg['mode']} | {rr}"


def evaluate(store, qs, cfg, k: int, candidates: int, lookup=None) -> tuple[dict, list[dict]]:
    """Retrieve for every question under one configuration and score it.

    ``lookup`` is :func:`chunk_lookup` for the configuration's chunking strategy.
    When given, the retrieved chunk ids are logged per question and, for questions
    with gold chunks, the answer level is scored.
    """
    ids, gold = lookup or ({}, {})
    rows = []
    for q in qs:
        start = time.perf_counter()
        if cfg["_reranker_id"]:
            hits = retrieve.retrieve(store, q.question, k=candidates, mode=cfg["mode"], candidates=candidates)
            hits = rerank.rerank(q.question, hits, model_name=cfg["_reranker_id"], k=k)
        else:
            hits = retrieve.retrieve(store, q.question, k=k, mode=cfg["mode"], candidates=candidates)
        seconds = time.perf_counter() - start
        rows.append({
            "id": q.id,
            "question": q.question,
            "split": q.split,
            "subspecialty": q.subspecialty,
            "category": q.category,
            **eval_retrieval.score(hits, q.source_doc_id, q.evidence, k=k, gold=gold.get(q.id) if q.gold_chunks else None),
            "latency_ms": round(seconds * 1000, 1),
            "retrieved_ids": " ".join(str(ids.get(hash((h.chunk.doc_id, h.chunk.text)), "?")) for h in hits),
            "top_titles": " || ".join(h.chunk.title[:60] for h in hits),
        })
    metric_names = [m for m in rows[0] if m.startswith(("article_", "evidence_", "answer_"))] + ["latency_ms"]
    summary = {m: mean(r[m] for r in rows) for m in metric_names}
    by_category = defaultdict(list)
    for r in rows:
        by_category[r["category"]].append(r[f"evidence_hit_at_{k}"])
    summary.update({f"evidence_hit_at_{k}__{c}": mean(v) for c, v in by_category.items()})
    # Per split, so configurations can be selected on validation and reported on test.
    for split in sorted({r["split"] for r in rows}):
        in_split = [r for r in rows if r["split"] == split]
        summary.update({f"{m}__{split}": mean(r[m] for r in in_split) for m in metric_names if m != "latency_ms"})
    return summary, rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--qdrant", default=os.getenv("QDRANT_URL", "http://localhost:6333"))
    parser.add_argument("--tracking-uri", default=os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001"))
    parser.add_argument("--collections", nargs="*", default=None, help="Defaults to every complete one.")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--candidates", type=int, default=50)
    parser.add_argument("--rerankers", nargs="+", default=list(RERANKERS))
    parser.add_argument("--limit", type=int, default=None, help="Use only the first N questions (a quick check).")
    parser.add_argument("--preview", action="store_true",
                        help="Log as a preview, which the full run neither skips nor counts as done.")
    args = parser.parse_args()
    tag = "retrieval-trial" if args.limit else "retrieval-preview" if args.preview else "retrieval"

    import mlflow
    import pandas as pd
    from qdrant_client import QdrantClient

    client = QdrantClient(url=args.qdrant, timeout=120)
    collections = args.collections or complete_collections(client)
    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(EXPERIMENT)
    qs, dataset = load_dataset()
    qs = qs[: args.limit]
    configs = grid(collections, args.rerankers)
    done = set()
    if tag == "retrieval":
        logged = mlflow.search_runs(filter_string="tags.stage = 'retrieval'", output_format="list")
        done = {r.data.tags.get("config") for r in logged if r.info.status == "FINISHED"}
    todo = [c for c in configs if run_name(c) not in done]
    print(f"{len(collections)} collections, {len(configs)} configurations, "
          f"{len(configs) - len(todo)} already logged, {len(todo)} to run on {len(qs)} questions")

    with mlflow.start_run(run_name={"retrieval": "retrieval experiments", "retrieval-preview": "retrieval experiments (preview)",
                                 "retrieval-trial": "retrieval experiments (trial)"}[tag]):
        mlflow.set_tags({"stage": "retrieval-parent", "dataset": DATASET, "dataset_digest": dataset.digest})
        mlflow.log_input(dataset, context="evaluation")
        mlflow.log_params({"k": args.k, "candidates": args.candidates, "n_questions": len(qs)})
        stores, lookups = {}, {}
        for cfg in todo:
            store = stores.setdefault(cfg["collection"], QdrantStore(client, cfg["collection"]))
            strategy = cfg["chunk_strategy"]
            if strategy not in lookups:
                lookups[strategy] = chunk_lookup(strategy, qs)
            summary, rows = evaluate(store, qs, cfg, args.k, args.candidates, lookups[strategy])
            with mlflow.start_run(run_name=run_name(cfg), nested=True):
                mlflow.set_tags({"stage": tag, "config": run_name(cfg), "dataset_digest": dataset.digest})
                mlflow.log_input(dataset, context="evaluation")
                mlflow.log_params({k: v for k, v in cfg.items() if not k.startswith("_")}
                                  | {"k": args.k, "candidates": args.candidates, "n_questions": len(qs)})
                mlflow.log_metrics(summary)
                mlflow.log_table(pd.DataFrame(rows), artifact_file="per_question.json")
            print(f"  {run_name(cfg):45} evidence hit@{args.k} {summary[f'evidence_hit_at_{args.k}']:.3f}  "
                  f"article hit@{args.k} {summary[f'article_hit_at_{args.k}']:.3f}  "
                  f"evidence MRR {summary['evidence_mrr']:.3f}  {summary['latency_ms']:.0f} ms")


if __name__ == "__main__":
    main()
