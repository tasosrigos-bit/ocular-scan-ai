"""Phase B: basic RAG vs agentic RAG on the selected retrieval configuration.

Phase A chose the retrieval configuration (fixed chunks, hybrid retrieval, the
gte-modernbert reranker, on the MedEmbed index). Phase B holds that configuration
fixed and compares the two answering pipelines from :mod:`ocular.rag.pipeline`:
basic (one retrieval then generate) versus agentic (a LangGraph agent that decides
how to search). Both are scored with the same reference-free metrics, and a
``pipeline`` column distinguishes them, so the cost of the agent (latency, extra
model calls) can be weighed against any gain in the metrics.

Usage
-----
    python experiments/run_phaseb.py               # both pipelines, all questions
    python experiments/run_phaseb.py --limit 10    # a quick pass
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from ocular.rag import eval_rag, index, pipeline, questions, rerank

RESULTS = Path(__file__).resolve().parent / "results" / "rag_eval_phaseb.csv"

# The retrieval configuration selected in Phase A.
INDEX = "fixed__MedEmbed-base-v0.1"
MODE = "hybrid"
RERANKER = rerank.DEFAULT_RERANKER

PIPELINES = {"basic": pipeline.basic_answer, "agentic": pipeline.agentic_answer}
_FIELDS = ["pipeline", "index", "mode", "rerank", "n", "faithfulness", "answer_relevance", "context_precision", "latency_s"]


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
    """Evaluate each pipeline on the selected config and append its metrics."""
    parser = argparse.ArgumentParser(description="Phase B: basic vs agentic RAG.")
    parser.add_argument("--index", default=INDEX, help="Index directory name (strategy__embedder).")
    parser.add_argument("--mode", default=MODE, help="Retrieval mode.")
    parser.add_argument("--rerank", choices=["on", "off"], default="on")
    parser.add_argument("--reranker", default=RERANKER, help="Cross-encoder when --rerank is on.")
    parser.add_argument("--pipelines", nargs="+", default=list(PIPELINES), choices=list(PIPELINES))
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only the first N questions.")
    args = parser.parse_args()

    rerank_model = args.reranker if args.rerank == "on" else None
    rerank_label = args.reranker.split("/")[-1] if rerank_model else "none"

    qs = questions.load_questions()
    idx = index.load_index(index.INDEX_DIR / args.index)
    print(f"Phase B on {args.index} ({args.mode}, rerank={rerank_label}) | {args.limit or len(qs)} questions")

    for name in args.pipelines:
        print(f"\n[{name}] evaluating ...")
        result = eval_rag.evaluate(
            qs, idx, mode=args.mode, rerank_model=rerank_model, limit=args.limit,
            answer_fn=PIPELINES[name],
        )
        _append({"pipeline": name, "index": args.index, "mode": args.mode, "rerank": rerank_label, **result})
        print(f"       faithfulness={result['faithfulness']} "
              f"relevance={result['answer_relevance']} "
              f"context_precision={result['context_precision']} "
              f"latency={result['latency_s']}s")

    print("\ndone ->", RESULTS)


if __name__ == "__main__":
    main()
