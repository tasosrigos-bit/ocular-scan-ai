"""Generate the synthetic evaluation question set from the corpus.

Samples passages from the corpus and asks the language model to write a realistic
clinical question for each (see :mod:`ocular.rag.questions`), then writes the set
to ``data/eval/questions.jsonl``. These questions drive the reference-free
evaluation; no gold answers or relevance labels are produced.

Usage
-----
    python scripts/build_questions.py --n 50
"""
from __future__ import annotations

import argparse

from ocular.rag import chunk, index, questions


def main() -> None:
    """Parse arguments, generate the question set, and write it to disk."""
    parser = argparse.ArgumentParser(description="Generate the evaluation question set.")
    parser.add_argument("--n", type=int, default=50, help="Number of questions to generate.")
    parser.add_argument(
        "--from-chunks", default="section",
        help="Which chunk cache to sample seed passages from (fixed/section/semantic).",
    )
    parser.add_argument("--model", default=questions.llm.MODEL, help="Generation model.")
    args = parser.parse_args()

    chunks = chunk.load_chunks(index.INDEX_DIR / "_chunks" / f"chunks_{args.from_chunks}.jsonl")
    print(f"sampling from {len(chunks):,} '{args.from_chunks}' chunks; generating {args.n} questions")

    qs = questions.generate_questions(chunks, n=args.n, model=args.model)
    questions.save_questions(qs)
    print(f"\nwrote {len(qs)} questions to {questions.EVAL_DIR}/questions.jsonl")


if __name__ == "__main__":
    main()
