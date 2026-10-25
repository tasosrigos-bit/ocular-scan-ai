"""Write the full-corpus evaluation question set with an OpenAI model.

Cuts every article into windows of whole sentences, independent of any chunking
strategy, and has a model from a different family than the answering model write
one clinical question per window, with the prompt of :mod:`ocular.rag.questions`.
The model also copies, word for word, the sentences its answer comes from; a
question whose sentences are not found exactly in the window is rejected. The
sentences are then mapped to the chunks that hold them in every chunking strategy,
the gold chunks the retrieval experiments score against. Questions too close in
meaning to an earlier one, or too far from their own sentences, are turned away
with a small local embedding model. The set is checked for copied or
self-referring questions, balanced across subspecialties and clinical categories,
split into train, validation and test for prompt optimisation, written to
``data/eval/questions_full.jsonl`` and registered as the MLflow evaluation dataset
``ocular-questions-full`` that the experiments read. An existing dataset is kept
unless ``--replace`` is given, so a rerun cannot change the set past runs used. The
original 50-question set the report was evaluated on is left untouched.

Needs ``OPENAI_API_KEY`` in ``.env``, the full corpus and its chunks for all three
strategies. A trial writes only a file, never the MLflow dataset. Usage::

    uv run --group eval python scripts/build_eval_questions.py
    uv run --group eval python scripts/build_eval_questions.py --n 10 --no-mlflow --out /tmp/try.jsonl
"""
from __future__ import annotations

import argparse
import os
from collections import Counter
from pathlib import Path

from ocular import config
from ocular.rag import chunk, corpus, eval_dataset, index, questions

# Standard per-token prices of the writing model in US dollars (input, output),
# from OpenAI's model page, for the cost report.
PRICE_PER_TOKEN = {"gpt-5.4-mini": (0.75e-6, 4.50e-6)}
EMBEDDER = "sentence-transformers/all-MiniLM-L6-v2"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n", type=int, default=75, help="Number of questions.")
    parser.add_argument("--model", default=questions.QUESTION_MODEL, help="OpenAI model that writes them.")
    parser.add_argument("--corpus-dir", type=Path, default=config.DATA_DIR / "corpus_full")
    parser.add_argument("--out", type=Path, default=questions.EVAL_DIR / "questions_full.jsonl")
    parser.add_argument("--tracking-uri", default=os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001"))
    parser.add_argument("--replace", action="store_true", help="Overwrite the existing MLflow dataset.")
    parser.add_argument("--no-mlflow", action="store_true", help="Only write the file (for trial runs).")
    args = parser.parse_args()

    if not args.no_mlflow and not args.replace:
        # Fail before spending API calls if the dataset would not be written.
        import mlflow
        from mlflow.genai import datasets

        mlflow.set_tracking_uri(args.tracking_uri)
        if datasets.search_datasets(filter_string=f"name = '{eval_dataset.NAME}'"):
            raise SystemExit(f"MLflow dataset {eval_dataset.NAME!r} exists; use --replace or --no-mlflow")

    seeds = questions.seed_windows(corpus.load_corpus(args.corpus_dir))
    print(f"sampling from {len(seeds):,} sentence windows; writing {args.n} questions with {args.model}")
    qs, passed_over = questions.generate_question_set(
        seeds, n=args.n, model=args.model, embed=index.load_embedder(EMBEDDER),
        exclude_sections=questions.SEED_EXCLUDE_SECTIONS,
    )
    if len(qs) < args.n:
        raise SystemExit(f"only {len(qs)} questions passed; the corpus ran out of usable passages")

    sizes = questions.SPLIT if args.n == sum(questions.SPLIT.values()) else {
        "train": round(args.n * 0.13), "validation": round(args.n * 0.54),
    }
    sizes.setdefault("test", args.n - sum(sizes.values()))
    qs = questions.split_questions(qs, sizes)
    print("mapping answer sentences to the chunks of every chunking strategy ...")
    qs = questions.map_gold_chunks(qs, {
        s: chunk.load_chunks(args.corpus_dir / f"chunks_{s}.jsonl") for s in chunk.STRATEGIES
    })
    questions.save_questions(qs, args.out)
    missing = [(q.id, s) for q in qs for s in chunk.STRATEGIES if not q.gold_chunks.get(s)]
    if missing:
        print(f"WARNING: {len(missing)} question-strategy pairs have no gold chunk: {missing[:10]}")
    split = [(q.id, s) for q in qs for s, how in q.gold_match.items() if how == "split"]
    print(f"gold sentences matched exactly in every strategy for {len(qs) - len({i for i, _ in split})} "
          f"of {len(qs)} questions; cut by a chunk boundary: {split}")
    used = questions.USAGE
    price = next((v for k, v in PRICE_PER_TOKEN.items() if args.model.startswith(k)), None)
    cost = f", about ${used['input'] * price[0] + used['output'] * price[1]:.2f}" if price else ""
    print(f"tokens: {used['input']:,} in, {used['output']:,} out{cost}")

    print(f"\nwrote {len(qs)} questions to {args.out}")
    if not args.no_mlflow:
        eval_dataset.register(qs, args.tracking_uri, replace=args.replace)
        print(f"registered them as MLflow dataset {eval_dataset.NAME!r}")
    print("passed over:", dict(passed_over.most_common()))
    for field in ("subspecialty", "category", "split"):
        print(f"{field}:", dict(Counter(getattr(q, field) for q in qs).most_common()))


if __name__ == "__main__":
    main()
