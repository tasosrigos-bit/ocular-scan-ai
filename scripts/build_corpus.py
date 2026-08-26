"""Build the RAG corpus from PubMed Central and write it to disk.

This is the first stage of the retrieval pipeline: it searches PMC's open-access
subset over a whitelist of ophthalmology journals, downloads and parses each
article, and writes ``corpus.jsonl`` plus a ``metadata.csv`` manifest under
``data/corpus/``. It is a deterministic data-build step, not an experiment - it
produces the artifact the indexer consumes, and reruns simply refresh it.

The heavy lifting lives in :mod:`ocular.rag.corpus`; this script only parses
arguments, runs the build, and prints a short summary (including a licence
breakdown, so the licences of what was collected are visible at a glance).

Usage
-----
    # optional, lifts NCBI's rate cap from 3 to 10 requests/second
    export NCBI_API_KEY=... NCBI_EMAIL=you@example.com

    python scripts/build_corpus.py               # the whole journal-whitelist scope
    python scripts/build_corpus.py --limit 500   # top 500 articles only
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from ocular.rag import corpus


def main() -> None:
    """Parse arguments, build the corpus, and report what was collected."""
    parser = argparse.ArgumentParser(description="Build the RAG corpus from PMC Open Access.")
    parser.add_argument(
        "--journal",
        action="append",
        dest="journals",
        help="Journal to include; repeatable. Defaults to the ophthalmology whitelist.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max articles to fetch, top-ranked first. Default: the whole scope.",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=corpus.CORPUS_DIR, help="Where to write the corpus."
    )
    parser.add_argument(
        "--allow-noncommercial",
        action="store_true",
        help="Also keep CC BY-NC articles (only if the project stays non-commercial).",
    )
    args = parser.parse_args()

    journals = args.journals or corpus.OPHTHALMOLOGY_JOURNALS
    scope = f"{len(journals)} ophthalmology journals"
    print(f"building corpus from {scope}" + (f", up to {args.limit} articles" if args.limit else ""))

    docs = corpus.build_corpus(
        journals, limit=args.limit, allow_noncommercial=args.allow_noncommercial
    )
    corpus.save_corpus(docs, out_dir=args.out_dir)

    total_chars = sum(len(d.full_text) for d in docs)
    licenses = Counter(d.license for d in docs)
    print(f"\n{len(docs)} articles, {total_chars:,} characters")
    print("licences:", dict(licenses))
    print(f"written to {args.out_dir}/corpus.jsonl and metadata.csv")


if __name__ == "__main__":
    main()
