"""Register the saved 75-question set as an MLflow evaluation dataset.

``scripts/build_eval_questions.py`` already registers the set it writes. This
script registers the saved file on its own, for example on a new MLflow server.
An existing dataset is kept unless ``--replace`` is given. Usage::

    uv run --group eval python scripts/register_eval_dataset.py
"""
from __future__ import annotations

import argparse
import os

from ocular.rag import eval_dataset, questions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tracking-uri", default=os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001"))
    parser.add_argument("--replace", action="store_true", help="Overwrite an existing dataset.")
    args = parser.parse_args()

    qs = questions.load_questions(questions.EVAL_DIR / "questions_full.jsonl")
    try:
        ds = eval_dataset.register(qs, args.tracking_uri, replace=args.replace)
    except FileExistsError as exc:
        raise SystemExit(str(exc))
    print(f"registered {eval_dataset.NAME!r} with {len(ds.to_df())} records")


if __name__ == "__main__":
    main()
